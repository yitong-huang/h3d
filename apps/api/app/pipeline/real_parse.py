"""Real image parsing / routing: MediaPipe Pose + SegFormer clothes (SCHP-equiv).

Dense correspondence: SegFormer body-part labels + optional SMPL-X IUV refinement
(`H3D_DENSEPOSE=smplx_iuv`, default). Set `H3D_DENSEPOSE=detectron2` when Detectron2
DensePose is installed to swap in the classic DensePose head.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import cv2
import numpy as np
from PIL import Image

from ..config import MEDIAPIPE_POSE_TASK, SCHP_MODEL_ID
from .interfaces import ImageMeta, ImageParseRouter, RoutedImage

# SegFormer clothes labels (mattmdjaga/segformer_b2_clothes)
CLOTH_LABELS = {4, 5, 6, 7, 8, 16, 17}  # upper, skirt, pants, dress, belt, bag, scarf
SKIN_LABELS = {11, 12, 13, 14, 15}  # face, legs, arms
FACE_LABELS = {2, 11}  # hair, face

PART_FROM_HINT = {
    "head": "Head",
    "face": "Head",
    "torso": "Torso",
    "chest": "Torso",
    "breast": "Torso",
    "arm": "Arms_Lower",
    "arms_lower": "Arms_Lower",
    "arms_upper": "Arms_Upper",
    "forearm": "Arms_Lower",
    "hand": "Hands",
    "hands": "Hands",
    "thigh": "Thighs",
    "thighs": "Thighs",
    "leg": "Calves",
    "calves": "Calves",
    "foot": "Feet",
    "feet": "Feet",
    "back": "Torso",
    "abdomen": "Torso",
}


@lru_cache(maxsize=1)
def _segformer():
    import torch
    from transformers import SegformerForSemanticSegmentation

    from ..assets.smplx_runtime import resolve_device

    device = resolve_device()
    model = SegformerForSemanticSegmentation.from_pretrained(SCHP_MODEL_ID)
    model.to(device)
    model.eval()
    return model, device


@lru_cache(maxsize=1)
def _pose_landmarker():
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    if not MEDIAPIPE_POSE_TASK.exists():
        raise FileNotFoundError(
            f"MediaPipe pose model missing: {MEDIAPIPE_POSE_TASK}. "
            "Run scripts/fetch_models.py"
        )
    base = mp_python.BaseOptions(model_asset_path=str(MEDIAPIPE_POSE_TASK))
    opts = vision.PoseLandmarkerOptions(
        base_options=base,
        num_poses=1,
        output_segmentation_masks=True,
        min_pose_detection_confidence=0.4,
        min_pose_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    return vision.PoseLandmarker.create_from_options(opts)


def run_schp(image_bgr: np.ndarray) -> np.ndarray:
    """Return HxW int label map (SCHP-equivalent SegFormer clothes parsing)."""
    import torch
    import torch.nn.functional as F

    model, device = _segformer()
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    # Processor-free preprocess (ImageNet norm, 512)
    resized = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_LINEAR)
    x = resized.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    x = (x - mean) / std
    t = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(pixel_values=t).logits
        up = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
        labels = up.argmax(dim=1)[0].detach().cpu().numpy().astype(np.int32)
    return labels


def run_pose(image_bgr: np.ndarray) -> dict[str, Any]:
    """MediaPipe Pose landmarks + optional segmentation mask."""
    import mediapipe as mp

    landmarker = _pose_landmarker()
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
    result = landmarker.detect(mp_img)
    h, w = rgb.shape[:2]
    kps = []
    if result.pose_landmarks:
        for lm in result.pose_landmarks[0]:
            kps.append(
                {
                    "x": lm.x * w,
                    "y": lm.y * h,
                    "z": lm.z,
                    "visibility": getattr(lm, "visibility", 1.0) or 1.0,
                }
            )
    seg = None
    if result.segmentation_masks:
        seg = result.segmentation_masks[0].numpy_view()
        if seg.ndim == 3:
            seg = seg[..., 0]
        seg = (seg > 0.5).astype(np.uint8)
    return {"keypoints": kps, "seg": seg, "size": (w, h)}


def _body_coverage(kps: list[dict], w: int, h: int) -> float:
    if len(kps) < 8:
        return 0.0
    ys = [p["y"] for p in kps if p.get("visibility", 1) > 0.3]
    xs = [p["x"] for p in kps if p.get("visibility", 1) > 0.3]
    if not ys:
        return 0.0
    return float((max(ys) - min(ys)) / max(h, 1) * (max(xs) - min(xs)) / max(w, 1))


def _infer_local_part(labels: np.ndarray) -> str:
    counts = np.bincount(labels.reshape(-1), minlength=18)
    # ignore background
    counts[0] = 0
    # Prefer extremities for local shots
    if counts[14] + counts[15] > counts[11] and counts[14] + counts[15] > counts.sum() * 0.15:
        return "Arms_Lower"
    if counts[12] + counts[13] > counts.sum() * 0.2:
        return "Thighs"
    if counts[9] + counts[10] > counts.sum() * 0.05:
        return "Feet"
    if counts[11] > counts.sum() * 0.15 and counts[4] < counts[11]:
        return "Head"
    return "Torso"


class RealImageParseRouter(ImageParseRouter):
    def parse_and_route(self, images: list[ImageMeta]) -> list[RoutedImage]:
        routed: list[RoutedImage] = []
        for img in images:
            bgr = cv2.imread(str(img.path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise ValueError(f"Cannot read image: {img.path}")
            h, w = bgr.shape[:2]
            labels = run_schp(bgr)
            pose = run_pose(bgr)
            kps = pose["keypoints"]
            coverage = _body_coverage(kps, w, h)

            skin = float(np.isin(labels, list(SKIN_LABELS)).mean())
            cloth = float(np.isin(labels, list(CLOTH_LABELS)).mean())
            bg = float((labels == 0).mean())

            hint = (img.part_hint or "").strip().lower()
            part_label = PART_FROM_HINT.get(hint.replace(" ", "_")) if hint else None

            # DensePose-style routing signal: keypoint span + clothing/skin area
            is_local = False
            if part_label:
                is_local = True
            elif coverage < 0.08 and max(w, h) >= 256:
                is_local = True
            elif len(kps) < 6:
                is_local = True
            elif coverage < 0.15 and skin > 0.25 and cloth < 0.15:
                is_local = True

            if not part_label and is_local:
                part_label = _infer_local_part(labels)

            # Persist parse artifacts beside the image for later stages
            art_dir = img.path.parent
            np.save(art_dir / f"{img.path.stem}_schp.npy", labels)
            if kps:
                np.save(
                    art_dir / f"{img.path.stem}_pose.npy",
                    np.array(
                        [[p["x"], p["y"], p["visibility"]] for p in kps],
                        dtype=np.float32,
                    ),
                )

            routed.append(
                RoutedImage(
                    path=img.path,
                    filename=img.filename,
                    route="local" if is_local else "global",
                    coverage={
                        "body_frac": coverage,
                        "width": float(w),
                        "height": float(h),
                        "num_keypoints": float(len(kps)),
                    },
                    part_label=part_label,
                    skin_mask_summary={
                        "skin": skin,
                        "clothes": cloth,
                        "background": bg,
                    },
                )
            )

        if routed and not any(r.route == "global" for r in routed):
            # Prefer the image with highest keypoint coverage as global
            best = max(routed, key=lambda r: r.coverage.get("body_frac", 0.0))
            best.route = "global"
            best.part_label = None
        return routed
