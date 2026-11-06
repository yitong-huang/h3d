"""Mock / stub GPU stages that produce demo-usable outputs without model weights."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from .interfaces import (
    AssembledAsset,
    AssetAssembler,
    BodyEstimate,
    DetailMergeResult,
    ImageMeta,
    ImageParseRouter,
    LocalDetailMerger,
    RoutedImage,
    UnderClothingEstimator,
)
from ..assets.glb_generator import write_mock_glb


PART_KEYWORDS = {
    "face": "Head",
    "head": "Head",
    "chest": "Torso",
    "breast": "Torso",
    "torso": "Torso",
    "arm": "Arms_Lower",
    "forearm": "Arms_Lower",
    "hand": "Hands",
    "tattoo": "Arms_Lower",
    "thigh": "Thighs",
    "leg": "Calves",
    "foot": "Feet",
    "back": "Torso",
    "abdomen": "Torso",
}


class MockImageParseRouter(ImageParseRouter):
    """Stub for YOLOv8-Pose + DensePose + SCHP routing."""

    def parse_and_route(self, images: list[ImageMeta]) -> list[RoutedImage]:
        routed: list[RoutedImage] = []
        for img in images:
            w, h = 512, 512
            try:
                with Image.open(img.path) as im:
                    w, h = im.size
            except Exception:
                pass

            aspect = w / max(h, 1)
            # Heuristic: wide / tall full-body-ish → global; square crop → local
            is_local = False
            if img.part_hint:
                is_local = True
            elif aspect > 0.85 and aspect < 1.15 and min(w, h) >= 400:
                # square close-up guess
                is_local = True
            elif h < 400 or w < 300:
                is_local = True

            # Prefer first large image as global if none yet
            if not any(r.route == "global" for r in routed) and not img.part_hint:
                is_local = False

            part_label = img.part_hint
            if not part_label and is_local:
                name_l = img.filename.lower()
                for key, label in PART_KEYWORDS.items():
                    if key in name_l:
                        part_label = label
                        break
                part_label = part_label or "Torso"

            routed.append(
                RoutedImage(
                    path=img.path,
                    filename=img.filename,
                    route="local" if is_local else "global",
                    coverage={
                        "body_frac": 0.35 if is_local else 0.78,
                        "width": float(w),
                        "height": float(h),
                    },
                    part_label=part_label,
                    skin_mask_summary={
                        "skin": 0.42 if is_local else 0.28,
                        "clothes": 0.15 if is_local else 0.55,
                        "background": 0.43 if is_local else 0.17,
                    },
                )
            )

        if not any(r.route == "global" for r in routed) and routed:
            routed[0].route = "global"
            routed[0].part_label = None
        return routed


class MockUnderClothingEstimator(UnderClothingEstimator):
    """Stub for SHAPY / BCNet — emits stable β from image hash."""

    def estimate(self, globals_: list[RoutedImage]) -> BodyEstimate:
        seed_src = "|".join(sorted(r.filename for r in globals_)) or "default"
        digest = hashlib.sha256(seed_src.encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        beta = (rng.normal(0, 0.35, size=10)).clip(-2.0, 2.0).tolist()
        theta = [0.0] * 16

        # Approximate skin tone from first global image mean color
        rgb = (0.78, 0.62, 0.52)
        if globals_:
            try:
                with Image.open(globals_[0].path) as im:
                    im = im.convert("RGB").resize((64, 64))
                    arr = np.asarray(im, dtype=np.float32) / 255.0
                    rgb = tuple(float(x) for x in arr.mean(axis=(0, 1)))
            except Exception:
                pass

        return BodyEstimate(
            beta=beta,
            theta_hint=theta,
            skin_tone_rgb=rgb,  # type: ignore[arg-type]
            notes="mock-shapy: clothing looseness compensated via prior",
        )


class MockLocalDetailMerger(LocalDetailMerger):
    """Stub for DensePose PnP + Poisson blend + normal bake."""

    def merge(
        self, locals_: list[RoutedImage], body: BodyEstimate
    ) -> DetailMergeResult:
        patches = []
        regions: list[str] = []
        for r in locals_:
            region = r.part_label or "Torso"
            regions.append(region)
            patches.append(
                {
                    "source": r.filename,
                    "region": region,
                    "camera": {"R": [1, 0, 0, 0, 1, 0, 0, 0, 1], "T": [0, 0, 1.2], "f": 35},
                    "blend": "poisson-mock",
                    "skin_tone_anchor": body.skin_tone_rgb,
                }
            )
        return DetailMergeResult(
            uv_patches=patches,
            normal_map_hint="height-from-shading-mock",
            fused_regions=regions,
        )


class MockAssetAssembler(AssetAssembler):
    """Assembles a contract-compliant mock GLB from estimate + detail metadata."""

    def assemble(
        self,
        job_dir: Path,
        body: BodyEstimate,
        details: DetailMergeResult,
        routed: list[RoutedImage],
    ) -> AssembledAsset:
        out = job_dir / "avatar.glb"
        meta = write_mock_glb(
            out,
            beta=body.beta,
            skin_tone=body.skin_tone_rgb,
            detail_regions=details.fused_regions,
        )
        meta.update(
            {
                "routed": [
                    {"file": r.filename, "route": r.route, "part": r.part_label}
                    for r in routed
                ],
                "beta": body.beta,
                "skin_tone_rgb": list(body.skin_tone_rgb),
                "detail_patches": details.uv_patches,
                "pipeline": "mock",
            }
        )
        return AssembledAsset(glb_path=out, meta=meta)
