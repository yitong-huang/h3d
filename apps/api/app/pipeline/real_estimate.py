"""Under-clothing body estimation via real SMPL-X fitting (SHAPY-equivalent).

Pipeline:
1. Load MediaPipe 2D keypoints from parse artifacts
2. Optimize SMPL-X (global_orient, body_pose major joints, betas, transl, focal)
3. Clothing looseness compensation: shrink girth-related betas using SCHP cloth/skin ratio
   (open equivalent of SHAPY / BCNet under-clothing prior when SHAPY weights are absent)
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch

from ..assets.smplx_runtime import load_smplx, resolve_device
from .interfaces import BodyEstimate, RoutedImage, UnderClothingEstimator

# MediaPipe Pose landmark indices → SMPL-X joint indices used in fitting
MP_TO_SMPLX = {
    11: 16,  # L shoulder
    12: 17,  # R shoulder
    13: 18,  # L elbow
    14: 19,  # R elbow
    15: 20,  # L wrist
    16: 21,  # R wrist
    23: 1,  # L hip
    24: 2,  # R hip
    25: 4,  # L knee
    26: 5,  # R knee
    27: 7,  # L ankle
    28: 8,  # R ankle
    0: 15,  # nose → head proxy
}


def _load_pose(path) -> np.ndarray | None:
    p = path.parent / f"{path.stem}_pose.npy"
    if p.exists():
        return np.load(p)
    return None


def _load_schp(path) -> np.ndarray | None:
    p = path.parent / f"{path.stem}_schp.npy"
    if p.exists():
        return np.load(p)
    return None


def _skin_tone(bgr: np.ndarray, labels: np.ndarray | None) -> tuple[float, float, float]:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if labels is not None:
        mask = np.isin(labels, [11, 14, 15])  # face + arms
        if mask.mean() > 0.01:
            return tuple(float(x) for x in rgb[mask].mean(axis=0))  # type: ignore[return-value]
    # center crop fallback
    h, w = rgb.shape[:2]
    patch = rgb[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    return tuple(float(x) for x in patch.mean(axis=0))  # type: ignore[return-value]


def _project(joints3d: torch.Tensor, cam_t: torch.Tensor, focal: torch.Tensor, center: torch.Tensor):
    """Weak-perspective-ish pinhole projection."""
    j = joints3d + cam_t.view(1, 1, 3)
    x = j[..., 0] / (j[..., 2] + 1e-6) * focal + center[0]
    y = j[..., 1] / (j[..., 2] + 1e-6) * focal + center[1]
    return torch.stack([x, y], dim=-1)


def fit_smplx_from_keypoints(
    keypoints: np.ndarray,
    image_size: tuple[int, int],
    cloth_ratio: float,
    steps: int = 120,
) -> dict[str, Any]:
    """Differentiable SMPL-X fit on CPU/GPU from 2D keypoints."""
    device = resolve_device()
    model = load_smplx()
    w, h = image_size
    center = torch.tensor([w / 2, h / 2], device=device)

    # Build 2D targets
    targets = []
    joint_ids = []
    for mp_idx, smplx_idx in MP_TO_SMPLX.items():
        if mp_idx >= len(keypoints):
            continue
        x, y, vis = keypoints[mp_idx]
        if vis < 0.35:
            continue
        targets.append([x, y])
        joint_ids.append(smplx_idx)
    if len(targets) < 4:
        # Not enough evidence — return mean shape facing camera
        return {
            "betas": np.zeros(10, dtype=np.float32),
            "body_pose": np.zeros(63, dtype=np.float32),
            "global_orient": np.zeros(3, dtype=np.float32),
            "transl": np.array([0, 0.2, 2.5], dtype=np.float32),
            "loss": None,
            "notes": "fallback-mean: insufficient keypoints",
        }

    tgt = torch.tensor(targets, dtype=torch.float32, device=device)
    jidx = torch.tensor(joint_ids, dtype=torch.long, device=device)

    betas = torch.zeros(1, 10, device=device, requires_grad=True)
    # Initialize A-pose-ish arms down
    body_pose = torch.zeros(1, 63, device=device, requires_grad=True)
    global_orient = torch.zeros(1, 3, device=device, requires_grad=True)
    # camera translation in model space
    cam_t = torch.tensor([[0.0, 0.2, 2.4]], device=device, requires_grad=True)
    focal = torch.tensor([float(max(w, h))], device=device, requires_grad=True)

    opt = torch.optim.Adam([betas, body_pose, global_orient, cam_t, focal], lr=0.05)

    last = None
    for i in range(steps):
        opt.zero_grad()
        out = model(
            betas=betas,
            body_pose=body_pose,
            global_orient=global_orient,
            transl=torch.zeros(1, 3, device=device),
        )
        joints = out.joints[:, :55]
        proj = _project(joints, cam_t, focal, center)[0, jidx]
        loss_reproj = ((proj - tgt) ** 2).mean()
        # priors
        loss_beta = (betas**2).mean() * 0.01
        loss_pose = (body_pose**2).mean() * 0.001
        loss = loss_reproj + loss_beta + loss_pose
        loss.backward()
        opt.step()
        # constrain focal
        with torch.no_grad():
            focal.clamp_(0.6 * max(w, h), 2.5 * max(w, h))
            betas.clamp_(-3.0, 3.0)
        last = float(loss.item())

    betas_np = betas.detach().cpu().numpy().reshape(-1).astype(np.float32)

    # --- Under-clothing / looseness compensation (SHAPY-equivalent prior) ---
    # When clothing dominates skin on the body, reduce girth-related shape dims.
    # SMPL-X beta[1], beta[2] correlate strongly with overall weight / waist.
    if cloth_ratio > 0.12:
        shrink = min(0.85, 0.25 + cloth_ratio)
        betas_np[1] -= shrink * 0.6
        betas_np[2] -= shrink * 0.45
        betas_np[0] *= 0.95
        notes = (
            f"smplx-fit+undercloth: cloth_ratio={cloth_ratio:.3f}, "
            f"girth_shrink={shrink:.2f}, loss={last}"
        )
    else:
        notes = f"smplx-fit: cloth_ratio={cloth_ratio:.3f}, loss={last}"

    return {
        "betas": betas_np,
        "body_pose": body_pose.detach().cpu().numpy().reshape(-1).astype(np.float32),
        "global_orient": global_orient.detach().cpu().numpy().reshape(-1).astype(np.float32),
        "transl": cam_t.detach().cpu().numpy().reshape(-1).astype(np.float32),
        "focal": float(focal.detach().cpu().item()),
        "loss": last,
        "notes": notes,
    }


class RealUnderClothingEstimator(UnderClothingEstimator):
    def estimate(self, globals_: list[RoutedImage]) -> BodyEstimate:
        if not globals_:
            raise ValueError("No global image for body estimation")
        primary = max(globals_, key=lambda r: r.coverage.get("body_frac", 0.0))
        bgr = cv2.imread(str(primary.path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"Cannot read {primary.path}")
        h, w = bgr.shape[:2]
        pose = _load_pose(primary.path)
        if pose is None:
            # re-run pose quickly
            from .real_parse import run_pose

            pose_dict = run_pose(bgr)
            kps = pose_dict["keypoints"]
            pose = np.array(
                [[p["x"], p["y"], p["visibility"]] for p in kps], dtype=np.float32
            ) if kps else np.zeros((0, 3), dtype=np.float32)

        labels = _load_schp(primary.path)
        cloth_ratio = float(primary.skin_mask_summary.get("clothes", 0.2))
        fit = fit_smplx_from_keypoints(pose, (w, h), cloth_ratio=cloth_ratio)
        tone = _skin_tone(bgr, labels)

        # Pack full pose into theta_hint (legacy) + notes; extras via BodyEstimate fields
        theta = fit["body_pose"][:16].tolist()
        return BodyEstimate(
            beta=fit["betas"].tolist(),
            theta_hint=theta,
            skin_tone_rgb=tone,
            notes=fit["notes"],
            full_params={
                "betas": fit["betas"],
                "body_pose": fit["body_pose"],
                "global_orient": fit["global_orient"],
                "transl": fit["transl"],
                "focal": fit.get("focal"),
                "source_image": str(primary.path),
                "keypoints": pose,
            },
        )
