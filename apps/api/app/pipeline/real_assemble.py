"""Assemble real SMPL-X mesh → H3D GLB."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..assets.smplx_glb import write_smplx_glb
from .interfaces import (
    AssembledAsset,
    AssetAssembler,
    BodyEstimate,
    DetailMergeResult,
    RoutedImage,
)


def _sample_vertex_colors(
    uv_path: str | None,
    n_verts: int,
    skin_tone: tuple[float, float, float],
) -> np.ndarray | None:
    """Approximate per-vertex colors from UV atlas (nearest)."""
    if not uv_path or not Path(uv_path).exists():
        return None
    # Full UV→vertex sampling needs vt/ft; for MVP tint by atlas mean + skin
    img = cv2.imread(uv_path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    mean = img.reshape(-1, 3).mean(axis=0)[::-1] / 255.0  # BGR→RGB
    colors = np.tile(np.array([*mean], dtype=np.float32), (n_verts, 1))
    # Mix with estimated skin tone for stability
    colors = 0.65 * colors + 0.35 * np.array(skin_tone, dtype=np.float32)
    return colors


class RealAssetAssembler(AssetAssembler):
    def assemble(
        self,
        job_dir: Path,
        body: BodyEstimate,
        details: DetailMergeResult,
        routed: list[RoutedImage],
    ) -> AssembledAsset:
        params = body.full_params or {}
        out = job_dir / "avatar.glb"
        basecolor = _sample_vertex_colors(
            params.get("uv_basecolor"), 10475, body.skin_tone_rgb
        )
        meta = write_smplx_glb(
            out,
            betas=params.get("betas", body.beta),
            body_pose=params.get("body_pose"),
            global_orient=params.get("global_orient"),
            transl=None,  # bake pose without camera translation for asset origin
            skin_tone=body.skin_tone_rgb,
            basecolor=basecolor,
            detail_regions=details.fused_regions,
            extra_meta={
                "pipeline": "real",
                "estimateNotes": body.notes,
                "normalMap": details.normal_map_hint,
            },
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
                "pipeline": "real",
                "uvBasecolor": params.get("uv_basecolor"),
                "uvNormal": params.get("uv_normal"),
            }
        )
        return AssembledAsset(glb_path=out, meta=meta)
