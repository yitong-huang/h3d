"""Local detail fusion: DensePose-style IUV + PnP + Poisson blend + normals."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch

from ..assets.smplx_runtime import load_smplx, resolve_device, smplx_faces
from .interfaces import BodyEstimate, DetailMergeResult, LocalDetailMerger, RoutedImage


def _smplx_vertices(body: BodyEstimate) -> np.ndarray:
    model = load_smplx()
    device = resolve_device()
    p = body.full_params or {}

    def t(key, shape, default=0.0):
        val = p.get(key)
        if val is None:
            return torch.full((1, *shape), default, device=device)
        return torch.tensor(np.asarray(val, dtype=np.float32).reshape(1, *shape), device=device)

    with torch.no_grad():
        out = model(
            betas=t("betas", (10,)),
            body_pose=t("body_pose", (63,)),
            global_orient=t("global_orient", (3,)),
            transl=torch.zeros(1, 3, device=device),
        )
        return out.vertices[0].detach().cpu().numpy().astype(np.float32)


def _build_iuv(vertices: np.ndarray, faces: np.ndarray, image_size: tuple[int, int],
               cam_t: np.ndarray, focal: float) -> dict[str, np.ndarray]:
    """Render a coarse DensePose-style IUV map by projecting SMPL-X triangles."""
    w, h = image_size
    i_map = np.zeros((h, w), dtype=np.int32)
    u_map = np.zeros((h, w), dtype=np.float32)
    v_map = np.zeros((h, w), dtype=np.float32)
    z_buf = np.full((h, w), np.inf, dtype=np.float32)

    # Project vertices
    j = vertices + cam_t.reshape(1, 3)
    x = j[:, 0] / (j[:, 2] + 1e-6) * focal + w / 2
    y = j[:, 1] / (j[:, 2] + 1e-6) * focal + h / 2
    pts = np.stack([x, y], axis=1)

    # Sample subset of faces for speed on CPU
    step = max(1, len(faces) // 4000)
    for fi, face in enumerate(faces[::step]):
        tri = pts[face]
        zs = j[face, 2]
        if np.any(zs <= 1e-3):
            continue
        # bbox
        minx, miny = np.floor(tri.min(axis=0)).astype(int)
        maxx, maxy = np.ceil(tri.max(axis=0)).astype(int)
        minx, miny = max(minx, 0), max(miny, 0)
        maxx, maxy = min(maxx, w - 1), min(maxy, h - 1)
        if minx >= maxx or miny >= maxy:
            continue
        # barycentric fill
        for py in range(miny, maxy + 1):
            for px in range(minx, maxx + 1):
                p = np.array([px + 0.5, py + 0.5])
                v0, v1, v2 = tri
                den = (v1[1] - v2[1]) * (v0[0] - v2[0]) + (v2[0] - v1[0]) * (v0[1] - v2[1])
                if abs(den) < 1e-8:
                    continue
                a = ((v1[1] - v2[1]) * (p[0] - v2[0]) + (v2[0] - v1[0]) * (p[1] - v2[1])) / den
                b = ((v2[1] - v0[1]) * (p[0] - v2[0]) + (v0[0] - v2[0]) * (p[1] - v2[1])) / den
                c = 1 - a - b
                if a < 0 or b < 0 or c < 0:
                    continue
                z = a * zs[0] + b * zs[1] + c * zs[2]
                if z < z_buf[py, px]:
                    z_buf[py, px] = z
                    i_map[py, px] = (fi % 24) + 1  # part-ish id
                    u_map[py, px] = a
                    v_map[py, px] = b
    return {"I": i_map, "U": u_map, "V": v_map, "Z": z_buf}


def _solve_pnp(vertices: np.ndarray, keypoints: np.ndarray, image_size: tuple[int, int],
               focal: float) -> dict[str, Any]:
    """PnP from SMPL-X 3D joints/verts vs 2D detections (local camera)."""
    w, h = image_size
    # Use a stable subset of vertices near torso as 3D-2D if keypoints sparse
    cam_matrix = np.array(
        [[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1]], dtype=np.float64
    )
    dist = np.zeros(5)
    # Sample 32 vertices with corresponding projected priors as init — use
    # OpenCV solvePnPRansac with synthetic 2D from center of image local crop.
    idx = np.linspace(0, len(vertices) - 1, 40).astype(int)
    obj = vertices[idx].astype(np.float64)
    # For local crops without reliable dense matches, approximate camera looking
    # at mesh centroid.
    center3d = vertices.mean(axis=0)
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.array([[0.0], [0.0], [float(np.linalg.norm(center3d) + 1.2)]], dtype=np.float64)
    if keypoints is not None and len(keypoints) >= 6:
        # Use available 2D points mapped loosely via nearest projected verts
        j = vertices + np.array([0, 0.2, 2.4])
        proj_x = j[:, 0] / (j[:, 2] + 1e-6) * focal + w / 2
        proj_y = j[:, 1] / (j[:, 2] + 1e-6) * focal + h / 2
        img_pts = []
        obj_pts = []
        for kp in keypoints:
            x, y = float(kp[0]), float(kp[1])
            vis = float(kp[2]) if len(kp) > 2 else 1.0
            if vis < 0.3:
                continue
            d = (proj_x - x) ** 2 + (proj_y - y) ** 2
            vi = int(np.argmin(d))
            if d[vi] < (0.15 * max(w, h)) ** 2:
                img_pts.append([x, y])
                obj_pts.append(vertices[vi])
        if len(img_pts) >= 4:
            ok, rvec, tvec, _ = cv2.solvePnPRansac(
                np.asarray(obj_pts, dtype=np.float64),
                np.asarray(img_pts, dtype=np.float64),
                cam_matrix,
                dist,
                flags=cv2.SOLVEPNP_EPNP,
            )
            if not ok:
                rvec = np.zeros((3, 1))
                tvec = np.array([[0.0], [0.0], [2.0]])
    R, _ = cv2.Rodrigues(rvec)
    return {
        "R": R.reshape(-1).tolist(),
        "T": tvec.reshape(-1).tolist(),
        "f": float(focal),
        "K": cam_matrix.reshape(-1).tolist(),
    }


def _poisson_blend_to_uv(
    local_bgr: np.ndarray,
    base_uv: np.ndarray,
    region: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Blend local detail into a UV canvas with OpenCV Poisson seamlessClone."""
    h, w = base_uv.shape[:2]
    # Place local patch in region-specific UV window
    region_boxes = {
        "Head": (0.35, 0.05, 0.65, 0.25),
        "Torso": (0.30, 0.25, 0.70, 0.55),
        "Arms_Lower": (0.05, 0.35, 0.25, 0.65),
        "Arms_Upper": (0.05, 0.20, 0.25, 0.40),
        "Hands": (0.00, 0.60, 0.20, 0.80),
        "Thighs": (0.30, 0.55, 0.70, 0.80),
        "Calves": (0.30, 0.80, 0.70, 0.95),
        "Feet": (0.30, 0.90, 0.70, 1.00),
    }
    box = region_boxes.get(region or "Torso", region_boxes["Torso"])
    x0, y0, x1, y1 = [int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)]
    bw, bh = max(x1 - x0, 8), max(y1 - y0, 8)
    patch = cv2.resize(local_bgr, (bw, bh), interpolation=cv2.INTER_AREA)
    # Mask: non-near-black
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    mask = (gray > 15).astype(np.uint8) * 255
    center = (x0 + bw // 2, y0 + bh // 2)
    try:
        blended = cv2.seamlessClone(patch, base_uv, mask, center, cv2.NORMAL_CLONE)
    except cv2.error:
        blended = base_uv.copy()
        roi = blended[y0:y0 + bh, x0:x0 + bw]
        alpha = (mask > 0).astype(np.float32)[..., None] / 255.0
        if roi.shape[:2] == patch.shape[:2]:
            blended[y0:y0 + bh, x0:x0 + bw] = (
                roi.astype(np.float32) * (1 - alpha) + patch.astype(np.float32) * alpha
            ).astype(np.uint8)

    # Height-from-shading → normal map (tangent-ish RGB)
    g = cv2.cvtColor(blended, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    normal = np.stack([-gx, -gy, np.ones_like(g)], axis=-1)
    n = np.linalg.norm(normal, axis=-1, keepdims=True) + 1e-6
    normal = (normal / n + 1) * 0.5
    normal_u8 = (normal * 255).astype(np.uint8)
    return blended, normal_u8


class RealLocalDetailMerger(LocalDetailMerger):
    def merge(
        self, locals_: list[RoutedImage], body: BodyEstimate
    ) -> DetailMergeResult:
        params = body.full_params or {}
        cam_t = np.asarray(params.get("transl", [0, 0.2, 2.4]), dtype=np.float32)
        focal = float(params.get("focal") or 1000.0)
        vertices = _smplx_vertices(body)
        faces = smplx_faces()

        # Base UV canvas from skin tone
        tone = body.skin_tone_rgb
        base = np.zeros((1024, 1024, 3), dtype=np.uint8)
        base[:] = (
            int(tone[2] * 255),
            int(tone[1] * 255),
            int(tone[0] * 255),
        )  # BGR

        patches: list[dict[str, Any]] = []
        regions: list[str] = []
        normal_map = np.full((1024, 1024, 3), 128, dtype=np.uint8)
        normal_map[..., 2] = 255

        for r in locals_:
            bgr = cv2.imread(str(r.path), cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            h, w = bgr.shape[:2]
            pose_path = r.path.parent / f"{r.path.stem}_pose.npy"
            kps = np.load(pose_path) if pose_path.exists() else np.zeros((0, 3), np.float32)
            iuv = _build_iuv(vertices, faces, (w, h), cam_t, focal)
            cam = _solve_pnp(vertices, kps, (w, h), focal)
            region = r.part_label or "Torso"
            regions.append(region)
            base, normal_map = _poisson_blend_to_uv(bgr, base, region)
            patches.append(
                {
                    "source": r.filename,
                    "region": region,
                    "camera": cam,
                    "blend": "opencv-seamlessClone-poisson",
                    "iuv_coverage": float((iuv["I"] > 0).mean()),
                    "densepose": "smplx_projected_iuv",
                }
            )

        # Stash textures on body for assembler
        tex_dir = None
        if params.get("source_image"):
            from pathlib import Path

            tex_dir = Path(params["source_image"]).parent
        elif locals_:
            tex_dir = locals_[0].path.parent
        if tex_dir is not None:
            cv2.imwrite(str(tex_dir / "uv_basecolor.png"), base)
            cv2.imwrite(str(tex_dir / "uv_normal.png"), normal_map)

        if tex_dir is not None:
            params["uv_basecolor"] = str(tex_dir / "uv_basecolor.png")
            params["uv_normal"] = str(tex_dir / "uv_normal.png")
            body.full_params = params

        return DetailMergeResult(
            uv_patches=patches,
            normal_map_hint="sobel-height-from-shading",
            fused_regions=regions,
        )
