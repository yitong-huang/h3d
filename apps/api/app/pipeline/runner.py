"""Orchestrates parse → estimate → merge → assemble."""

from __future__ import annotations

import json
from pathlib import Path

from .factory import backend_name, create_stages
from .interfaces import ImageMeta


def run_pipeline(
    job_id: str,
    job_dir: Path,
    image_paths: list[Path],
    part_hints: dict[str, str | None] | None = None,
    update_status=None,
) -> dict:
    """Run the inference pipeline (real by default; H3D_PIPELINE=mock for stubs)."""
    part_hints = part_hints or {}
    backend = backend_name()

    def status(stage: str, progress: float, message: str) -> None:
        if update_status:
            update_status(stage, progress, message)

    images = [
        ImageMeta(path=p, filename=p.name, part_hint=part_hints.get(p.name))
        for p in image_paths
    ]

    parse, estimate, merge, assemble = create_stages()

    status(
        "parse",
        0.08,
        f"Parsing & routing ({backend}: MediaPipe/SegFormer/DensePose-IUV)",
    )
    routed = parse.parse_and_route(images)

    globals_ = [r for r in routed if r.route == "global"]
    locals_ = [r for r in routed if r.route == "local"]

    status(
        "estimate",
        0.35,
        f"Under-clothing SMPL-X estimation ({backend})",
    )
    body = estimate.estimate(globals_ or routed[:1])

    status("merge", 0.62, f"Local detail fusion PnP/Poisson ({backend})")
    details = merge.merge(locals_, body)

    status("assemble", 0.82, f"Assembling SMPL-X GLB ({backend})")
    asset = assemble.assemble(job_dir, body, details, routed)

    result = {
        "jobId": job_id,
        "backend": backend,
        "glbPath": str(asset.glb_path),
        "glbUrl": f"/api/jobs/{job_id}/avatar.glb",
        "meta": asset.meta,
        "stages": {
            "parse": {"routed": asset.meta.get("routed")},
            "estimate": {"beta": body.beta, "notes": body.notes},
            "merge": {
                "regions": details.fused_regions,
                "normalMap": details.normal_map_hint,
                "patches": details.uv_patches,
            },
            "assemble": {
                "vertexCount": asset.meta.get("vertexCount"),
                "bones": len(asset.meta.get("bones", [])),
                "topology": asset.meta.get("topology"),
            },
        },
    }
    (job_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    status("done", 1.0, f"Pipeline complete ({backend})")
    return result
