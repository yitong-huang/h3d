"""Orchestrates parse → estimate → merge → assemble. Celery-ready; inline by default."""

from __future__ import annotations

import json
import time
from pathlib import Path

from .interfaces import ImageMeta
from .mocks import (
    MockAssetAssembler,
    MockImageParseRouter,
    MockLocalDetailMerger,
    MockUnderClothingEstimator,
)


def run_pipeline(
    job_id: str,
    job_dir: Path,
    image_paths: list[Path],
    part_hints: dict[str, str | None] | None = None,
    update_status=None,
) -> dict:
    """Run the full inference pipeline with mock GPU stages.

    `update_status(stage, progress, message)` is optional for job store updates.
    """
    part_hints = part_hints or {}

    def status(stage: str, progress: float, message: str) -> None:
        if update_status:
            update_status(stage, progress, message)

    images = [
        ImageMeta(path=p, filename=p.name, part_hint=part_hints.get(p.name))
        for p in image_paths
    ]

    status("parse", 0.1, "Parsing & routing images (DensePose/SCHP stub)")
    time.sleep(0.15)
    routed = MockImageParseRouter().parse_and_route(images)

    globals_ = [r for r in routed if r.route == "global"]
    locals_ = [r for r in routed if r.route == "local"]

    status("estimate", 0.35, "Under-clothing body estimation (SHAPY stub)")
    time.sleep(0.2)
    body = MockUnderClothingEstimator().estimate(globals_ or routed[:1])

    status("merge", 0.6, "Local detail fusion (PnP/Poisson stub)")
    time.sleep(0.15)
    details = MockLocalDetailMerger().merge(locals_, body)

    status("assemble", 0.8, "Assembling standardized GLB")
    time.sleep(0.1)
    asset = MockAssetAssembler().assemble(job_dir, body, details, routed)

    result = {
        "jobId": job_id,
        "glbPath": str(asset.glb_path),
        "glbUrl": f"/api/jobs/{job_id}/avatar.glb",
        "meta": asset.meta,
        "stages": {
            "parse": {"routed": asset.meta.get("routed")},
            "estimate": {"beta": body.beta, "notes": body.notes},
            "merge": {
                "regions": details.fused_regions,
                "normalMap": details.normal_map_hint,
            },
            "assemble": {
                "vertexCount": asset.meta.get("vertexCount"),
                "bones": len(asset.meta.get("bones", [])),
            },
        },
    }
    (job_dir / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    status("done", 1.0, "Pipeline complete")
    return result
