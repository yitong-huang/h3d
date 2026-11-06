"""H3D FastAPI application."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import (
    ALLOWED_IMAGE_TYPES,
    JOB_DIR,
    MAX_IMAGES_PER_JOB,
    PIPELINE_BACKEND,
    UPLOAD_DIR,
    USE_INLINE_PIPELINE,
)
from .jobs import store
from .pipeline.factory import backend_name

app = FastAPI(title="H3D API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "service": "h3d-api",
        "backend": backend_name(),
        "pipelineMode": "inline" if USE_INLINE_PIPELINE else "celery",
        "env": {"H3D_PIPELINE": PIPELINE_BACKEND},
    }


@app.post("/api/jobs")
async def create_job(
    files: Annotated[list[UploadFile], File(description="1–N portrait photos")],
    part_hints: Annotated[str | None, Form()] = None,
) -> dict:
    if not files:
        raise HTTPException(400, "At least one image is required")
    if len(files) > MAX_IMAGES_PER_JOB:
        raise HTTPException(400, f"Max {MAX_IMAGES_PER_JOB} images per job")

    hints: dict[str, str | None] = {}
    if part_hints:
        try:
            hints = json.loads(part_hints)
        except json.JSONDecodeError as exc:
            raise HTTPException(400, f"Invalid part_hints JSON: {exc}") from exc

    # Validate + stage uploads after creating job id
    # Create job first to get id
    filenames: list[str] = []
    # Pre-read content
    payloads: list[tuple[str, bytes, str | None]] = []
    for f in files:
        content_type = f.content_type or "application/octet-stream"
        if content_type not in ALLOWED_IMAGE_TYPES and not (
            f.filename or ""
        ).lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            raise HTTPException(400, f"Unsupported type: {content_type} ({f.filename})")
        data = await f.read()
        if not data:
            raise HTTPException(400, f"Empty file: {f.filename}")
        name = Path(f.filename or "image.jpg").name
        payloads.append((name, data, hints.get(name)))
        filenames.append(name)

    job = store.create(filenames, {n: h for n, _, h in payloads})
    job_id = job["id"]
    upload_dir = UPLOAD_DIR / job_id
    for name, data, _ in payloads:
        (upload_dir / name).write_bytes(data)

    if USE_INLINE_PIPELINE:
        store.start_inline(job_id)
    else:
        # Placeholder for Celery: from .tasks import run_job.delay; run_job.delay(job_id)
        store.start_inline(job_id)

    return job


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    job = store.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@app.get("/api/jobs/{job_id}/avatar.glb")
def get_avatar(job_id: str) -> FileResponse:
    path = store.glb_path(job_id)
    if not path:
        raise HTTPException(404, "Avatar not ready")
    return FileResponse(
        path,
        media_type="model/gltf-binary",
        filename="avatar.glb",
    )


@app.get("/api/jobs/{job_id}/result")
def get_result(job_id: str) -> dict:
    job = store.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    if job.get("status") != "done":
        raise HTTPException(409, f"Job status is {job.get('status')}")
    result_path = JOB_DIR / job_id / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))
    return job.get("result") or {}


def _write_demo_glb(path: Path) -> dict:
    """Generate a real SMPL-X (or mock if forced) demo asset."""
    if backend_name() == "mock":
        from .assets.glb_generator import write_mock_glb

        return write_mock_glb(path)
    from .assets.smplx_glb import write_smplx_glb

    return write_smplx_glb(path, betas=[0.0] * 10)


@app.post("/api/demo/generate-glb")
def demo_generate_glb() -> dict:
    """Dev helper: write a sample GLB without uploading photos."""
    out = JOB_DIR / "_demo" / "avatar.glb"
    meta = _write_demo_glb(out)
    return {"glbUrl": "/api/demo/avatar.glb", "meta": meta, "backend": backend_name()}


@app.get("/api/demo/avatar.glb")
def demo_avatar() -> FileResponse:
    path = JOB_DIR / "_demo" / "avatar.glb"
    if not path.exists():
        _write_demo_glb(path)
    return FileResponse(path, media_type="model/gltf-binary", filename="avatar.glb")


# Optional: serve built web app if present
WEB_DIST = Path(__file__).resolve().parents[3] / "apps" / "web" / "dist"
if WEB_DIST.exists():
    app.mount("/", StaticFiles(directory=str(WEB_DIST), html=True), name="web")
