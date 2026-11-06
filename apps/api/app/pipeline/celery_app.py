"""Celery scaffolding (optional). Dev default uses inline threads in jobs.py.

Enable later with:
  CELERY_BROKER_URL=redis://localhost:6379/0
  and set USE_INLINE_PIPELINE=False in config.
"""

from __future__ import annotations

try:
    from celery import Celery
except ImportError:  # pragma: no cover
    Celery = None  # type: ignore

celery_app = None
if Celery is not None:
    celery_app = Celery(
        "h3d",
        broker="redis://localhost:6379/0",
        backend="redis://localhost:6379/1",
    )
    celery_app.conf.task_routes = {"h3d.run_job": {"queue": "gpu"}}


def run_job(job_id: str) -> dict:
    """Celery task body — importable as shared entrypoint."""
    from ..config import JOB_DIR, UPLOAD_DIR
    from ..jobs import store
    from .runner import run_pipeline

    images = sorted(p for p in (UPLOAD_DIR / job_id).glob("*") if p.is_file())
    job = store.get(job_id) or {}

    def on_status(stage: str, progress: float, message: str) -> None:
        store.update(
            job_id,
            status="done" if stage == "done" else "running",
            stage=stage,
            progress=progress,
            message=message,
        )

    return run_pipeline(
        job_id,
        JOB_DIR / job_id,
        images,
        part_hints=job.get("partHints") or {},
        update_status=on_status,
    )


if celery_app is not None:

    @celery_app.task(name="h3d.run_job")
    def run_job_task(job_id: str) -> dict:
        return run_job(job_id)
