"""In-memory job store with disk persistence. Redis/Celery can replace this later."""

from __future__ import annotations

import json
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import JOB_DIR, UPLOAD_DIR
from .pipeline.runner import run_pipeline


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, dict[str, Any]] = {}

    def create(self, filenames: list[str], part_hints: dict[str, str | None]) -> dict:
        job_id = uuid.uuid4().hex[:12]
        job_dir = JOB_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        upload_dir = UPLOAD_DIR / job_id
        upload_dir.mkdir(parents=True, exist_ok=True)

        job = {
            "id": job_id,
            "status": "queued",
            "stage": "queued",
            "progress": 0.0,
            "message": "Queued",
            "createdAt": _now(),
            "updatedAt": _now(),
            "files": filenames,
            "partHints": part_hints,
            "error": None,
            "result": None,
        }
        (job_dir / "job.json").write_text(json.dumps(job, indent=2), encoding="utf-8")
        with self._lock:
            self._jobs[job_id] = job
        return job

    def get(self, job_id: str) -> dict | None:
        with self._lock:
            if job_id in self._jobs:
                return dict(self._jobs[job_id])
        path = JOB_DIR / job_id / "job.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return None

    def update(self, job_id: str, **fields: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                path = JOB_DIR / job_id / "job.json"
                if not path.exists():
                    return
                job = json.loads(path.read_text(encoding="utf-8"))
                self._jobs[job_id] = job
            job.update(fields)
            job["updatedAt"] = _now()
            (JOB_DIR / job_id / "job.json").write_text(
                json.dumps(job, indent=2), encoding="utf-8"
            )

    def start_inline(self, job_id: str) -> None:
        """Background thread mimicking a Celery worker."""

        def _run() -> None:
            try:
                self.update(
                    job_id, status="running", stage="parse", progress=0.05, message="Starting"
                )
                upload_dir = UPLOAD_DIR / job_id
                images = sorted(upload_dir.glob("*"))
                images = [p for p in images if p.is_file()]
                job = self.get(job_id) or {}
                part_hints = job.get("partHints") or {}

                def on_status(stage: str, progress: float, message: str) -> None:
                    status = "done" if stage == "done" else "running"
                    self.update(
                        job_id,
                        status=status,
                        stage=stage,
                        progress=progress,
                        message=message,
                    )

                result = run_pipeline(
                    job_id,
                    JOB_DIR / job_id,
                    images,
                    part_hints=part_hints,
                    update_status=on_status,
                )
                self.update(
                    job_id,
                    status="done",
                    stage="done",
                    progress=1.0,
                    message="Complete",
                    result=result,
                )
            except Exception as exc:  # noqa: BLE001 — surface to client
                self.update(
                    job_id,
                    status="error",
                    stage="error",
                    message=str(exc),
                    error=str(exc),
                )

        threading.Thread(target=_run, daemon=True).start()

    def glb_path(self, job_id: str) -> Path | None:
        p = JOB_DIR / job_id / "avatar.glb"
        return p if p.exists() else None

    def cleanup_old(self, keep: int = 50) -> None:
        jobs = sorted(JOB_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        for stale in jobs[keep:]:
            if stale.is_dir():
                shutil.rmtree(stale, ignore_errors=True)
            up = UPLOAD_DIR / stale.name
            if up.exists():
                shutil.rmtree(up, ignore_errors=True)


store = JobStore()
