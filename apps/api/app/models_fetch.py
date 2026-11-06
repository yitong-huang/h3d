"""Shared weight download / presence checks for the real H3D pipeline.

Used by FastAPI startup (auto-fetch) and `scripts/fetch_models.py` (offline/CI).
"""

from __future__ import annotations

import logging
import shutil
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .config import MEDIAPIPE_POSE_TASK, MODEL_DIR, SCHP_MODEL_ID, SMPLX_MODEL_DIR

logger = logging.getLogger("h3d.models")

LogFn = Callable[[str], None]


@dataclass
class FetchReport:
    checked: list[str] = field(default_factory=list)
    present: list[str] = field(default_factory=list)
    downloaded: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def _log(msg: str, log: LogFn | None = None) -> None:
    if log:
        log(msg)
    else:
        logger.info(msg)


def smplx_pkl_path() -> Path:
    root = SMPLX_MODEL_DIR
    candidates = [
        root / "smplx" / "SMPLX_NEUTRAL.pkl",
        root / "SMPLX_NEUTRAL.pkl",
    ]
    for c in candidates:
        if c.exists():
            return c
    return root / "smplx" / "SMPLX_NEUTRAL.pkl"


def mediapipe_task_path() -> Path:
    return MEDIAPIPE_POSE_TASK


def smplx_present() -> bool:
    p = smplx_pkl_path()
    return p.exists() and p.stat().st_size > 1_000_000


def mediapipe_present() -> bool:
    p = mediapipe_task_path()
    return p.exists() and p.stat().st_size > 100_000


def schp_cached() -> bool:
    """Best-effort: true if HF cache already has the SegFormer weights."""
    try:
        from huggingface_hub import try_to_load_from_cache
        from huggingface_hub.constants import CONFIG_NAME

        path = try_to_load_from_cache(SCHP_MODEL_ID, CONFIG_NAME)
        if path is None:
            return False
        # path may be a sentinel object when known-missing; only accept real files
        p = Path(str(path))
        return p.exists() and p.is_file()
    except Exception:
        pass
    try:
        from huggingface_hub import constants

        hub = Path(constants.HF_HUB_CACHE)
        safe = "models--" + SCHP_MODEL_ID.replace("/", "--")
        snap = hub / safe / "snapshots"
        return snap.exists() and any(snap.iterdir())
    except Exception:
        return False


def fetch_smplx(*, log: LogFn | None = None) -> str:
    """Ensure SMPL-X neutral pkl exists. Returns status: present|downloaded."""
    target = smplx_pkl_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    if smplx_present():
        _log(f"[ok] SMPL-X already present: {target}", log)
        return "present"

    _log(f"[fetch] Downloading SMPLX_NEUTRAL.pkl (≈520MB) → {target}", log)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency huggingface_hub. Install: pip install huggingface_hub"
        ) from exc

    staging = MODEL_DIR / "smplx" / "_hf"
    staging.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id="stran9682/smplx_model",
        filename="SMPLX_NEUTRAL.pkl",
        local_dir=str(staging),
    )
    shutil.copy2(path, target)
    if not smplx_present():
        raise RuntimeError(f"SMPL-X download finished but file invalid: {target}")
    _log(f"[ok] SMPL-X ready: {target} ({target.stat().st_size} bytes)", log)
    _log(
        "[note] SMPL-X © Max Planck Institute — register at "
        "https://smpl-x.is.tue.mpg.de/ before redistributing / commercial use.",
        log,
    )
    return "downloaded"


def fetch_mediapipe(*, log: LogFn | None = None) -> str:
    target = mediapipe_task_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    if mediapipe_present():
        _log(f"[ok] MediaPipe pose already present: {target}", log)
        return "present"

    url = (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
    )
    _log(f"[fetch] Downloading MediaPipe pose landmarker → {target}", log)
    try:
        urllib.request.urlretrieve(url, target)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download MediaPipe pose model from {url}: {exc}. "
            "Check network access or place the file at "
            f"{target} manually."
        ) from exc
    if not mediapipe_present():
        raise RuntimeError(f"MediaPipe download finished but file invalid: {target}")
    _log(f"[ok] MediaPipe pose ready: {target}", log)
    return "downloaded"


def fetch_schp(*, log: LogFn | None = None) -> str:
    if schp_cached():
        _log(f"[ok] SegFormer ({SCHP_MODEL_ID}) already in HF cache", log)
        return "present"

    _log(
        f"[fetch] Prefetching SegFormer clothes parser ({SCHP_MODEL_ID}) into HF cache…",
        log,
    )
    try:
        from transformers import SegformerForSemanticSegmentation
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency transformers. Install: pip install transformers"
        ) from exc

    try:
        SegformerForSemanticSegmentation.from_pretrained(SCHP_MODEL_ID)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download SegFormer model '{SCHP_MODEL_ID}': {exc}. "
            "Ensure network access to huggingface.co, or pre-cache the model."
        ) from exc
    _log(f"[ok] SegFormer cached: {SCHP_MODEL_ID}", log)
    return "downloaded"


def ensure_real_models(
    *,
    skip_schp: bool = False,
    log: LogFn | None = None,
) -> FetchReport:
    """Check and fetch all weights required by the real pipeline.

    Raises RuntimeError if any required fetch fails (does not fall back to mock).
    """
    report = FetchReport()
    steps: list[tuple[str, Callable[..., str], bool]] = [
        ("smplx", fetch_smplx, True),
        ("mediapipe_pose", fetch_mediapipe, True),
        ("segformer_schp", fetch_schp, not skip_schp),
    ]

    _log("[check] Verifying real-pipeline model weights…", log)
    for name, fn, enabled in steps:
        report.checked.append(name)
        if not enabled:
            report.skipped.append(name)
            _log(f"[skip] {name}", log)
            continue
        try:
            status = fn(log=log)
            if status == "present":
                report.present.append(name)
            else:
                report.downloaded.append(name)
        except Exception as exc:
            msg = f"{name}: {exc}"
            report.errors.append(msg)
            _log(f"[fail] {msg}", log)

    if report.errors:
        detail = "; ".join(report.errors)
        raise RuntimeError(
            "H3D real pipeline model fetch failed. "
            f"{detail}. "
            "Fix network/deps and retry, or run `python scripts/fetch_models.py`. "
            "To skip auto-fetch set H3D_AUTO_FETCH_MODELS=0 after placing weights. "
            "Mock is only used when H3D_PIPELINE=mock is set explicitly."
        )

    _log(
        "[check] Models ready "
        f"(present={report.present}, downloaded={report.downloaded}, skipped={report.skipped})",
        log,
    )
    return report
