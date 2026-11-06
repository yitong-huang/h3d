from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
JOB_DIR = DATA_DIR / "jobs"
MODEL_DIR = DATA_DIR / "models"
ASSET_DIR = Path(__file__).resolve().parent / "assets"

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
JOB_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_IMAGES_PER_JOB = 10

# When Redis/Celery are unavailable, run the pipeline inline (dev default).
USE_INLINE_PIPELINE = os.environ.get("H3D_INLINE_PIPELINE", "1") not in {
    "0",
    "false",
    "False",
}

# Default: real backends. Set H3D_PIPELINE=mock to force stubs.
PIPELINE_BACKEND = os.environ.get("H3D_PIPELINE", "real").strip().lower()

SMPLX_MODEL_DIR = Path(os.environ.get("H3D_SMPLX_DIR", str(MODEL_DIR)))
SMPLX_GENDER = os.environ.get("H3D_SMPLX_GENDER", "neutral")
DEVICE = os.environ.get("H3D_DEVICE", "cpu")  # cpu | cuda

MEDIAPIPE_POSE_TASK = Path(
    os.environ.get(
        "H3D_POSE_TASK",
        str(MODEL_DIR / "mediapipe" / "pose_landmarker_full.task"),
    )
)
SCHP_MODEL_ID = os.environ.get(
    "H3D_SCHP_MODEL", "mattmdjaga/segformer_b2_clothes"
)

# Optional Detectron2 DensePose; when unset, use SegFormer parts + SMPL-X IUV.
DENSEPOSE_BACKEND = os.environ.get("H3D_DENSEPOSE", "smplx_iuv").strip().lower()

# On API startup (real backend): auto-download missing weights. Set 0 to disable.
AUTO_FETCH_MODELS = os.environ.get("H3D_AUTO_FETCH_MODELS", "1") not in {
    "0",
    "false",
    "False",
}
