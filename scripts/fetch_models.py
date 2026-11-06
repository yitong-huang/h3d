#!/usr/bin/env python3
"""Download weights required by the real H3D pipeline.

SMPL-X: research license from https://smpl-x.is.tue.mpg.de/ — this script pulls a
commonly mirrored `SMPLX_NEUTRAL.pkl` for development. You must accept the MPI
SMPL-X license before redistributing or commercial use.

SCHP-equivalent: Hugging Face `mattmdjaga/segformer_b2_clothes` (Apache-2.0 / model card).
MediaPipe Pose: Google Apache-2.0 task bundle.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "data" / "models"


def fetch_smplx(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / "SMPLX_NEUTRAL.pkl"
    if target.exists() and target.stat().st_size > 1_000_000:
        print(f"[ok] SMPL-X already present: {target}")
        return
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit("pip install huggingface_hub")
    print("Downloading SMPLX_NEUTRAL.pkl (≈520MB)…")
    path = hf_hub_download(
        repo_id="stran9682/smplx_model",
        filename="SMPLX_NEUTRAL.pkl",
        local_dir=str(dest / "_hf"),
    )
    shutil.copy2(path, target)
    print(f"[ok] {target} ({target.stat().st_size} bytes)")
    print(
        "NOTE: SMPL-X is © Max Planck Institute. Register at "
        "https://smpl-x.is.tue.mpg.de/ and comply with the license."
    )


def fetch_mediapipe(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / "pose_landmarker_full.task"
    if target.exists():
        print(f"[ok] MediaPipe pose already present: {target}")
        return
    url = (
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
        "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
    )
    print("Downloading MediaPipe pose landmarker…")
    urllib.request.urlretrieve(url, target)
    print(f"[ok] {target}")


def prefetch_schp() -> None:
    print("Prefetching SegFormer clothes parser (SCHP-equivalent)…")
    from transformers import SegformerForSemanticSegmentation

    SegformerForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
    print("[ok] SegFormer cached via Hugging Face hub")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-schp", action="store_true")
    args = ap.parse_args()
    fetch_smplx(MODEL_DIR / "smplx")
    fetch_mediapipe(MODEL_DIR / "mediapipe")
    if not args.skip_schp:
        prefetch_schp()
    print("Done. Default pipeline backend is real (H3D_PIPELINE=real).")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
