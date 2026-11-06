"""Load and cache the official SMPL-X body model."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..config import SMPLX_GENDER, SMPLX_MODEL_DIR
from ..device import resolve_device


def _model_root() -> str:
    """Return directory that contains a `smplx/` subfolder with SMPLX_*.pkl/npz."""
    model_dir = SMPLX_MODEL_DIR
    smplx_dir = model_dir / "smplx"
    if (smplx_dir / f"SMPLX_{SMPLX_GENDER.upper()}.pkl").exists() or (
        smplx_dir / "SMPLX_NEUTRAL.pkl"
    ).exists() or (smplx_dir / f"SMPLX_{SMPLX_GENDER.upper()}.npz").exists():
        return str(model_dir)
    if model_dir.name == "smplx" and (
        (model_dir / "SMPLX_NEUTRAL.pkl").exists()
        or (model_dir / f"SMPLX_{SMPLX_GENDER.upper()}.pkl").exists()
    ):
        return str(model_dir.parent)
    raise FileNotFoundError(
        f"SMPL-X weights not found under {model_dir}. Run: python scripts/fetch_models.py"
    )


@lru_cache(maxsize=1)
def load_smplx() -> Any:
    import smplx

    root = Path(_model_root())
    device = resolve_device()
    gender_file = f"SMPLX_{SMPLX_GENDER.upper()}"
    ext = "pkl"
    if not (root / "smplx" / f"{gender_file}.pkl").exists() and not (
        root / "smplx" / "SMPLX_NEUTRAL.pkl"
    ).exists():
        if (root / "smplx" / f"{gender_file}.npz").exists():
            ext = "npz"

    model = smplx.create(
        str(root),
        model_type="smplx",
        gender=SMPLX_GENDER,
        use_face_contour=False,
        num_betas=10,
        num_expression_coeffs=10,
        ext=ext,
        use_pca=False,
        flat_hand_mean=True,
    ).to(device)
    model.eval()
    return model


def smplx_faces() -> np.ndarray:
    return np.asarray(load_smplx().faces, dtype=np.int64)


def smplx_weights() -> np.ndarray:
    m = load_smplx()
    return m.lbs_weights.detach().cpu().numpy().astype(np.float32)


def smplx_parents() -> list[int]:
    return [int(p) for p in load_smplx().parents.tolist()]


SMPLX_JOINT_NAMES: list[str] = [
    "Pelvis",
    "LeftHip",
    "RightHip",
    "Spine1",
    "LeftKnee",
    "RightKnee",
    "Spine2",
    "LeftAnkle",
    "RightAnkle",
    "Spine3",
    "LeftFoot",
    "RightFoot",
    "Neck",
    "LeftCollar",
    "RightCollar",
    "Head",
    "LeftShoulder",
    "RightShoulder",
    "LeftElbow",
    "RightElbow",
    "LeftWrist",
    "RightWrist",
    "Jaw",
    "LeftEye",
    "RightEye",
    "LeftIndex1",
    "LeftIndex2",
    "LeftIndex3",
    "LeftMiddle1",
    "LeftMiddle2",
    "LeftMiddle3",
    "LeftPinky1",
    "LeftPinky2",
    "LeftPinky3",
    "LeftRing1",
    "LeftRing2",
    "LeftRing3",
    "LeftThumb1",
    "LeftThumb2",
    "LeftThumb3",
    "RightIndex1",
    "RightIndex2",
    "RightIndex3",
    "RightMiddle1",
    "RightMiddle2",
    "RightMiddle3",
    "RightPinky1",
    "RightPinky2",
    "RightPinky3",
    "RightRing1",
    "RightRing2",
    "RightRing3",
    "RightThumb1",
    "RightThumb2",
    "RightThumb3",
]

HELPER_BONES = [
    ("Spine3_Breast_L", "Spine3"),
    ("Spine3_Breast_R", "Spine3"),
    ("Pelvis_Glute_L", "Pelvis"),
    ("Pelvis_Glute_R", "Pelvis"),
]
