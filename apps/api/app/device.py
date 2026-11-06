"""Resolve H3D_DEVICE=auto|mps|cpu|cuda to a torch.device."""

from __future__ import annotations

import logging
import sys
from functools import lru_cache

import torch

from .config import DEVICE

logger = logging.getLogger("uvicorn.error")


def _mps_available() -> bool:
    try:
        return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())
    except Exception:
        return False


@lru_cache(maxsize=1)
def resolve_device() -> torch.device:
    """Pick compute device from H3D_DEVICE (default: auto).

    auto:
      - CUDA if available
      - else MPS on macOS if available (Apple Silicon / supported Intel+AMD setups)
      - else CPU
    """
    pref = (DEVICE or "auto").strip().lower()

    if pref in {"auto", ""}:
        if torch.cuda.is_available():
            chosen = torch.device("cuda")
        elif sys.platform == "darwin" and _mps_available():
            chosen = torch.device("mps")
        else:
            chosen = torch.device("cpu")
        logger.info(
            "Device auto-selected: %s (platform=%s, cuda=%s, mps=%s)",
            chosen,
            sys.platform,
            torch.cuda.is_available(),
            _mps_available(),
        )
        return chosen

    if pref.startswith("cuda"):
        if not torch.cuda.is_available():
            logger.warning("H3D_DEVICE=%s but CUDA unavailable — falling back to CPU", pref)
            return torch.device("cpu")
        return torch.device(pref if ":" in pref else "cuda")

    if pref == "mps":
        if not _mps_available():
            logger.warning("H3D_DEVICE=mps but MPS unavailable — falling back to CPU")
            return torch.device("cpu")
        return torch.device("mps")

    if pref == "cpu":
        return torch.device("cpu")

    logger.warning("Unknown H3D_DEVICE=%s — using CPU", pref)
    return torch.device("cpu")


def device_report() -> dict:
    d = resolve_device()
    return {
        "requested": DEVICE or "auto",
        "resolved": str(d),
        "platform": sys.platform,
        "cudaAvailable": torch.cuda.is_available(),
        "mpsAvailable": _mps_available(),
    }
