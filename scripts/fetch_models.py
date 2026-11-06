#!/usr/bin/env python3
"""CLI wrapper around `app.models_fetch` for offline / CI use.

API startup also calls the same module when H3D_PIPELINE=real and
H3D_AUTO_FETCH_MODELS=1 (default).
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running from repo root without installing the package.
API_ROOT = Path(__file__).resolve().parents[1] / "apps" / "api"
sys.path.insert(0, str(API_ROOT))

from app.models_fetch import ensure_real_models  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Download H3D real-pipeline weights")
    ap.add_argument("--skip-schp", action="store_true", help="Skip SegFormer prefetch")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report = ensure_real_models(skip_schp=args.skip_schp, log=print)
    print(
        "Done.",
        f"present={report.present}",
        f"downloaded={report.downloaded}",
        f"skipped={report.skipped}",
    )
    print("Default pipeline backend is real (H3D_PIPELINE=real).")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
