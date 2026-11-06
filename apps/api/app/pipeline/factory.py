"""Select real vs mock stage implementations."""

from __future__ import annotations

from ..config import PIPELINE_BACKEND
from .interfaces import (
    AssetAssembler,
    ImageParseRouter,
    LocalDetailMerger,
    UnderClothingEstimator,
)


def create_stages() -> tuple[
    ImageParseRouter, UnderClothingEstimator, LocalDetailMerger, AssetAssembler
]:
    backend = PIPELINE_BACKEND
    if backend in {"mock", "stub"}:
        from .mocks import (
            MockAssetAssembler,
            MockImageParseRouter,
            MockLocalDetailMerger,
            MockUnderClothingEstimator,
        )

        return (
            MockImageParseRouter(),
            MockUnderClothingEstimator(),
            MockLocalDetailMerger(),
            MockAssetAssembler(),
        )

    # Default: real
    from .real_assemble import RealAssetAssembler
    from .real_estimate import RealUnderClothingEstimator
    from .real_merge import RealLocalDetailMerger
    from .real_parse import RealImageParseRouter

    return (
        RealImageParseRouter(),
        RealUnderClothingEstimator(),
        RealLocalDetailMerger(),
        RealAssetAssembler(),
    )


def backend_name() -> str:
    return "mock" if PIPELINE_BACKEND in {"mock", "stub"} else "real"
