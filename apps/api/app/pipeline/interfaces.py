"""Replaceable GPU stage interfaces. Real DensePose / SHAPY / etc. plug in here."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ImageMeta:
    path: Path
    filename: str
    part_hint: str | None = None  # optional user-provided body part


@dataclass
class RoutedImage:
    path: Path
    filename: str
    route: str  # "global" | "local"
    coverage: dict[str, float] = field(default_factory=dict)
    part_label: str | None = None
    skin_mask_summary: dict[str, float] = field(default_factory=dict)


@dataclass
class BodyEstimate:
    beta: list[float]  # SMPL-X shape params ℝ¹⁰
    theta_hint: list[float]  # coarse pose hint
    skin_tone_rgb: tuple[float, float, float]
    notes: str = ""
    # Real pipeline: full SMPL-X parameter pack for assembly / fusion
    full_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class DetailMergeResult:
    uv_patches: list[dict[str, Any]]
    normal_map_hint: str
    fused_regions: list[str]


@dataclass
class AssembledAsset:
    glb_path: Path
    meta: dict[str, Any]


class ImageParseRouter(ABC):
    @abstractmethod
    def parse_and_route(self, images: list[ImageMeta]) -> list[RoutedImage]:
        ...


class UnderClothingEstimator(ABC):
    @abstractmethod
    def estimate(self, globals_: list[RoutedImage]) -> BodyEstimate:
        ...


class LocalDetailMerger(ABC):
    @abstractmethod
    def merge(
        self, locals_: list[RoutedImage], body: BodyEstimate
    ) -> DetailMergeResult:
        ...


class AssetAssembler(ABC):
    @abstractmethod
    def assemble(
        self,
        job_dir: Path,
        body: BodyEstimate,
        details: DetailMergeResult,
        routed: list[RoutedImage],
    ) -> AssembledAsset:
        ...
