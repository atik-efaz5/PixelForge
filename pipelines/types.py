"""Pipeline-level result types. Image/mask contracts live in ``models.types``."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from models.types import (
    BackendType,
    ImageArray,
    InpaintingResult,
    MaskArray,
    SegmentationResult,
    validate_image,
    validate_mask,
)


@dataclass
class MaskRefinementOps:
    """Lightweight mask edits before inpainting."""

    add: MaskArray | None = None
    remove: MaskArray | None = None
    dilate: int = 0
    erode: int = 0

    def __post_init__(self) -> None:
        if self.add is not None:
            self.add = validate_mask(self.add)
        if self.remove is not None:
            self.remove = validate_mask(self.remove)
        if self.dilate < 0 or self.erode < 0:
            raise ValueError("dilate and erode must be non-negative")


@dataclass
class PipelineLatency:
    segmentation_ms: float | None = None
    inpainting_ms: float | None = None
    total_ms: float = 0.0


@dataclass
class ImageEditPipelineResult:
    """Output of a full or partial edit workflow."""

    result: ImageArray
    mask: MaskArray
    segmentation: SegmentationResult | None
    inpainting: InpaintingResult | None
    selected_model: str
    backend: BackendType
    latency: PipelineLatency
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.result = validate_image(self.result)
        self.mask = validate_mask(self.mask)
