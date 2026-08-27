"""Application-level image, mask, and result contracts.

These types are the only data shapes the rest of PixelForge should exchange
with model adapters. Research-repository internals stay behind the adapter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray

ImageArray = NDArray[np.uint8]
"""RGB image, shape ``(H, W, 3)``, dtype ``uint8``."""

MaskArray = NDArray[np.bool_]
"""Binary mask, shape ``(H, W)``, dtype ``bool``. True = selected / inpaint."""


class BackendType(str, Enum):
    """Where a model is classified to run. Placement is an empirical outcome."""

    LOCAL_MPS = "LOCAL_MPS"
    CLOUD_GPU = "CLOUD_GPU"
    CPU = "CPU"
    UNAVAILABLE = "UNAVAILABLE"


class ModelStatus(str, Enum):
    """Lifecycle of one adapter instance. Distinct from :class:`BackendType`."""

    READY = "READY"
    UNAVAILABLE = "UNAVAILABLE"
    LOADING = "LOADING"
    ERROR = "ERROR"


def validate_image(image: np.ndarray) -> ImageArray:
    """Raise ``ValueError`` unless ``image`` is H×W×3 RGB uint8."""
    if not isinstance(image, np.ndarray):
        raise TypeError(f"image must be a numpy.ndarray, got {type(image).__name__}")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must have shape H×W×3, got {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"image dtype must be uint8, got {image.dtype}")
    return image


def validate_mask(mask: np.ndarray, *, image: np.ndarray | None = None) -> MaskArray:
    """Raise ``ValueError`` unless ``mask`` is H×W bool (optionally matching ``image``)."""
    if not isinstance(mask, np.ndarray):
        raise TypeError(f"mask must be a numpy.ndarray, got {type(mask).__name__}")
    if mask.ndim != 2:
        raise ValueError(f"mask must have shape H×W, got {mask.shape}")
    if mask.dtype != np.bool_ and mask.dtype != bool:
        raise ValueError(f"mask dtype must be bool, got {mask.dtype}")
    if image is not None and mask.shape != image.shape[:2]:
        raise ValueError(
            f"mask shape {mask.shape} does not match image {image.shape[:2]}"
        )
    return mask.astype(bool, copy=False)


def mask_to_pil_l(mask: np.ndarray):
    """Convert PixelForge bool H×W to Moebius/PixelHacker PIL ``L`` (255 = inpaint)."""
    from PIL import Image

    validate_mask(mask)
    return Image.fromarray((mask.astype(np.uint8)) * 255, mode="L")


def image_to_pil_rgb(image: np.ndarray):
    """Convert PixelForge RGB uint8 to PIL ``RGB``."""
    from PIL import Image

    validate_image(image)
    return Image.fromarray(image, mode="RGB")


def pil_rgb_to_array(image) -> ImageArray:
    """Convert a PIL image to H×W×3 uint8."""
    from PIL import Image

    if not isinstance(image, Image.Image):
        raise TypeError(f"expected a PIL Image, got {type(image).__name__}")
    rgb = image.convert("RGB")
    arr = np.asarray(rgb)
    return validate_image(arr)


@dataclass
class SegmentationResult:
    """Standard output of a segmentation adapter."""

    mask: MaskArray
    confidence: float | None
    model: str
    method: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.mask = validate_mask(self.mask)


@dataclass
class BoundingBox:
    """Axis-aligned box in source-image pixel coordinates (XYXY)."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    label: str

    def as_xyxy_int(self) -> tuple[int, int, int, int]:
        return int(self.x1), int(self.y1), int(self.x2), int(self.y2)


@dataclass
class GroundingResult:
    """Open-vocabulary detections for a text prompt."""

    detections: list[BoundingBox]
    model: str
    prompt: str
    backend: BackendType
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TextSelectionResult:
    """Mask produced from text grounding + box segmentation."""

    mask: MaskArray
    segmentation: SegmentationResult
    grounding: GroundingResult
    selected_detection: BoundingBox
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.mask = validate_mask(self.mask)


@dataclass
class InpaintParams:
    """Optional inpainting knobs. ``None`` fields use the adapter's upstream defaults."""

    num_steps: int | None = None
    guidance_scale: float | None = None
    strength: float | None = None
    paste: bool | None = None
    noise_offset: float | None = None
    image_size: int | None = None


@dataclass
class InpaintingResult:
    """Standard output of an inpainting adapter."""

    result: ImageArray
    latency_ms: float
    memory_mb: float | None
    model: str
    backend: BackendType
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.result = validate_image(self.result)
