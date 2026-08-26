"""PixelForge model boundary.

Pipeline code talks to adapters through this package. It must not import
``research.upstream`` or FastAPI/frontend modules.
"""

from models.errors import (
    ModelInferenceError,
    ModelLoadError,
    ModelUnavailableError,
)
from models.types import (
    BackendType,
    ImageArray,
    InpaintParams,
    InpaintingResult,
    MaskArray,
    ModelStatus,
    SegmentationResult,
)

__all__ = [
    "BackendType",
    "ImageArray",
    "InpaintParams",
    "InpaintingResult",
    "MaskArray",
    "ModelInferenceError",
    "ModelLoadError",
    "ModelStatus",
    "ModelUnavailableError",
    "SegmentationResult",
]
