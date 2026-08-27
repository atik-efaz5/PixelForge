"""Abstract adapter contract.

Adapters are infrastructure boundaries only: no FastAPI, no UI, no database.
Research repositories are imported inside ``load()`` / ``infer()``, never at
adapter import time.

Only one local-MPS adapter may be loaded in-process at a time. Call
``unload()`` before loading another. There is no LRU and no preload.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from models.errors import ModelLoadError
from models.types import BackendType, ModelStatus, SegmentationResult


class ModelAdapter(ABC):
    """Lifecycle + inference boundary for one research model.

    Typed ``infer`` signatures live on the role subclasses so callers are not
    forced through a generic kwargs bag.
    """

    _local_resident: ClassVar[ModelAdapter | None] = None

    def __init__(self) -> None:
        self._loaded = False
        self._loading = False
        self._error: str | None = None

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identity, e.g. ``SAM 2.1 Hiera-Tiny``."""

    @property
    @abstractmethod
    def backend_type(self) -> BackendType:
        """Classified execution placement for this adapter."""

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def status(self) -> ModelStatus:
        if self._error is not None:
            return ModelStatus.ERROR
        if self._loading:
            return ModelStatus.LOADING
        if self._loaded:
            return ModelStatus.READY
        return ModelStatus.UNAVAILABLE

    @abstractmethod
    def is_available(self) -> bool:
        """True when this process could load the model (weights, device, config).

        Must not load weights. File and device probes only.
        """

    @abstractmethod
    def load(self) -> None:
        """Load weights onto the backend. Idempotent if already loaded."""

    @abstractmethod
    def unload(self) -> None:
        """Release weights and GPU/MPS residency. Safe if nothing is loaded."""

    @abstractmethod
    def infer(self, *args: Any, **kwargs: Any) -> Any:
        """Run the model. Subclasses document the exact typed arguments."""

    def _begin_local_load(self) -> None:
        if self.backend_type != BackendType.LOCAL_MPS:
            return
        other = ModelAdapter._local_resident
        if other is not None and other is not self and other.is_loaded:
            raise ModelLoadError(
                f"{other.model_name} is still loaded. Unload it before loading "
                f"{self.model_name}."
            )
        ModelAdapter._local_resident = self

    def _end_local_unload(self) -> None:
        if ModelAdapter._local_resident is self:
            ModelAdapter._local_resident = None


class SegmentationAdapter(ModelAdapter):
    """Adapter that produces a boolean H×W mask from an RGB image."""

    @abstractmethod
    def infer(self, image, x: int, y: int, /):
        """Point-prompted segmentation. See :meth:`SAM2Adapter.segment_point`."""

    def segment_box(
        self, image, x1: int, y1: int, x2: int, y2: int, /
    ) -> SegmentationResult:
        """Box-prompted segmentation when supported by the adapter."""
        raise NotImplementedError(
            f"{self.model_name} does not support box segmentation."
        )


class GroundingAdapter(ModelAdapter):
    """Adapter that maps a text prompt to bounding boxes on an image."""

    @abstractmethod
    def infer(self, image, text_prompt: str, /):
        """Ground ``text_prompt`` on ``image``. See :meth:`GroundingDINOAdapter.ground`."""


class InpaintingAdapter(ModelAdapter):
    """Adapter that fills a boolean H×W mask in an RGB image."""

    @abstractmethod
    def infer(self, image, mask, params=None, /):
        """Inpaint ``mask`` (True = generate) on ``image``."""
