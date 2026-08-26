"""PixelHacker adapter — cloud GPU boundary only.

Phase 6 does **not** implement remote inference. This class declares capability,
checks whether an endpoint is configured, and defines the request/response
shape a later phase will send over HTTP.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from models.adapters.base import InpaintingAdapter
from models.config import model_entry
from models.errors import ModelInferenceError, ModelUnavailableError
from models.types import (
    BackendType,
    InpaintParams,
    InpaintingResult,
    validate_image,
    validate_mask,
)

# Upstream infer_pixelhacker.py defaults (Phase 5 source audit).
_DEFAULT_STEPS = 20
_DEFAULT_GUIDANCE = 4.5
_DEFAULT_STRENGTH = 0.999
_DEFAULT_PASTE = False
_DEFAULT_NOISE_OFFSET = 0.0357
_DEFAULT_SIZE = 512


class PixelHackerAdapter(InpaintingAdapter):
    """Classified ``CLOUD_GPU``. Inference is not implemented in Phase 6."""

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        enabled: bool | None = None,
    ) -> None:
        super().__init__()
        cfg = model_entry("pixelhacker")
        env_ep = os.environ.get("PIXELFORGE_PIXELHACKER_ENDPOINT")
        raw = (
            env_ep
            if env_ep is not None and env_ep != ""
            else endpoint
            if endpoint is not None
            else cfg.get("endpoint") or ""
        )
        self._endpoint = str(raw).strip()
        self._enabled = cfg.get("enabled", True) if enabled is None else enabled
        self._display = str(cfg.get("display_name") or "PixelHacker")

    @property
    def model_name(self) -> str:
        return self._display

    @property
    def backend_type(self) -> BackendType:
        return BackendType.CLOUD_GPU

    def is_available(self) -> bool:
        """True only when a cloud endpoint is configured. Weights are not local."""
        if not self._enabled:
            return False
        return bool(self._endpoint)

    def load(self) -> None:
        if self._loaded:
            return
        if not self.is_available():
            self._error = "endpoint"
            raise ModelUnavailableError("PixelHacker cloud backend is not configured.")
        # No local weights. Marking loaded means "endpoint recorded", not "model resident".
        self._loaded = True
        self._error = None

    def unload(self) -> None:
        self._loaded = False
        self._loading = False

    def infer(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        params: InpaintParams | None = None,
        /,
    ) -> InpaintingResult:
        validate_image(image)
        validate_mask(mask, image=image)
        if not self.is_available():
            raise ModelUnavailableError("PixelHacker cloud backend is not configured.")
        # NOT IMPLEMENTED IN PHASE 6
        raise ModelInferenceError(
            "PixelHacker cloud inference is not implemented."
        )

    def build_remote_request(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        params: InpaintParams | None = None,
    ) -> dict[str, Any]:
        """Declared wire shape. Serialization is not implemented in Phase 6."""
        p = params or InpaintParams()
        return {
            "endpoint": self._endpoint,
            "image_shape": list(image.shape),
            "mask_shape": list(mask.shape),
            "num_steps": p.num_steps if p.num_steps is not None else _DEFAULT_STEPS,
            "guidance_scale": (
                p.guidance_scale if p.guidance_scale is not None else _DEFAULT_GUIDANCE
            ),
            "strength": p.strength if p.strength is not None else _DEFAULT_STRENGTH,
            "paste": p.paste if p.paste is not None else _DEFAULT_PASTE,
            "noise_offset": (
                p.noise_offset if p.noise_offset is not None else _DEFAULT_NOISE_OFFSET
            ),
            "image_size": p.image_size if p.image_size is not None else _DEFAULT_SIZE,
        }

    def deserialize_remote_result(self, payload: Any) -> InpaintingResult:
        """Declared response shape. Not implemented in Phase 6."""
        raise ModelInferenceError(
            "PixelHacker cloud inference is not implemented."
        )
