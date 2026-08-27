"""InstructPix2Pix adapter — cloud GPU boundary only.

Phase 13A declares capability, checks whether a cloud endpoint is configured,
and defines the request shape a later phase will send over HTTP. Local CUDA/MPS
execution is not offered from this adapter.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from models.adapters.base import InstructionEditAdapter
from models.config import model_entry
from models.errors import ModelInferenceError, ModelUnavailableError
from models.types import (
    BackendType,
    InstructionEditParams,
    InstructionEditResult,
    validate_image,
)

# Upstream edit_cli.py defaults (Phase 12A source audit).
_DEFAULT_STEPS = 100
_DEFAULT_GUIDANCE_TEXT = 7.5
_DEFAULT_GUIDANCE_IMAGE = 1.5
_DEFAULT_RESOLUTION = 512


class InstructPix2PixAdapter(InstructionEditAdapter):
    """Classified ``CLOUD_GPU``. Remote inference is not implemented in Phase 13A."""

    def __init__(
        self,
        *,
        endpoint: str | None = None,
        enabled: bool | None = None,
    ) -> None:
        super().__init__()
        cfg = model_entry("instruct_pix2pix")
        env_ep = os.environ.get("PIXELFORGE_INSTRUCT_PIX2PIX_ENDPOINT")
        raw = (
            env_ep
            if env_ep is not None and env_ep != ""
            else endpoint
            if endpoint is not None
            else cfg.get("endpoint") or ""
        )
        self._endpoint = str(raw).strip()
        self._enabled = cfg.get("enabled", True) if enabled is None else enabled
        self._display = str(cfg.get("display_name") or "InstructPix2Pix")

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
            raise ModelUnavailableError(
                "InstructPix2Pix cloud backend is not configured."
            )
        self._loaded = True
        self._error = None

    def unload(self) -> None:
        self._loaded = False
        self._loading = False

    def infer(
        self,
        image: np.ndarray,
        instruction: str,
        params: InstructionEditParams | None = None,
        /,
    ) -> InstructionEditResult:
        validate_image(image)
        prompt = instruction.strip()
        if not prompt:
            raise ValueError("Instruction is empty.")
        if not self.is_available():
            raise ModelUnavailableError(
                "InstructPix2Pix cloud backend is not configured."
            )
        raise ModelInferenceError(
            "InstructPix2Pix cloud inference is not implemented."
        )

    def build_remote_request(
        self,
        image: np.ndarray,
        instruction: str,
        params: InstructionEditParams | None = None,
    ) -> dict[str, Any]:
        """Declared wire shape. Serialization is not implemented in Phase 13A."""
        p = params or InstructionEditParams()
        return {
            "endpoint": self._endpoint,
            "instruction": instruction.strip(),
            "image_shape": list(image.shape),
            "num_steps": p.num_steps if p.num_steps is not None else _DEFAULT_STEPS,
            "guidance_text": (
                p.guidance_text if p.guidance_text is not None else _DEFAULT_GUIDANCE_TEXT
            ),
            "guidance_image": (
                p.guidance_image
                if p.guidance_image is not None
                else _DEFAULT_GUIDANCE_IMAGE
            ),
            "resolution": p.resolution if p.resolution is not None else _DEFAULT_RESOLUTION,
        }

    def deserialize_remote_result(self, payload: Any) -> InstructionEditResult:
        """Declared response shape. Not implemented in Phase 13A."""
        raise ModelInferenceError(
            "InstructPix2Pix cloud inference is not implemented."
        )
