"""SAM 2.1 Hiera-Tiny adapter — local MPS point segmentation.

Does not copy SAM 2 source. The pinned checkout is imported from
``research/upstream/sam2`` inside ``load()`` only. Upstream stays READ-ONLY.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from models.adapters._runtime import (
    mps_is_available,
    mps_memory_mb,
    mps_synchronize,
    prepare_upstream_import,
)
from models.adapters.base import SegmentationAdapter
from models.config import model_entry, project_root, resolve_path
from models.errors import ModelInferenceError, ModelLoadError, ModelUnavailableError
from models.types import (
    BackendType,
    SegmentationResult,
    validate_image,
)


class SAM2Adapter(SegmentationAdapter):
    """Validated Phase 3 backend. Minimum operation: ``segment_point``."""

    def __init__(
        self,
        *,
        checkpoint_path: Path | None = None,
        upstream_dir: Path | None = None,
        config_file: str | None = None,
        enabled: bool | None = None,
    ) -> None:
        super().__init__()
        cfg = model_entry("sam2")
        env_ckpt = os.environ.get("PIXELFORGE_SAM2_CHECKPOINT")
        self._checkpoint = (
            Path(env_ckpt).expanduser()
            if env_ckpt
            else checkpoint_path
            or resolve_path(cfg.get("checkpoint"))
        )
        self._upstream = upstream_dir or resolve_path(cfg.get("upstream_dir"))
        self._config_file = config_file or str(
            cfg.get("config_file") or "configs/sam2.1/sam2.1_hiera_t.yaml"
        )
        self._enabled = cfg.get("enabled", True) if enabled is None else enabled
        self._display = str(cfg.get("display_name") or "SAM 2.1 Hiera-Tiny")
        self._predictor: Any = None

    @property
    def model_name(self) -> str:
        return self._display

    @property
    def backend_type(self) -> BackendType:
        return BackendType.LOCAL_MPS

    def is_available(self) -> bool:
        if not self._enabled:
            return False
        if self._checkpoint is None or not self._checkpoint.is_file():
            return False
        if self._upstream is None or not (self._upstream / "sam2").is_dir():
            return False
        return self._mps_is_available()

    def _mps_is_available(self) -> bool:
        return mps_is_available()

    def load(self) -> None:
        if self._loaded:
            return
        if not self._enabled:
            self._error = "disabled"
            raise ModelUnavailableError("SAM 2 is disabled.")
        if self._checkpoint is None or not self._checkpoint.is_file():
            self._error = "checkpoint"
            raise ModelUnavailableError("SAM 2 checkpoint is not available.")
        if not self._mps_is_available():
            self._error = "mps"
            raise ModelUnavailableError(
                "SAM 2 requires Apple MPS, which is not available in this process."
            )
        if self._upstream is None or not self._upstream.is_dir():
            self._error = "upstream"
            raise ModelUnavailableError("SAM 2 upstream repository is not available.")

        self._begin_local_load()
        self._loading = True
        self._error = None
        try:
            prepare_upstream_import(self._upstream)
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            model = build_sam2(
                config_file=self._config_file,
                ckpt_path=str(self._checkpoint),
                device="mps",
                mode="eval",
                apply_postprocessing=True,
            )
            device = str(next(model.parameters()).device)
            if not device.startswith("mps"):
                raise ModelLoadError(
                    f"SAM 2 loaded on {device} instead of MPS."
                )
            self._predictor = SAM2ImagePredictor(model)
            self._loaded = True
        except (ModelLoadError, ModelUnavailableError):
            self._end_local_unload()
            raise
        except Exception as exc:
            self._end_local_unload()
            self._error = type(exc).__name__
            raise ModelLoadError("SAM 2 failed to load.") from exc
        finally:
            self._loading = False

    def unload(self) -> None:
        self._predictor = None
        self._loaded = False
        self._loading = False
        self._end_local_unload()
        try:
            import gc

            import torch

            gc.collect()
            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
        except Exception:
            pass

    def segment_point(self, image: np.ndarray, x: int, y: int) -> SegmentationResult:
        """Click-prompted mask. ``(x, y)`` are pixel coordinates, origin top-left."""
        return self.infer(image, x, y)

    def infer(self, image: np.ndarray, x: int, y: int, /) -> SegmentationResult:
        image = validate_image(image)
        if not self._loaded:
            self.load()
        if self._predictor is None:
            raise ModelLoadError("SAM 2 is not loaded.")

        h, w = image.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            raise ModelInferenceError(
                f"Point ({x}, {y}) is outside the image ({w}×{h})."
            )

        coords = np.array([[x, y]], dtype=np.float32)
        labels = np.array([1], dtype=np.int32)
        t0 = time.perf_counter()
        try:
            self._predictor.set_image(image)
            masks, scores, _low_res = self._predictor.predict(
                point_coords=coords,
                point_labels=labels,
                multimask_output=False,
            )
            mps_synchronize()
        except Exception as exc:
            self._error = type(exc).__name__
            raise ModelInferenceError("SAM 2 inference failed.") from exc

        latency_ms = (time.perf_counter() - t0) * 1000.0
        mask = np.asarray(masks[0]).astype(bool)
        if mask.shape != (h, w):
            raise ModelInferenceError(
                f"SAM 2 mask shape {mask.shape} does not match image {(h, w)}."
            )
        confidence = float(np.atleast_1d(scores)[0])
        return SegmentationResult(
            mask=mask,
            confidence=confidence,
            model=self.model_name,
            method="point",
            metadata={
                "prompt_xy": [int(x), int(y)],
                "prompt_label": 1,
                "device": "mps",
                "checkpoint": str(self._checkpoint),
                "latency_ms": round(latency_ms, 3),
                "memory_mb": mps_memory_mb(),
                "project_root": str(project_root()),
            },
        )
