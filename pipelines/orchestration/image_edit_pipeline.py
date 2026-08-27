"""Core image-editing pipeline over standardized adapters.

Workflow: image → segmentation → mask refinement → inpainting → result.

This module imports adapters only through the registry (or injected fakes in
tests). It never imports ``research.upstream``.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import Any

import numpy as np

from models.adapters.base import GroundingAdapter, InpaintingAdapter, ModelAdapter, SegmentationAdapter
from models.errors import ModelInferenceError, ModelLoadError, ModelUnavailableError
from models.registry import get_adapter
from models.types import (
    BackendType,
    ImageArray,
    InpaintParams,
    InpaintingResult,
    GroundingResult,
    MaskArray,
    SegmentationResult,
    TextSelectionResult,
    validate_image,
    validate_mask,
)
from pipelines.errors import (
    PipelineBackendError,
    PipelinePromptError,
    PipelineValidationError,
    UnsupportedEditIntentError,
)
from pipelines.mask_refinement import refine_mask, require_non_empty_mask
from pipelines.editing_capabilities import EditIntent, localized_edit_supported
from pipelines.types import ImageEditPipelineResult, MaskRefinementOps, PipelineLatency

logger = logging.getLogger(__name__)

SUPPORTED_INPAINT_BACKENDS = frozenset({"moebius", "pixelhacker"})
SUPPORTED_GROUNDING_BACKENDS = frozenset({"grounding_dino"})


class ImageEditPipeline:
    """SAM 2 segmentation + adapter inpainting behind one application API."""

    def __init__(
        self,
        *,
        segmentation_provider: Callable[[], SegmentationAdapter] | None = None,
        inpaint_provider: Callable[[str], InpaintingAdapter] | None = None,
        grounding_provider: Callable[[str], GroundingAdapter] | None = None,
        unload_between_stages: bool = True,
    ) -> None:
        self._segmentation_provider = segmentation_provider or (
            lambda: get_adapter("sam2")  # type: ignore[return-value]
        )
        self._inpaint_provider = inpaint_provider or (
            lambda name: get_adapter(name)  # type: ignore[return-value]
        )
        self._grounding_provider = grounding_provider or (
            lambda name: get_adapter(name)  # type: ignore[return-value]
        )
        self._unload_between_stages = unload_between_stages

    def segment(self, image: np.ndarray, x: int, y: int) -> SegmentationResult:
        """Point-prompted segmentation via the configured segmentation adapter."""
        image = validate_image(image)
        self._validate_point(image, x, y)
        adapter = self._segmentation_provider()
        self._require_segmentation_adapter(adapter)
        if not adapter.is_available():
            raise ModelUnavailableError(
                f"{adapter.model_name} is not available in this environment."
            )
        logger.info(
            "segment_start model=%s point=(%d,%d) shape=%s",
            adapter.model_name,
            x,
            y,
            image.shape,
        )
        t0 = time.perf_counter()
        try:
            self._prepare_local_adapter(adapter)
            result = adapter.infer(image, x, y)
        except (ModelLoadError, ModelInferenceError, ModelUnavailableError):
            raise
        except Exception as exc:
            raise ModelInferenceError("Segmentation failed.") from exc
        finally:
            if self._unload_between_stages:
                self._release_local_adapter(adapter)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if not bool(result.mask.any()):
            raise ModelInferenceError("Segmentation produced an empty mask.")
        result.metadata.setdefault("pipeline_latency_ms", round(elapsed_ms, 3))
        logger.info(
            "segment_done model=%s latency_ms=%.3f area=%d",
            result.model,
            elapsed_ms,
            int(result.mask.sum()),
        )
        return result

    def select_by_text(
        self,
        image: np.ndarray,
        text_prompt: str,
        *,
        detection_index: int = 0,
        grounding_backend: str = "grounding_dino",
    ) -> TextSelectionResult:
        """Text → grounding boxes → SAM 2 box mask. Does not inpaint."""
        image = validate_image(image)
        prompt = self._validate_text_prompt(text_prompt)
        grounding = self._run_grounding(image, prompt, grounding_backend)
        return self.select_from_grounding(
            image, grounding, detection_index=detection_index
        )

    def select_from_grounding(
        self,
        image: np.ndarray,
        grounding: GroundingResult,
        *,
        detection_index: int = 0,
    ) -> TextSelectionResult:
        """Segment one grounded detection with SAM 2."""
        image = validate_image(image)
        if not grounding.detections:
            raise ModelInferenceError(
                f"No objects found for prompt '{grounding.prompt}'."
            )
        if detection_index < 0 or detection_index >= len(grounding.detections):
            raise PipelineValidationError(
                f"detection_index {detection_index} is out of range "
                f"(found {len(grounding.detections)} detection(s))."
            )
        detection = grounding.detections[detection_index]
        segmentation = self._segment_box(image, detection)
        if not bool(segmentation.mask.any()):
            raise ModelInferenceError("Box segmentation produced an empty mask.")
        return TextSelectionResult(
            mask=segmentation.mask,
            segmentation=segmentation,
            grounding=grounding,
            selected_detection=detection,
            metadata={
                "prompt": grounding.prompt,
                "detection_index": detection_index,
                "detection_count": len(grounding.detections),
                "grounding_model": grounding.model,
                "segmentation_model": segmentation.model,
            },
        )

    def refine_mask(
        self,
        mask: np.ndarray,
        image: np.ndarray,
        operations: MaskRefinementOps | None = None,
        **kwargs: Any,
    ) -> MaskArray:
        """Apply lightweight boolean mask edits. Dimensions must match ``image``."""
        image = validate_image(image)
        return refine_mask(mask, image, operations, **kwargs)

    def inpaint(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        backend: str = "moebius",
        params: InpaintParams | None = None,
    ) -> InpaintingResult:
        """Inpaint ``mask`` (True = generate) using the requested backend."""
        image = validate_image(image)
        mask = require_non_empty_mask(validate_mask(mask, image=image), stage="inpaint")
        adapter = self._resolve_inpaint_adapter(backend)
        if not adapter.is_available():
            raise ModelUnavailableError(
                f"{adapter.model_name} ({backend}) is not available."
            )
        logger.info(
            "inpaint_start model=%s backend=%s mask_area=%d shape=%s",
            adapter.model_name,
            adapter.backend_type.value,
            int(mask.sum()),
            image.shape,
        )
        t0 = time.perf_counter()
        try:
            self._prepare_local_adapter(adapter)
            result = adapter.infer(image, mask, params)
        except (ModelLoadError, ModelInferenceError, ModelUnavailableError):
            raise
        except Exception as exc:
            raise ModelInferenceError("Inpainting failed.") from exc
        finally:
            if self._unload_between_stages:
                self._release_local_adapter(adapter)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if result.latency_ms is None or result.latency_ms == 0:
            result.metadata.setdefault("pipeline_latency_ms", round(elapsed_ms, 3))
        logger.info(
            "inpaint_done model=%s latency_ms=%.3f adapter_ms=%s",
            result.model,
            elapsed_ms,
            result.latency_ms,
        )
        return result

    def edit_localized(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        *,
        backend: str = "moebius",
        params: InpaintParams | None = None,
        instruction: str | None = None,
    ) -> InpaintingResult:
        """Localized mask-conditioned edit (fill/remove). Does not accept text instructions."""
        if instruction is not None and instruction.strip():
            raise UnsupportedEditIntentError(
                f"Backend '{backend}' does not accept text instructions. "
                "Use localized inpainting on the selected mask only."
            )
        if not localized_edit_supported(backend):
            raise PipelineBackendError(
                f"Backend '{backend}' does not support localized mask editing."
            )
        image = validate_image(image)
        mask = validate_mask(mask, image=image)
        logger.info(
            "edit_localized_start backend=%s intent=%s mask_area=%d",
            backend,
            EditIntent.LOCALIZED_INPAINT.value,
            int(mask.sum()),
        )
        return self.inpaint(image, mask, backend=backend, params=params)

    def remove_object(
        self,
        image: np.ndarray,
        x: int,
        y: int,
        *,
        backend: str = "moebius",
        refinement: MaskRefinementOps | None = None,
        params: InpaintParams | None = None,
    ) -> ImageEditPipelineResult:
        """Full workflow: segment → refine → inpaint."""
        image = validate_image(image)
        total_t0 = time.perf_counter()

        segmentation = self.segment(image, x, y)
        mask = segmentation.mask
        if refinement is not None:
            mask = self.refine_mask(mask, image, refinement)
        mask = require_non_empty_mask(mask, stage="refinement")

        inpainting = self.inpaint(image, mask, backend=backend, params=params)
        total_ms = (time.perf_counter() - total_t0) * 1000.0

        seg_ms = segmentation.metadata.get("pipeline_latency_ms") or segmentation.metadata.get(
            "latency_ms"
        )
        inpaint_ms = inpainting.latency_ms or inpainting.metadata.get("pipeline_latency_ms")

        return ImageEditPipelineResult(
            result=inpainting.result,
            mask=mask,
            segmentation=segmentation,
            inpainting=inpainting,
            selected_model=inpainting.model,
            backend=inpainting.backend,
            latency=PipelineLatency(
                segmentation_ms=float(seg_ms) if seg_ms is not None else None,
                inpainting_ms=float(inpaint_ms) if inpaint_ms is not None else None,
                total_ms=round(total_ms, 3),
            ),
            metadata={
                "prompt_xy": [int(x), int(y)],
                "segmentation_model": segmentation.model,
                "inpaint_backend": backend,
            },
        )

    def _run_grounding(
        self, image: np.ndarray, prompt: str, grounding_backend: str
    ) -> GroundingResult:
        key = grounding_backend.strip().lower()
        if key not in SUPPORTED_GROUNDING_BACKENDS:
            raise PipelineBackendError(
                f"Unsupported grounding backend '{grounding_backend}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_GROUNDING_BACKENDS))}."
            )
        adapter = self._grounding_provider(key)
        self._require_grounding_adapter(adapter)
        if not adapter.is_available():
            raise ModelUnavailableError(
                f"{adapter.model_name} ({grounding_backend}) is not available."
            )
        logger.info("ground_start backend=%s prompt=%r", key, prompt)
        t0 = time.perf_counter()
        try:
            if adapter.backend_type == BackendType.LOCAL_MPS:
                self._prepare_local_adapter(adapter)
            elif not adapter.is_loaded:
                adapter.load()
            result = adapter.infer(image, prompt)
        except (ModelLoadError, ModelInferenceError, ModelUnavailableError):
            raise
        except Exception as exc:
            raise ModelInferenceError("Grounding failed.") from exc
        finally:
            if adapter.backend_type == BackendType.LOCAL_MPS and self._unload_between_stages:
                self._release_local_adapter(adapter)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        result.metadata.setdefault("pipeline_latency_ms", round(elapsed_ms, 3))
        logger.info(
            "ground_done detections=%d latency_ms=%.3f",
            len(result.detections),
            elapsed_ms,
        )
        return result

    def _segment_box(self, image: np.ndarray, detection) -> SegmentationResult:
        adapter = self._segmentation_provider()
        self._require_segmentation_adapter(adapter)
        if not adapter.is_available():
            raise ModelUnavailableError(
                f"{adapter.model_name} is not available in this environment."
            )
        x1, y1, x2, y2 = detection.as_xyxy_int()
        logger.info("segment_box box=(%d,%d,%d,%d)", x1, y1, x2, y2)
        t0 = time.perf_counter()
        try:
            self._prepare_local_adapter(adapter)
            result = adapter.segment_box(image, x1, y1, x2, y2)
        except (ModelLoadError, ModelInferenceError, ModelUnavailableError):
            raise
        except Exception as exc:
            raise ModelInferenceError("Box segmentation failed.") from exc
        finally:
            if self._unload_between_stages:
                self._release_local_adapter(adapter)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        result.metadata.setdefault("pipeline_latency_ms", round(elapsed_ms, 3))
        return result

    def _resolve_inpaint_adapter(self, backend: str) -> InpaintingAdapter:
        key = backend.strip().lower()
        if key not in SUPPORTED_INPAINT_BACKENDS:
            raise PipelineBackendError(
                f"Unsupported inpainting backend '{backend}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_INPAINT_BACKENDS))}."
            )
        adapter = self._inpaint_provider(key)
        self._require_inpainting_adapter(adapter)
        return adapter

    @staticmethod
    def _validate_text_prompt(text_prompt: str) -> str:
        prompt = text_prompt.strip()
        if not prompt:
            raise PipelinePromptError("Text prompt is empty.")
        return prompt

    @staticmethod
    def _require_grounding_adapter(adapter: ModelAdapter) -> GroundingAdapter:
        if not isinstance(adapter, GroundingAdapter):
            raise PipelineValidationError("Configured grounding adapter is invalid.")
        return adapter

    @staticmethod
    def _validate_point(image: np.ndarray, x: int, y: int) -> None:
        h, w = image.shape[:2]
        if not isinstance(x, int) or not isinstance(y, int):
            raise PipelineValidationError("Point coordinates must be integers.")
        if not (0 <= x < w and 0 <= y < h):
            raise PipelineValidationError(
                f"Point ({x}, {y}) is outside the image bounds ({w}×{h})."
            )

    @staticmethod
    def _require_segmentation_adapter(adapter: ModelAdapter) -> SegmentationAdapter:
        if not isinstance(adapter, SegmentationAdapter):
            raise PipelineValidationError("Configured segmentation adapter is invalid.")
        return adapter

    @staticmethod
    def _require_inpainting_adapter(adapter: ModelAdapter) -> InpaintingAdapter:
        if not isinstance(adapter, InpaintingAdapter):
            raise PipelineValidationError("Configured inpainting adapter is invalid.")
        return adapter

    @staticmethod
    def _prepare_local_adapter(adapter: ModelAdapter) -> None:
        if adapter.backend_type != BackendType.LOCAL_MPS:
            if not adapter.is_loaded:
                adapter.load()
            return
        other = ModelAdapter._local_resident
        if other is not None and other is not adapter and other.is_loaded:
            logger.debug("unload_other model=%s", other.model_name)
            other.unload()
        if not adapter.is_loaded:
            adapter.load()

    @staticmethod
    def _release_local_adapter(adapter: ModelAdapter) -> None:
        if adapter.is_loaded:
            adapter.unload()
