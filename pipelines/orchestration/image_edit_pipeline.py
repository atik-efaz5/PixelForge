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

from models.adapters.base import (
    GroundingAdapter,
    InpaintingAdapter,
    InstructionEditAdapter,
    ModelAdapter,
    SegmentationAdapter,
)
from models.errors import ModelInferenceError, ModelLoadError, ModelUnavailableError
from models.router import (
    ExecutionPreference,
    RoutingCapability,
    RoutingError,
    RoutingOperation,
    RoutingRequest,
    is_automatic_backend,
    route,
)
from models.registry import get_adapter
from models.types import (
    BackendType,
    ImageArray,
    InpaintCandidate,
    InpaintCandidatesResult,
    InpaintParams,
    InpaintingResult,
    InstructionEditParams,
    InstructionEditResult,
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
from pipelines.candidate_seeds import derive_candidate_seeds, resolve_base_seed, validate_candidate_count
from pipelines.editing_capabilities import (
    EditIntent,
    global_instruction_edit_supported,
    localized_edit_supported,
)
from pipelines.types import ImageEditPipelineResult, MaskRefinementOps, PipelineLatency

logger = logging.getLogger(__name__)

SUPPORTED_INPAINT_BACKENDS = frozenset({"moebius", "pixelhacker"})
SUPPORTED_INSTRUCTION_EDIT_BACKENDS = frozenset({"instruct_pix2pix"})
SUPPORTED_GROUNDING_BACKENDS = frozenset({"grounding_dino"})


class ImageEditPipeline:
    """SAM 2 segmentation + adapter inpainting behind one application API."""

    def __init__(
        self,
        *,
        segmentation_provider: Callable[[], SegmentationAdapter] | None = None,
        inpaint_provider: Callable[[str], InpaintingAdapter] | None = None,
        instruction_edit_provider: Callable[[str], InstructionEditAdapter] | None = None,
        grounding_provider: Callable[[str], GroundingAdapter] | None = None,
        unload_between_stages: bool = True,
    ) -> None:
        self._segmentation_provider = segmentation_provider or (
            lambda: get_adapter("sam2")  # type: ignore[return-value]
        )
        self._inpaint_provider = inpaint_provider or (
            lambda name: get_adapter(name)  # type: ignore[return-value]
        )
        self._instruction_edit_provider = instruction_edit_provider or (
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
        routing = self._route_backend(
            operation=RoutingOperation.SEGMENT_POINT,
            capability=RoutingCapability.OBJECT_SELECTION_POINT,
            preferred_backend="sam2",
        )
        adapter = self._segmentation_provider()
        self._require_segmentation_adapter(adapter)
        if not adapter.is_available():
            raise ModelUnavailableError(
                f"{adapter.model_name} is not available in this environment."
            )
        logger.info(
            "segment_start model=%s point=(%d,%d) shape=%s route=%s",
            adapter.model_name,
            x,
            y,
            image.shape,
            routing.model,
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
        result.metadata.setdefault("routing", routing.to_dict())
        return result

    def select_smart(
        self,
        image: np.ndarray,
        *,
        x: int | None = None,
        y: int | None = None,
        text_prompt: str | None = None,
        selection_mode: str = "smart",
        detection_index: int | None = None,
        grounding_backend: str = "grounding_dino",
    ):
        """Rank segmentation candidates with deterministic heuristics.

        ``selection_mode``:
        - ``smart`` — text path when prompt provided, else point path
        - ``point`` — explicit click segmentation (with multimask ranking)
        - ``text`` — explicit grounding + box path (all detections ranked)
        """
        from models.types import SmartSelectionResult
        from pipelines.selection_quality import rank_candidates, score_mask_candidate

        image = validate_image(image)
        mode = selection_mode.strip().lower()
        if mode not in {"smart", "point", "text"}:
            raise PipelineValidationError(
                f"Unknown selection_mode '{selection_mode}'. Use smart, point, or text."
            )

        if mode == "point":
            if x is None or y is None:
                raise PipelineValidationError("Point mode requires x and y coordinates.")
            return self._select_smart_point(
                image, x, y, selection_mode=mode, rank_fn=rank_candidates, score_fn=score_mask_candidate
            )

        if mode == "text":
            prompt = self._validate_text_prompt(text_prompt or "")
            return self._select_smart_text(
                image,
                prompt,
                selection_mode=mode,
                detection_index=detection_index,
                grounding_backend=grounding_backend,
                rank_fn=rank_candidates,
                score_fn=score_mask_candidate,
            )

        # smart — route by available input without reinterpreting explicit modes
        if text_prompt and text_prompt.strip():
            prompt = self._validate_text_prompt(text_prompt)
            return self._select_smart_text(
                image,
                prompt,
                selection_mode=mode,
                detection_index=detection_index,
                grounding_backend=grounding_backend,
                rank_fn=rank_candidates,
                score_fn=score_mask_candidate,
            )
        if x is not None and y is not None:
            return self._select_smart_point(
                image, x, y, selection_mode=mode, rank_fn=rank_candidates, score_fn=score_mask_candidate
            )
        raise PipelineValidationError(
            "Smart selection requires a text prompt or point coordinates."
        )

    def _select_smart_point(self, image, x, y, *, selection_mode, rank_fn, score_fn):
        from models.types import SmartSelectionResult

        self._validate_point(image, x, y)
        routing = self._route_backend(
            operation=RoutingOperation.SEGMENT_POINT,
            capability=RoutingCapability.OBJECT_SELECTION_POINT,
            preferred_backend="sam2",
        )
        adapter = self._segmentation_provider()
        self._require_segmentation_adapter(adapter)
        if not adapter.is_available():
            raise ModelUnavailableError(f"{adapter.model_name} is not available.")

        logger.info("select_smart_point point=(%d,%d) mode=%s", x, y, selection_mode)
        t0 = time.perf_counter()
        try:
            self._prepare_local_adapter(adapter)
            if hasattr(adapter, "segment_point_candidates"):
                candidates = adapter.segment_point_candidates(image, x, y)
            else:
                candidates = [adapter.infer(image, x, y)]
        except (ModelLoadError, ModelInferenceError, ModelUnavailableError):
            raise
        except Exception as exc:
            raise ModelInferenceError("Smart point selection failed.") from exc
        finally:
            if self._unload_between_stages:
                self._release_local_adapter(adapter)

        scored = [
            score_fn(
                seg.mask,
                candidate_id=f"sam2_point_{idx}",
                point_xy=(x, y),
                sam_confidence=seg.confidence,
            )
            for idx, seg in enumerate(candidates)
        ]
        try:
            ranking = rank_fn(scored)
        except ValueError as exc:
            raise ModelInferenceError(
                "No valid mask candidates for the selected point."
            ) from exc
        best_idx = int(ranking.selected.candidate_id.rsplit("_", 1)[-1])
        segmentation = candidates[best_idx]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        segmentation.metadata.setdefault("pipeline_latency_ms", round(elapsed_ms, 3))
        segmentation.metadata.setdefault("routing", routing.to_dict())
        segmentation.metadata["selection_ranking"] = ranking.to_dict()

        return SmartSelectionResult(
            mask=segmentation.mask,
            method="point",
            confidence_tier=ranking.selected.tier.value,
            selection_mode=selection_mode,
            segmentation=segmentation,
            ranking=ranking.to_dict(),
            metadata={
                "point_xy": [x, y],
                "candidate_count": len(candidates),
                "pipeline_latency_ms": round(elapsed_ms, 3),
                "routing": routing.to_dict(),
            },
        )

    def _select_smart_text(
        self,
        image,
        prompt: str,
        *,
        selection_mode: str,
        detection_index: int | None,
        grounding_backend: str,
        rank_fn,
        score_fn,
    ):
        from models.types import SmartSelectionResult

        grounding = self._run_grounding(image, prompt, grounding_backend)
        if not grounding.detections:
            raise ModelInferenceError(f"No objects found for prompt '{prompt}'.")

        detections = list(grounding.detections)
        if detection_index is not None:
            if detection_index < 0 or detection_index >= len(detections):
                raise PipelineValidationError(
                    f"detection_index {detection_index} out of range "
                    f"(found {len(detections)})."
                )
            detections = [detections[detection_index]]

        adapter = self._segmentation_provider()
        self._require_segmentation_adapter(adapter)
        if not adapter.is_available():
            raise ModelUnavailableError(f"{adapter.model_name} is not available.")

        logger.info(
            "select_smart_text prompt=%r detections=%d mode=%s",
            prompt,
            len(detections),
            selection_mode,
        )
        t0 = time.perf_counter()
        segmentations: list[SegmentationResult] = []
        try:
            self._prepare_local_adapter(adapter)
            for det_idx, detection in enumerate(detections):
                x1, y1, x2, y2 = detection.as_xyxy_int()
                if hasattr(adapter, "segment_box"):
                    seg = adapter.segment_box(image, x1, y1, x2, y2)
                else:
                    seg = adapter.infer(image, x1, y1)
                segmentations.append(seg)
        except (ModelLoadError, ModelInferenceError, ModelUnavailableError):
            raise
        except Exception as exc:
            raise ModelInferenceError("Smart text selection failed.") from exc
        finally:
            if self._unload_between_stages:
                self._release_local_adapter(adapter)

        scored = []
        for det_idx, (detection, seg) in enumerate(zip(detections, segmentations)):
            x1, y1, x2, y2 = detection.as_xyxy_int()
            scored.append(
                score_fn(
                    seg.mask,
                    candidate_id=f"detection_{det_idx}",
                    box_xyxy=(float(x1), float(y1), float(x2), float(y2)),
                    sam_confidence=seg.confidence,
                )
            )

        if not scored:
            raise ModelInferenceError(f"No objects found for prompt '{prompt}'.")

        try:
            ranking = rank_fn(scored)
        except ValueError as exc:
            raise ModelInferenceError(
                f"No valid mask candidates for prompt '{prompt}'."
            ) from exc
        best_idx = int(ranking.selected.candidate_id.replace("detection_", ""))
        segmentation = segmentations[best_idx]
        selected_detection = detections[best_idx]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        segmentation.metadata.setdefault("pipeline_latency_ms", round(elapsed_ms, 3))
        segmentation.metadata["selection_ranking"] = ranking.to_dict()

        return SmartSelectionResult(
            mask=segmentation.mask,
            method="text",
            confidence_tier=ranking.selected.tier.value,
            selection_mode=selection_mode,
            segmentation=segmentation,
            grounding=grounding,
            selected_detection=selected_detection,
            ranking=ranking.to_dict(),
            metadata={
                "prompt": prompt,
                "detection_count": len(grounding.detections),
                "selected_detection_index": best_idx,
                "grounding_model": grounding.model,
                "segmentation_model": segmentation.model,
                "pipeline_latency_ms": round(elapsed_ms, 3),
            },
        )

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
        routing = self._route_backend(
            operation=RoutingOperation.INPAINT,
            capability=RoutingCapability.LOCALIZED_INPAINT,
            preferred_backend=backend,
        )
        backend = routing.model
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
        result.metadata.setdefault("routing", routing.to_dict())
        logger.info(
            "inpaint_done model=%s latency_ms=%.3f adapter_ms=%s",
            result.model,
            elapsed_ms,
            result.latency_ms,
        )
        return result

    def generate_candidates(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        backend: str = "moebius",
        *,
        count: int = 1,
        params: InpaintParams | None = None,
        base_seed: int | None = None,
    ) -> InpaintCandidatesResult:
        """Generate ``count`` inpainting candidates (1 or 2) and rank them."""
        from evaluation.candidate_ranking import rank_inpaint_candidates
        from evaluation.reproducibility import sha256_bytes

        validate_candidate_count(count)
        image = validate_image(image)
        mask = require_non_empty_mask(validate_mask(mask, image=image), stage="inpaint")
        routing = self._route_backend(
            operation=RoutingOperation.INPAINT,
            capability=RoutingCapability.LOCALIZED_INPAINT,
            preferred_backend=backend,
        )
        backend = routing.model
        adapter = self._resolve_inpaint_adapter(backend)
        if not adapter.is_available():
            raise ModelUnavailableError(
                f"{adapter.model_name} ({backend}) is not available."
            )

        p = params or InpaintParams()
        resolved_base = resolve_base_seed(base_seed if base_seed is not None else p.seed)
        seeds = derive_candidate_seeds(resolved_base, count)

        logger.info(
            "generate_candidates_start model=%s backend=%s count=%d base_seed=%d",
            adapter.model_name,
            adapter.backend_type.value,
            count,
            resolved_base,
        )

        raw_candidates: list[tuple[str, InpaintingResult, int | None]] = []
        total_t0 = time.perf_counter()
        try:
            self._prepare_local_adapter(adapter)
            for index, seed in enumerate(seeds):
                candidate_id = f"candidate_{index + 1}"
                candidate_params = InpaintParams(
                    num_steps=p.num_steps,
                    guidance_scale=p.guidance_scale,
                    strength=p.strength,
                    paste=p.paste,
                    noise_offset=p.noise_offset,
                    image_size=p.image_size,
                    seed=seed,
                )
                result = adapter.infer(image, mask, candidate_params)
                raw_candidates.append((candidate_id, result, seed))
        except (ModelLoadError, ModelInferenceError, ModelUnavailableError):
            raise
        except Exception as exc:
            raise ModelInferenceError("Candidate generation failed.") from exc
        finally:
            if self._unload_between_stages:
                self._release_local_adapter(adapter)

        total_ms = (time.perf_counter() - total_t0) * 1000.0
        ranking = rank_inpaint_candidates(
            image,
            mask,
            [(cid, res.result) for cid, res, _ in raw_candidates],
        )

        by_id = {cid: (res, seed) for cid, res, seed in raw_candidates}
        ordered: list[InpaintCandidate] = []
        for scored in ranking.ordered:
            result, seed = by_id[scored.candidate_id]
            gen_params = dict(result.metadata)
            gen_params["seed"] = seed
            ordered.append(
                InpaintCandidate(
                    candidate_id=scored.candidate_id,
                    result=result.result,
                    seed=seed,
                    latency_ms=result.latency_ms,
                    memory_mb=result.memory_mb,
                    model=result.model,
                    backend=result.backend,
                    output_hash=scored.output_hash or sha256_bytes(result.result.tobytes()),
                    validity_status=scored.validity_status,
                    generation_params=gen_params,
                    metadata={
                        "candidate_score": scored.score,
                        "score_components": {
                            k: v.to_dict() for k, v in scored.components.items()
                        },
                    },
                )
            )

        ranking_dict = ranking.to_dict()
        ranking_dict["eval"] = ranking.to_eval_record(
            candidate_count=count,
            selected_candidate_id=ranking.selected.candidate_id,
        )

        logger.info(
            "generate_candidates_done model=%s count=%d selected=%s total_ms=%.3f",
            adapter.model_name,
            count,
            ranking.selected.candidate_id,
            total_ms,
        )

        return InpaintCandidatesResult(
            candidates=ordered,
            selected_candidate_id=ranking.selected.candidate_id,
            ranking=ranking_dict,
            model=ordered[0].model if ordered else adapter.model_name,
            backend=ordered[0].backend if ordered else adapter.backend_type,
            metadata={
                "candidate_count": count,
                "base_seed": resolved_base,
                "derived_seeds": seeds,
                "pipeline_latency_ms": round(total_ms, 3),
                "routing": routing.to_dict(),
            },
        )

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
        if not is_automatic_backend(backend) and not localized_edit_supported(backend):
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

    def edit_by_instruction(
        self,
        image: np.ndarray,
        instruction: str,
        *,
        backend: str = "instruct_pix2pix",
        params: InstructionEditParams | None = None,
        mask: np.ndarray | None = None,
    ) -> InstructionEditResult:
        """Global instruction edit on the full image. Masks are not supported."""
        if mask is not None:
            raise UnsupportedEditIntentError(
                f"Backend '{backend}' does not support mask-conditioned editing."
            )
        if not is_automatic_backend(backend) and not global_instruction_edit_supported(backend):
            raise PipelineBackendError(
                f"Backend '{backend}' does not support global instruction editing."
            )
        image = validate_image(image)
        prompt = self._validate_text_prompt(instruction)
        routing = self._route_backend(
            operation=RoutingOperation.EDIT_BY_INSTRUCTION,
            capability=RoutingCapability.GLOBAL_INSTRUCTION_EDIT,
            preferred_backend=backend,
        )
        backend = routing.model
        adapter = self._resolve_instruction_edit_adapter(backend)
        if not adapter.is_available():
            raise ModelUnavailableError(
                f"{adapter.model_name} ({backend}) is not available."
            )
        logger.info(
            "edit_by_instruction_start backend=%s intent=%s instruction_len=%d",
            backend,
            EditIntent.GLOBAL_INSTRUCTION_EDIT.value,
            len(prompt),
        )
        t0 = time.perf_counter()
        try:
            if not adapter.is_loaded:
                adapter.load()
            result = adapter.infer(image, prompt, params)
        except (ModelLoadError, ModelInferenceError, ModelUnavailableError):
            raise
        except Exception as exc:
            raise ModelInferenceError("Instruction editing failed.") from exc

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        if result.latency_ms is None or result.latency_ms == 0:
            result.metadata.setdefault("pipeline_latency_ms", round(elapsed_ms, 3))
        result.metadata.setdefault("routing", routing.to_dict())
        logger.info(
            "edit_by_instruction_done model=%s latency_ms=%.3f",
            result.model,
            elapsed_ms,
        )
        return result

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
        routing = self._route_backend(
            operation=RoutingOperation.SELECT_BY_TEXT,
            capability=RoutingCapability.OBJECT_SELECTION_TEXT,
            preferred_backend=grounding_backend,
        )
        key = routing.model
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
        result.metadata.setdefault("routing", routing.to_dict())
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

    def _resolve_instruction_edit_adapter(self, backend: str) -> InstructionEditAdapter:
        key = backend.strip().lower()
        if key not in SUPPORTED_INSTRUCTION_EDIT_BACKENDS:
            raise PipelineBackendError(
                f"Unsupported instruction-edit backend '{backend}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_INSTRUCTION_EDIT_BACKENDS))}."
            )
        adapter = self._instruction_edit_provider(key)
        self._require_instruction_edit_adapter(adapter)
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
    def _require_instruction_edit_adapter(adapter: ModelAdapter) -> InstructionEditAdapter:
        if not isinstance(adapter, InstructionEditAdapter):
            raise PipelineValidationError(
                "Configured instruction-edit adapter is invalid."
            )
        return adapter

    def _pipeline_availability_probe(self, model_id: str) -> bool:
        """Probe availability via injected providers (tests) or registry."""
        key = model_id.strip().lower()
        try:
            if key == "sam2":
                return self._segmentation_provider().is_available()
            if key in SUPPORTED_GROUNDING_BACKENDS:
                return self._grounding_provider(key).is_available()
            if key in SUPPORTED_INPAINT_BACKENDS:
                return self._inpaint_provider(key).is_available()
            if key in SUPPORTED_INSTRUCTION_EDIT_BACKENDS:
                return self._instruction_edit_provider(key).is_available()
        except Exception:
            pass
        return get_adapter(key).is_available()

    def _route_backend(
        self,
        *,
        operation: RoutingOperation,
        capability: RoutingCapability,
        preferred_backend: str | None,
        execution_preference: ExecutionPreference = ExecutionPreference.LOCAL_FIRST,
    ):
        try:
            return route(
                RoutingRequest(
                    operation=operation,
                    required_capability=capability,
                    preferred_backend=preferred_backend,
                    execution_preference=execution_preference,
                ),
                availability_probe=self._pipeline_availability_probe,
            )
        except RoutingError as exc:
            message = str(exc)
            if "does not support" in message:
                raise PipelineBackendError(message) from exc
            raise ModelUnavailableError(message) from exc

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
