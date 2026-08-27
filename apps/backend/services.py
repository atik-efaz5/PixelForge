"""Application service layer over :class:`ImageEditPipeline`."""

from __future__ import annotations

import io
import json
import logging
from typing import Any

import numpy as np
from fastapi import UploadFile
from PIL import Image

from apps.backend.errors import InvalidInputError
from apps.backend.settings import get_settings
from apps.backend.validation import (
    MASK_INPAINT_THRESHOLD,
    MAX_IMAGE_PIXELS,
    decode_mask_bytes,
    normalize_upload_image,
    require_nonempty_mask,
    sanitize_filename,
    validate_inpaint_params_fields,
    validate_instruction_edit_params_fields,
)
from apps.backend.isolated_runner import ground_via_isolated_env, inpaint_via_isolated_env
from models.errors import ModelLoadError
from models.registry import get_adapter, known_models
from models.router import list_routing_catalog
from models.types import (
    BackendType,
    InpaintCandidatesResult,
    InpaintParams,
    InpaintingResult,
    InstructionEditParams,
    InstructionEditResult,
    SegmentationResult,
    TextSelectionResult,
    SmartSelectionResult,
    pil_rgb_to_array,
    validate_image,
    validate_mask,
)
from pipelines.editing_capabilities import BACKEND_EDIT_CAPABILITIES, EditIntent
from pipelines.orchestration.image_edit_pipeline import ImageEditPipeline
from pipelines.types import ImageEditPipelineResult, MaskRefinementOps

logger = logging.getLogger(__name__)

# MVP upload limits — see apps.backend.validation for bounds.
_MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
_MASK_INPAINT_THRESHOLD = MASK_INPAINT_THRESHOLD

# Mask PNG contract (documented for API clients):
# - Upload: grayscale or RGB PNG where pixel value >= 128 means inpaint (True).
# - Download: single-channel PNG, 255 = inpaint, 0 = preserve.


class ImageEditingService:
    """Thin wrapper delegating to :class:`ImageEditPipeline`."""

    def __init__(self, pipeline: ImageEditPipeline | None = None) -> None:
        self._pipeline = pipeline or ImageEditPipeline()

    @property
    def pipeline(self) -> ImageEditPipeline:
        return self._pipeline

    def list_models(self) -> list[dict[str, Any]]:
        """Probe adapter availability without loading weights."""
        entries: list[dict[str, Any]] = []
        for model_id in known_models():
            adapter = get_adapter(model_id)
            if model_id == "sam2":
                role = "segmentation"
            elif model_id == "grounding_dino":
                role = "grounding"
            elif model_id == "instruct_pix2pix":
                role = "instruction_editing"
            else:
                role = "inpainting"
            entries.append(
                {
                    "id": model_id,
                    "name": adapter.model_name,
                    "role": role,
                    "backend": adapter.backend_type.value,
                    "available": adapter.is_available(),
                    "loaded": adapter.is_loaded,
                    "status": adapter.status.value,
                }
            )
        return entries

    def list_editing_capabilities(self) -> list[dict[str, Any]]:
        """Declare which edit intents each editing backend actually supports."""
        rows: list[dict[str, Any]] = []
        intent_values = lambda cap: {i.value for i in cap.supported_intents}
        for backend_id, cap in BACKEND_EDIT_CAPABILITIES.items():
            intents = intent_values(cap)
            rows.append(
                {
                    "backend_id": backend_id,
                    "localized_inpaint": EditIntent.LOCALIZED_INPAINT.value in intents,
                    "semantic_replace": EditIntent.SEMANTIC_REPLACE.value in intents,
                    "global_instruction_edit": EditIntent.GLOBAL_INSTRUCTION_EDIT.value
                    in intents,
                    "mask_conditioned_edit": EditIntent.MASK_CONDITIONED_EDIT.value
                    in intents,
                    "accepts_text_instruction": cap.accepts_text_instruction,
                    "accepts_reference_image": cap.accepts_reference_image,
                    "notes": cap.notes,
                }
            )
        return rows

    def list_routing(self) -> dict[str, Any]:
        """Capability-aware routing catalog for ``GET /routing``."""
        return list_routing_catalog()

    def edit_localized(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        *,
        backend: str = "moebius",
        params: InpaintParams | None = None,
    ) -> InpaintingResult:
        """Localized mask edit only. Rejects non-empty text instructions at the pipeline layer."""
        logger.info("service_edit_localized backend=%s", backend)
        try:
            return self._pipeline.edit_localized(
                image, mask, backend=backend, params=params
            )
        except ModelLoadError as exc:
            if backend.strip().lower() != "moebius":
                raise
            logger.warning(
                "in-process Moebius load failed; using isolated env: %s", exc
            )
            return inpaint_via_isolated_env(image, mask, params=params)

    def edit_by_instruction(
        self,
        image: np.ndarray,
        instruction: str,
        *,
        backend: str = "instruct_pix2pix",
        params: InstructionEditParams | None = None,
    ) -> InstructionEditResult:
        """Global instruction edit via the configured cloud adapter."""
        logger.info("service_edit_by_instruction backend=%s", backend)
        return self._pipeline.edit_by_instruction(
            image, instruction, backend=backend, params=params
        )

    def segment(self, image: np.ndarray, x: int, y: int) -> SegmentationResult:
        logger.info("service_segment x=%d y=%d", x, y)
        return self._pipeline.segment(image, x, y)

    def inpaint(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        *,
        backend: str = "moebius",
        params: InpaintParams | None = None,
    ) -> InpaintingResult:
        logger.info("service_inpaint backend=%s", backend)
        try:
            return self._pipeline.inpaint(image, mask, backend=backend, params=params)
        except ModelLoadError as exc:
            if backend.strip().lower() != "moebius":
                raise
            logger.warning("in-process Moebius load failed; using isolated env: %s", exc)
            return inpaint_via_isolated_env(image, mask, params=params)

    def generate_candidates(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        *,
        backend: str = "moebius",
        params: InpaintParams | None = None,
        count: int = 1,
    ) -> InpaintCandidatesResult:
        logger.info("service_generate_candidates backend=%s count=%d", backend, count)
        try:
            return self._pipeline.generate_candidates(
                image, mask, backend=backend, count=count, params=params
            )
        except ModelLoadError as exc:
            if backend.strip().lower() != "moebius" or count != 1:
                raise
            logger.warning(
                "in-process Moebius load failed for single candidate; using isolated env: %s",
                exc,
            )
            single = inpaint_via_isolated_env(image, mask, params=params)
            from evaluation.candidate_ranking import rank_inpaint_candidates
            from evaluation.reproducibility import sha256_bytes
            from models.types import InpaintCandidate

            ranking = rank_inpaint_candidates(
                image, mask, [("candidate_1", single.result)]
            )
            scored = ranking.selected
            return InpaintCandidatesResult(
                candidates=[
                    InpaintCandidate(
                        candidate_id="candidate_1",
                        result=single.result,
                        seed=params.seed if params else None,
                        latency_ms=single.latency_ms,
                        memory_mb=single.memory_mb,
                        model=single.model,
                        backend=single.backend,
                        output_hash=scored.output_hash or sha256_bytes(single.result.tobytes()),
                        validity_status=scored.validity_status,
                        generation_params=dict(single.metadata),
                        metadata={
                            "candidate_score": scored.score,
                            "score_components": {
                                k: v.to_dict() for k, v in scored.components.items()
                            },
                        },
                    )
                ],
                selected_candidate_id="candidate_1",
                ranking=ranking.to_dict(),
                model=single.model,
                backend=single.backend,
                metadata={"isolated_fallback": True},
            )

    def select_by_text(
        self,
        image: np.ndarray,
        text_prompt: str,
        *,
        detection_index: int = 0,
        grounding_backend: str = "grounding_dino",
    ) -> TextSelectionResult:
        logger.info(
            "service_select_by_text prompt=%r detection_index=%d",
            text_prompt,
            detection_index,
        )
        try:
            return self._pipeline.select_by_text(
                image,
                text_prompt,
                detection_index=detection_index,
                grounding_backend=grounding_backend,
            )
        except ModelLoadError as exc:
            if grounding_backend.strip().lower() != "grounding_dino":
                raise
            logger.warning(
                "in-process Grounding DINO load failed; using isolated env: %s", exc
            )
            grounding = ground_via_isolated_env(image, text_prompt)
            return self._pipeline.select_from_grounding(
                image, grounding, detection_index=detection_index
            )

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
    ) -> SmartSelectionResult:
        logger.info(
            "service_select_smart mode=%s x=%s y=%s prompt=%r",
            selection_mode,
            x,
            y,
            text_prompt,
        )
        return self._pipeline.select_smart(
            image,
            x=x,
            y=y,
            text_prompt=text_prompt,
            selection_mode=selection_mode,
            detection_index=detection_index,
            grounding_backend=grounding_backend,
        )

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
        logger.info("service_remove_object x=%d y=%d backend=%s", x, y, backend)
        return self._pipeline.remove_object(
            image,
            x,
            y,
            backend=backend,
            refinement=refinement,
            params=params,
        )


async def read_upload_bytes(upload: UploadFile, *, max_bytes: int) -> bytes:
    """Read an upload in chunks with a hard byte-size cap."""
    safe_name = sanitize_filename(getattr(upload, "filename", None))
    logger.debug("upload_read name=%s max_bytes=%d", safe_name, max_bytes)
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise InvalidInputError(
                f"Upload exceeds maximum size ({max_bytes} bytes)."
            )
        chunks.append(chunk)
    return b"".join(chunks)


async def decode_upload_image(upload: UploadFile) -> np.ndarray:
    """Decode a multipart image upload to H×W×3 uint8 RGB."""
    settings = get_settings()
    raw = await read_upload_bytes(upload, max_bytes=settings.max_upload_bytes)
    return normalize_upload_image(raw, content_type=upload.content_type)


async def decode_upload_mask(
    upload: UploadFile,
    *,
    image: np.ndarray | None = None,
    require_nonempty: bool = False,
    context: str = "inpaint",
) -> np.ndarray:
    """Decode a mask PNG to bool H×W (white/255 = inpaint)."""
    settings = get_settings()
    raw = await read_upload_bytes(upload, max_bytes=settings.max_mask_upload_bytes)
    mask = decode_mask_bytes(raw, image=image)
    if require_nonempty:
        require_nonempty_mask(mask, context=context)
    return mask


def mask_to_png_bytes(mask: np.ndarray) -> bytes:
    """Encode bool H×W mask as PNG (255 = inpaint)."""
    from models.types import mask_to_pil_l

    buf = io.BytesIO()
    mask_to_pil_l(mask).save(buf, format="PNG")
    return buf.getvalue()


def image_to_png_bytes(image: np.ndarray) -> bytes:
    """Encode H×W×3 uint8 RGB as PNG."""
    from models.types import image_to_pil_rgb

    buf = io.BytesIO()
    image_to_pil_rgb(image).save(buf, format="PNG")
    return buf.getvalue()


def png_response_headers(metadata: dict[str, Any]) -> dict[str, str]:
    """Attach JSON metadata via response headers for PNG endpoints."""
    headers = {}
    for key, value in metadata.items():
        if value is None:
            continue
        header = f"x-pf-{key.replace('_', '-')}"
        if isinstance(value, (dict, list)):
            headers[header] = json.dumps(value, separators=(",", ":"), default=str)
        else:
            headers[header] = str(value)
    return headers


def parse_inpaint_params(
    *,
    num_steps: int | None = None,
    guidance_scale: float | None = None,
    strength: float | None = None,
    paste: bool | None = None,
    noise_offset: float | None = None,
    image_size: int | None = None,
    seed: int | None = None,
) -> InpaintParams | None:
    fields = {
        "num_steps": num_steps,
        "guidance_scale": guidance_scale,
        "strength": strength,
        "paste": paste,
        "noise_offset": noise_offset,
        "image_size": image_size,
        "seed": seed,
    }
    if all(value is None for value in fields.values()):
        return None
    validate_inpaint_params_fields(
        num_steps=num_steps,
        guidance_scale=guidance_scale,
        strength=strength,
        noise_offset=noise_offset,
        image_size=image_size,
        seed=seed,
    )
    return InpaintParams(**fields)


def build_inpaint_candidates_metadata(result: InpaintCandidatesResult) -> dict[str, Any]:
    """Serialize multi-candidate metadata for multipart responses."""
    total_latency = sum(c.latency_ms for c in result.candidates)
    memory = next((c.memory_mb for c in result.candidates if c.memory_mb is not None), None)
    candidates = []
    for rank, candidate in enumerate(result.candidates, start=1):
        candidates.append(
            {
                "candidate_id": candidate.candidate_id,
                "rank": rank,
                "score": candidate.metadata.get("candidate_score", 0.0),
                "seed": candidate.seed,
                "latency_ms": candidate.latency_ms,
                "memory_mb": candidate.memory_mb,
                "output_hash": candidate.output_hash,
                "validity_status": candidate.validity_status,
                "generation_params": candidate.generation_params,
                "score_components": candidate.metadata.get("score_components", {}),
            }
        )
    return {
        "model": result.model,
        "backend": backend_label(result.backend),
        "candidate_count": len(result.candidates),
        "selected_candidate_id": result.selected_candidate_id,
        "candidates": candidates,
        "ranking": result.ranking,
        "latency_ms": round(total_latency, 3),
        "memory_mb": memory,
        "metadata": result.metadata,
    }


def build_multipart_inpaint_response(
    metadata: dict[str, Any],
    candidates: list[tuple[str, bytes]],
) -> tuple[bytes, str]:
    """Build multipart/form-data body with JSON metadata and PNG parts."""
    boundary = "pixelforge-candidate-" + "0" * 16
    body = bytearray()
    meta_bytes = json.dumps(metadata, separators=(",", ":"), default=str).encode("utf-8")
    body.extend(f"--{boundary}\r\n".encode())
    body.extend(
        b'Content-Disposition: form-data; name="metadata"\r\n'
        b"Content-Type: application/json\r\n\r\n"
    )
    body.extend(meta_bytes)
    body.extend(b"\r\n")
    for candidate_id, png_bytes in candidates:
        body.extend(f"--{boundary}\r\n".encode())
        body.extend(
            f'Content-Disposition: form-data; name="{candidate_id}"; '
            f'filename="{candidate_id}.png"\r\n'.encode()
        )
        body.extend(b"Content-Type: image/png\r\n\r\n")
        body.extend(png_bytes)
        body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())
    content_type = f"multipart/form-data; boundary={boundary}"
    return bytes(body), content_type


def parse_instruction_edit_params(
    *,
    num_steps: int | None = None,
    guidance_text: float | None = None,
    guidance_image: float | None = None,
    resolution: int | None = None,
) -> InstructionEditParams | None:
    fields = {
        "num_steps": num_steps,
        "guidance_text": guidance_text,
        "guidance_image": guidance_image,
        "resolution": resolution,
    }
    if all(value is None for value in fields.values()):
        return None
    validate_instruction_edit_params_fields(
        num_steps=num_steps,
        guidance_text=guidance_text,
        guidance_image=guidance_image,
        resolution=resolution,
    )
    return InstructionEditParams(**fields)


def segmentation_backend() -> str:
    """Backend label for the default segmentation adapter."""
    adapter = get_adapter("sam2")
    return adapter.backend_type.value


def backend_label(result_backend: BackendType) -> str:
    return result_backend.value
