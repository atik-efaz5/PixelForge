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
from apps.backend.isolated_runner import ground_via_isolated_env, inpaint_via_isolated_env
from models.errors import ModelLoadError
from models.registry import get_adapter, known_models
from models.types import (
    BackendType,
    InpaintParams,
    InpaintingResult,
    InstructionEditParams,
    InstructionEditResult,
    SegmentationResult,
    TextSelectionResult,
    pil_rgb_to_array,
    validate_image,
    validate_mask,
)
from pipelines.editing_capabilities import BACKEND_EDIT_CAPABILITIES, EditIntent
from pipelines.orchestration.image_edit_pipeline import ImageEditPipeline
from pipelines.types import ImageEditPipelineResult, MaskRefinementOps

logger = logging.getLogger(__name__)

# MVP upload limits (pixels). Tune via env later if needed.
_MAX_IMAGE_PIXELS = 16_777_216  # 4096×4096
_MASK_INPAINT_THRESHOLD = 128

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


async def decode_upload_image(upload: UploadFile) -> np.ndarray:
    """Decode a multipart image upload to H×W×3 uint8 RGB."""
    if upload.content_type and not upload.content_type.startswith("image/"):
        raise InvalidInputError(f"Expected an image upload, got {upload.content_type}.")
    raw = await upload.read()
    if not raw:
        raise InvalidInputError("Image upload is empty.")
    try:
        pil = Image.open(io.BytesIO(raw))
        pil.load()
    except Exception as exc:
        raise InvalidInputError("Could not decode image upload.") from exc
    if pil.width * pil.height > _MAX_IMAGE_PIXELS:
        raise InvalidInputError("Image exceeds the maximum allowed pixel count.")
    return pil_rgb_to_array(pil)


async def decode_upload_mask(
    upload: UploadFile,
    *,
    image: np.ndarray | None = None,
) -> np.ndarray:
    """Decode a mask PNG to bool H×W (white/255 = inpaint)."""
    raw = await upload.read()
    if not raw:
        raise InvalidInputError("Mask upload is empty.")
    try:
        pil = Image.open(io.BytesIO(raw))
        pil.load()
    except Exception as exc:
        raise InvalidInputError("Could not decode mask upload.") from exc
    gray = np.asarray(pil.convert("L"))
    if gray.ndim != 2:
        raise InvalidInputError("Mask must decode to a single-channel image.")
    mask = gray >= _MASK_INPAINT_THRESHOLD
    if image is not None:
        validate_mask(mask, image=image)
    else:
        validate_mask(mask)
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
) -> InpaintParams | None:
    fields = {
        "num_steps": num_steps,
        "guidance_scale": guidance_scale,
        "strength": strength,
        "paste": paste,
        "noise_offset": noise_offset,
        "image_size": image_size,
    }
    if all(value is None for value in fields.values()):
        return None
    return InpaintParams(**fields)


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
    return InstructionEditParams(**fields)


def segmentation_backend() -> str:
    """Backend label for the default segmentation adapter."""
    adapter = get_adapter("sam2")
    return adapter.backend_type.value


def backend_label(result_backend: BackendType) -> str:
    return result_backend.value
