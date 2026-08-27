"""FastAPI MVP entrypoint for PixelForge image editing."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from apps.backend.concurrency import generation_slot
from apps.backend.dependencies import configure_cors, get_editing_service
from apps.backend.errors import InvalidInputError, register_exception_handlers
from apps.backend.isolated_runner import shutdown_persistent_workers
from apps.backend.middleware import RequestContextMiddleware
from apps.backend.settings import get_settings
from apps.backend.schemas import (
    EditByInstructionMetadata,
    EditingCapabilitiesResponse,
    EditingCapabilityEntry,
    HealthResponse,
    InpaintMetadata,
    ModelsResponse,
    ModelInfo,
    RemoveObjectMetadata,
    RoutingResponse,
    SegmentMetadata,
    SelectByTextMetadata,
    SelectSmartMetadata,
)
from apps.backend.services import (
    ImageEditingService,
    backend_label,
    build_inpaint_candidates_metadata,
    build_multipart_inpaint_response,
    decode_upload_image,
    decode_upload_mask,
    image_to_png_bytes,
    mask_to_png_bytes,
    parse_inpaint_params,
    parse_instruction_edit_params,
    png_response_headers,
    segmentation_backend,
)
from apps.backend.validation import (
    validate_candidate_count_field,
    validate_morph_amount,
    validate_point,
    validate_text_field,
)
from pipelines.types import MaskRefinementOps

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _app_lifespan(_application: FastAPI):
    yield
    shutdown_persistent_workers()


def create_app() -> FastAPI:
    application = FastAPI(
        title="PixelForge",
        description="MVP image-editing API over SAM 2 segmentation and adapter inpainting.",
        version="0.1.0",
        lifespan=_app_lifespan,
    )
    application.add_middleware(RequestContextMiddleware)
    configure_cors(application)
    register_exception_handlers(application)
    application.include_router(build_router())
    return application


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "-")


def build_router():
    from fastapi import APIRouter

    router = APIRouter()

    @router.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        settings = get_settings()
        return HealthResponse(
            status="ok",
            version="0.1.0",
            max_upload_bytes=settings.max_upload_bytes,
            max_concurrent_generations=settings.max_concurrent_generations,
        )

    @router.get("/models", response_model=ModelsResponse)
    def models(
        service: Annotated[ImageEditingService, Depends(get_editing_service)],
    ) -> ModelsResponse:
        return ModelsResponse(models=[ModelInfo(**entry) for entry in service.list_models()])

    @router.get("/capabilities", response_model=EditingCapabilitiesResponse)
    def capabilities(
        service: Annotated[ImageEditingService, Depends(get_editing_service)],
    ) -> EditingCapabilitiesResponse:
        return EditingCapabilitiesResponse(
            capabilities=[
                EditingCapabilityEntry(**entry)
                for entry in service.list_editing_capabilities()
            ]
        )

    @router.get("/routing", response_model=RoutingResponse)
    def routing(
        service: Annotated[ImageEditingService, Depends(get_editing_service)],
    ) -> RoutingResponse:
        payload = service.list_routing()
        return RoutingResponse(**payload)

    @router.post("/edit-by-instruction")
    async def edit_by_instruction(
        request: Request,
        service: Annotated[ImageEditingService, Depends(get_editing_service)],
        image: UploadFile = File(..., description="RGB image."),
        instruction: str = Form(
            ...,
            description="Global edit instruction, e.g. make the sky look like sunset.",
        ),
        backend: str = Form("instruct_pix2pix"),
        num_steps: int | None = Form(None),
        guidance_text: float | None = Form(None),
        guidance_image: float | None = Form(None),
        resolution: int | None = Form(None),
    ) -> Response:
        settings = get_settings()
        rgb = await decode_upload_image(image)
        prompt = validate_text_field(
            instruction,
            field_name="instruction",
            max_length=settings.max_instruction_length,
        )
        params = parse_instruction_edit_params(
            num_steps=num_steps,
            guidance_text=guidance_text,
            guidance_image=guidance_image,
            resolution=resolution,
        )
        with generation_slot(operation="edit_by_instruction", request_id=_request_id(request)):
            result = service.edit_by_instruction(
                rgb,
                prompt,
                backend=backend,
                params=params,
            )
        meta = EditByInstructionMetadata(
            model=result.model,
            backend=backend_label(result.backend),
            instruction=result.instruction,
            latency_ms=result.latency_ms,
            memory_mb=result.memory_mb,
            metadata=result.metadata,
        )
        headers = png_response_headers(meta.model_dump())
        headers["Content-Disposition"] = 'inline; filename="edit.png"'
        return Response(
            content=image_to_png_bytes(result.result),
            media_type="image/png",
            headers=headers,
        )

    @router.post("/segment")
    async def segment(
        service: Annotated[ImageEditingService, Depends(get_editing_service)],
        image: UploadFile = File(..., description="RGB image (PNG/JPEG/WebP)."),
        x: int = Form(..., description="Prompt X coordinate (column)."),
        y: int = Form(..., description="Prompt Y coordinate (row)."),
    ) -> Response:
        rgb = await decode_upload_image(image)
        validate_point(rgb, x, y)
        result = service.segment(rgb, x, y)
        meta = SegmentMetadata(
            confidence=result.confidence,
            model=result.model,
            backend=segmentation_backend(),
            method=result.method,
            metadata=result.metadata,
        )
        headers = png_response_headers(meta.model_dump())
        headers["Content-Disposition"] = 'inline; filename="mask.png"'
        return Response(
            content=mask_to_png_bytes(result.mask),
            media_type="image/png",
            headers=headers,
        )

    @router.post("/inpaint")
    async def inpaint(
        request: Request,
        service: Annotated[ImageEditingService, Depends(get_editing_service)],
        image: UploadFile = File(..., description="RGB image."),
        mask: UploadFile = File(
            ...,
            description="Mask PNG: white (255) = inpaint, black (0) = preserve.",
        ),
        backend: str = Form("moebius"),
        candidate_count: int = Form(
            1,
            description="Number of candidates to generate (1 or 2). Default 1.",
        ),
        num_steps: int | None = Form(None),
        guidance_scale: float | None = Form(None),
        strength: float | None = Form(None),
        paste: bool | None = Form(None),
        noise_offset: float | None = Form(None),
        image_size: int | None = Form(None),
        seed: int | None = Form(None),
    ) -> Response:
        rgb = await decode_upload_image(image)
        mask_arr = await decode_upload_mask(mask, image=rgb, require_nonempty=True)
        count = validate_candidate_count_field(candidate_count)
        params = parse_inpaint_params(
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            paste=paste,
            noise_offset=noise_offset,
            image_size=image_size,
            seed=seed,
        )
        if count == 1:
            with generation_slot(operation="inpaint", request_id=_request_id(request)):
                result = service.inpaint(rgb, mask_arr, backend=backend, params=params)
            meta = InpaintMetadata(
                model=result.model,
                backend=backend_label(result.backend),
                latency_ms=result.latency_ms,
                memory_mb=result.memory_mb,
                metadata=result.metadata,
            )
            headers = png_response_headers(meta.model_dump())
            headers["Content-Disposition"] = 'inline; filename="inpaint.png"'
            return Response(
                content=image_to_png_bytes(result.result),
                media_type="image/png",
                headers=headers,
            )

        with generation_slot(operation="inpaint_candidates", request_id=_request_id(request)):
            candidates_result = service.generate_candidates(
                rgb,
                mask_arr,
                backend=backend,
                params=params,
                count=count,
            )
        metadata = build_inpaint_candidates_metadata(candidates_result)
        png_parts = [
            (candidate.candidate_id, image_to_png_bytes(candidate.result))
            for candidate in candidates_result.candidates
        ]
        body, content_type = build_multipart_inpaint_response(metadata, png_parts)
        return Response(content=body, media_type=content_type)

    @router.post("/remove-object")
    async def remove_object(
        request: Request,
        service: Annotated[ImageEditingService, Depends(get_editing_service)],
        image: UploadFile = File(..., description="RGB image."),
        x: int = Form(..., description="Object point X (column)."),
        y: int = Form(..., description="Object point Y (row)."),
        backend: str = Form("moebius"),
        dilate: int = Form(0),
        erode: int = Form(0),
        add_mask: UploadFile | None = File(
            None,
            description="Optional PNG region to add to the segmentation mask.",
        ),
        remove_mask: UploadFile | None = File(
            None,
            description="Optional PNG region to remove from the segmentation mask.",
        ),
        num_steps: int | None = Form(None),
        guidance_scale: float | None = Form(None),
        strength: float | None = Form(None),
        paste: bool | None = Form(None),
        noise_offset: float | None = Form(None),
        image_size: int | None = Form(None),
    ) -> Response:
        rgb = await decode_upload_image(image)
        validate_point(rgb, x, y)
        dilate = validate_morph_amount(dilate, field_name="dilate")
        erode = validate_morph_amount(erode, field_name="erode")
        refinement: MaskRefinementOps | None = None
        if add_mask is not None or remove_mask is not None or dilate or erode:
            add_arr = (
                await decode_upload_mask(add_mask, image=rgb) if add_mask is not None else None
            )
            remove_arr = (
                await decode_upload_mask(remove_mask, image=rgb)
                if remove_mask is not None
                else None
            )
            refinement = MaskRefinementOps(
                add=add_arr,
                remove=remove_arr,
                dilate=dilate,
                erode=erode,
            )
        params = parse_inpaint_params(
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            paste=paste,
            noise_offset=noise_offset,
            image_size=image_size,
        )
        with generation_slot(operation="remove_object", request_id=_request_id(request)):
            result = service.remove_object(
                rgb,
                x,
                y,
                backend=backend,
                refinement=refinement,
                params=params,
            )
        meta = RemoveObjectMetadata(
            model=result.selected_model,
            backend=backend_label(result.backend),
            segmentation_model=(
                result.segmentation.model if result.segmentation is not None else None
            ),
            latency_ms=result.latency.total_ms,
            segmentation_ms=result.latency.segmentation_ms,
            inpainting_ms=result.latency.inpainting_ms,
            metadata=result.metadata,
        )
        headers = png_response_headers(meta.model_dump())
        headers["Content-Disposition"] = 'inline; filename="result.png"'
        return Response(
            content=image_to_png_bytes(result.result),
            media_type="image/png",
            headers=headers,
        )

    @router.post("/select-by-text")
    async def select_by_text(
        service: Annotated[ImageEditingService, Depends(get_editing_service)],
        image: UploadFile = File(..., description="RGB image."),
        prompt: str = Form(..., description="Object description, e.g. dog or red car."),
        detection_index: int = Form(0, description="Which detection to segment when multiple."),
        grounding_backend: str = Form("grounding_dino"),
    ) -> Response:
        settings = get_settings()
        rgb = await decode_upload_image(image)
        clean_prompt = validate_text_field(
            prompt,
            field_name="prompt",
            max_length=settings.max_prompt_length,
        )
        if detection_index < 0:
            raise InvalidInputError("detection_index must be non-negative.")
        result = service.select_by_text(
            rgb,
            clean_prompt,
            detection_index=detection_index,
            grounding_backend=grounding_backend,
        )
        detections = [
            {
                "index": idx,
                "label": det.label,
                "confidence": det.confidence,
                "box_xyxy": [det.x1, det.y1, det.x2, det.y2],
            }
            for idx, det in enumerate(result.grounding.detections)
        ]
        box = result.selected_detection
        meta = SelectByTextMetadata(
            prompt=result.grounding.prompt,
            model=result.grounding.model,
            segmentation_model=result.segmentation.model,
            grounding_backend=result.grounding.backend.value,
            confidence=result.segmentation.confidence,
            method=result.segmentation.method,
            detection_index=result.metadata.get("detection_index", detection_index),
            detection_count=len(result.grounding.detections),
            selected_label=box.label,
            selected_box_xyxy=[box.x1, box.y1, box.x2, box.y2],
            detections=detections,
            metadata={
                **result.metadata,
                **result.grounding.metadata,
                **result.segmentation.metadata,
            },
        )
        headers = png_response_headers(meta.model_dump())
        headers["Content-Disposition"] = 'inline; filename="mask.png"'
        return Response(
            content=mask_to_png_bytes(result.mask),
            media_type="image/png",
            headers=headers,
        )

    @router.post("/select-smart")
    async def select_smart(
        service: Annotated[ImageEditingService, Depends(get_editing_service)],
        image: UploadFile = File(..., description="RGB image."),
        selection_mode: str = Form("smart", description="smart | point | text"),
        x: int | None = Form(None, description="Prompt X (column) for point/smart click."),
        y: int | None = Form(None, description="Prompt Y (row) for point/smart click."),
        prompt: str | None = Form(None, description="Object description for text/smart."),
        detection_index: int | None = Form(
            None, description="Optional detection index for text mode."
        ),
        grounding_backend: str = Form("grounding_dino"),
    ) -> Response:
        settings = get_settings()
        rgb = await decode_upload_image(image)
        if x is not None and y is not None:
            validate_point(rgb, x, y)
        text_prompt = None
        if prompt is not None and prompt.strip():
            text_prompt = validate_text_field(
                prompt,
                field_name="prompt",
                max_length=settings.max_prompt_length,
            )
        if detection_index is not None and detection_index < 0:
            raise InvalidInputError("detection_index must be non-negative.")
        result = service.select_smart(
            rgb,
            x=x,
            y=y,
            text_prompt=text_prompt,
            selection_mode=selection_mode,
            detection_index=detection_index,
            grounding_backend=grounding_backend,
        )
        detections: list[dict] = []
        grounding_backend_value = None
        selected_label = None
        selected_box = None
        if result.grounding is not None:
            grounding_backend_value = result.grounding.backend.value
            detections = [
                {
                    "index": idx,
                    "label": det.label,
                    "confidence": det.confidence,
                    "box_xyxy": [det.x1, det.y1, det.x2, det.y2],
                }
                for idx, det in enumerate(result.grounding.detections)
            ]
        if result.selected_detection is not None:
            box = result.selected_detection
            selected_label = box.label
            selected_box = [box.x1, box.y1, box.x2, box.y2]

        meta = SelectSmartMetadata(
            selection_mode=result.selection_mode,
            method=result.method,
            confidence_tier=result.confidence_tier,
            model=result.grounding.model if result.grounding else result.segmentation.model,
            segmentation_model=result.segmentation.model,
            grounding_backend=grounding_backend_value,
            confidence=result.segmentation.confidence,
            prompt=result.metadata.get("prompt"),
            point_xy=result.metadata.get("point_xy"),
            detection_index=result.metadata.get("selected_detection_index"),
            detection_count=result.metadata.get("detection_count"),
            selected_label=selected_label,
            selected_box_xyxy=selected_box,
            detections=detections,
            ranking=result.ranking,
            metadata={
                **result.metadata,
                **result.segmentation.metadata,
            },
        )
        headers = png_response_headers(meta.model_dump())
        headers["Content-Disposition"] = 'inline; filename="mask.png"'
        return Response(
            content=mask_to_png_bytes(result.mask),
            media_type="image/png",
            headers=headers,
        )

    return router


app = create_app()
