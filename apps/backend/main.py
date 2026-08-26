"""FastAPI MVP entrypoint for PixelForge image editing."""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse, Response

from apps.backend.dependencies import configure_cors, get_editing_service
from apps.backend.errors import register_exception_handlers
from apps.backend.schemas import (
    HealthResponse,
    InpaintMetadata,
    ModelsResponse,
    ModelInfo,
    RemoveObjectMetadata,
    SegmentMetadata,
)
from apps.backend.services import (
    ImageEditingService,
    backend_label,
    decode_upload_image,
    decode_upload_mask,
    image_to_png_bytes,
    mask_to_png_bytes,
    parse_inpaint_params,
    png_response_headers,
    segmentation_backend,
)
from pipelines.types import MaskRefinementOps

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    application = FastAPI(
        title="PixelForge",
        description="MVP image-editing API over SAM 2 segmentation and adapter inpainting.",
        version="0.1.0",
    )
    configure_cors(application)
    register_exception_handlers(application)
    application.include_router(build_router())
    return application


def build_router():
    from fastapi import APIRouter

    router = APIRouter()

    @router.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse()

    @router.get("/models", response_model=ModelsResponse)
    def models(
        service: Annotated[ImageEditingService, Depends(get_editing_service)],
    ) -> ModelsResponse:
        return ModelsResponse(models=[ModelInfo(**entry) for entry in service.list_models()])

    @router.post("/segment")
    async def segment(
        service: Annotated[ImageEditingService, Depends(get_editing_service)],
        image: UploadFile = File(..., description="RGB image (PNG/JPEG/WebP)."),
        x: int = Form(..., description="Prompt X coordinate (column)."),
        y: int = Form(..., description="Prompt Y coordinate (row)."),
    ) -> Response:
        rgb = await decode_upload_image(image)
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
        service: Annotated[ImageEditingService, Depends(get_editing_service)],
        image: UploadFile = File(..., description="RGB image."),
        mask: UploadFile = File(
            ...,
            description="Mask PNG: white (255) = inpaint, black (0) = preserve.",
        ),
        backend: str = Form("moebius"),
        num_steps: int | None = Form(None),
        guidance_scale: float | None = Form(None),
        strength: float | None = Form(None),
        paste: bool | None = Form(None),
        noise_offset: float | None = Form(None),
        image_size: int | None = Form(None),
    ) -> Response:
        rgb = await decode_upload_image(image)
        mask_arr = await decode_upload_mask(mask, image=rgb)
        params = parse_inpaint_params(
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            strength=strength,
            paste=paste,
            noise_offset=noise_offset,
            image_size=image_size,
        )
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

    @router.post("/remove-object")
    async def remove_object(
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

    return router


app = create_app()
