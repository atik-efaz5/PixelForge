"""FastAPI dependencies and CORS settings."""

from __future__ import annotations

from functools import lru_cache

from fastapi.middleware.cors import CORSMiddleware

from apps.backend.services import ImageEditingService
from apps.backend.settings import get_settings
from pipelines.orchestration.image_edit_pipeline import ImageEditPipeline

# Headers the frontend reads from PNG inference responses.
_EXPOSE_HEADERS = [
    "X-Request-ID",
    "Content-Disposition",
    "x-pf-model",
    "x-pf-backend",
    "x-pf-latency-ms",
    "x-pf-memory-mb",
    "x-pf-metadata",
    "x-pf-confidence",
    "x-pf-method",
    "x-pf-prompt",
    "x-pf-segmentation-model",
    "x-pf-grounding-backend",
    "x-pf-detection-index",
    "x-pf-detection-count",
    "x-pf-selected-label",
    "x-pf-selected-box-xyxy",
    "x-pf-detections",
    "x-pf-instruction",
    "x-pf-segmentation-ms",
    "x-pf-inpainting-ms",
    "x-pf-selection-mode",
    "x-pf-confidence-tier",
    "x-pf-segmentation-model",
    "x-pf-point-xy",
    "x-pf-ranking",
]


def configure_cors(app) -> None:
    settings = get_settings()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_origin_regex=settings.cors_origin_regex or None,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Accept", settings.request_id_header],
        expose_headers=_EXPOSE_HEADERS,
    )


@lru_cache(maxsize=1)
def get_pipeline() -> ImageEditPipeline:
    return ImageEditPipeline()


def get_editing_service() -> ImageEditingService:
    return ImageEditingService(pipeline=get_pipeline())
