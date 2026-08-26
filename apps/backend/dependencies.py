"""FastAPI dependencies and development CORS settings."""

from __future__ import annotations

import os
from functools import lru_cache

from fastapi.middleware.cors import CORSMiddleware

from apps.backend.services import ImageEditingService
from pipelines.orchestration.image_edit_pipeline import ImageEditPipeline

# Narrow dev origins; override via PIXELFORGE_CORS_ORIGINS (comma-separated).
_DEFAULT_CORS_ORIGINS = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)


def cors_origins() -> list[str]:
    raw = os.environ.get("PIXELFORGE_CORS_ORIGINS", "").strip()
    if not raw:
        return list(_DEFAULT_CORS_ORIGINS)
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def configure_cors(app) -> None:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins(),
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )


@lru_cache(maxsize=1)
def get_pipeline() -> ImageEditPipeline:
    return ImageEditPipeline()


def get_editing_service() -> ImageEditingService:
    return ImageEditingService(pipeline=get_pipeline())
