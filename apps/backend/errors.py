"""HTTP-facing errors and exception handlers."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from models.errors import (
    ModelInferenceError,
    ModelLoadError,
    ModelUnavailableError,
    PixelForgeModelError,
)
from pipelines.errors import PipelineBackendError, PipelineValidationError

logger = logging.getLogger(__name__)


class InvalidInputError(Exception):
    """Client supplied invalid or unusable input (HTTP 400)."""


class ResourceNotFoundError(Exception):
    """Requested model or resource does not exist (HTTP 404)."""


def error_body(code: str, message: str, **extra: Any) -> dict[str, Any]:
    body: dict[str, Any] = {"error": code, "message": message}
    body.update(extra)
    return body


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(InvalidInputError)
    async def _invalid_input(_request: Request, exc: InvalidInputError) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content=error_body("invalid_input", str(exc)),
        )

    @app.exception_handler(PipelineValidationError)
    async def _pipeline_validation(
        _request: Request, exc: PipelineValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content=error_body("invalid_input", str(exc)),
        )

    @app.exception_handler(PipelineBackendError)
    async def _pipeline_backend(_request: Request, exc: PipelineBackendError) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content=error_body("unsupported_backend", str(exc)),
        )

    @app.exception_handler(ValueError)
    async def _value_error(_request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content=error_body("invalid_input", str(exc)),
        )

    @app.exception_handler(ResourceNotFoundError)
    async def _not_found(_request: Request, exc: ResourceNotFoundError) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content=error_body("not_found", str(exc)),
        )

    @app.exception_handler(ModelUnavailableError)
    async def _model_unavailable(
        _request: Request, exc: ModelUnavailableError
    ) -> JSONResponse:
        message = str(exc)
        status = 404 if message.startswith("Unknown model") else 503
        code = "model_not_found" if status == 404 else "model_unavailable"
        return JSONResponse(
            status_code=status,
            content=error_body(code, message),
        )

    @app.exception_handler(ModelLoadError)
    async def _model_load(_request: Request, exc: ModelLoadError) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content=error_body("model_load_failed", str(exc)),
        )

    @app.exception_handler(ModelInferenceError)
    async def _model_inference(
        _request: Request, exc: ModelInferenceError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content=error_body("inference_failed", str(exc)),
        )

    @app.exception_handler(PixelForgeModelError)
    async def _model_error(_request: Request, exc: PixelForgeModelError) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content=error_body("model_error", str(exc)),
        )

    @app.exception_handler(RequestValidationError)
    async def _request_validation(
        _request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=error_body("validation_error", "Request validation failed.", details=exc.errors()),
        )

    @app.exception_handler(Exception)
    async def _unexpected(_request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled_error")
        return JSONResponse(
            status_code=500,
            content=error_body("internal_error", "An unexpected error occurred."),
        )
