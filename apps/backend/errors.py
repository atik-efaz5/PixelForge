"""HTTP-facing errors and exception handlers."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from apps.backend.persistent_worker import WorkerCrashedError, WorkerTimeoutError
from apps.backend.settings import get_settings
from models.errors import (
    ModelInferenceError,
    ModelLoadError,
    ModelUnavailableError,
    PixelForgeModelError,
)
from pipelines.errors import (
    PipelineBackendError,
    PipelinePromptError,
    PipelineValidationError,
    UnsupportedEditIntentError,
)

logger = logging.getLogger(__name__)


class InvalidInputError(Exception):
    """Client supplied invalid or unusable input (HTTP 400)."""


class ResourceNotFoundError(Exception):
    """Requested model or resource does not exist (HTTP 404)."""


class ServiceBusyError(Exception):
    """Server is at generation capacity (HTTP 503)."""


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def error_body(
    code: str,
    message: str,
    *,
    request_id: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Structured error payload with legacy top-level fields for older clients."""
    body: dict[str, Any] = {
        "error": {"code": code, "message": message},
        # Legacy flat fields (do not remove — existing clients read these).
        "code": code,
        "message": message,
    }
    if request_id:
        body["request_id"] = request_id
        body["error"]["request_id"] = request_id
    body.update(extra)
    return body


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(InvalidInputError)
    async def _invalid_input(request: Request, exc: InvalidInputError) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content=error_body(
                "invalid_input", str(exc), request_id=_request_id(request)
            ),
        )

    @app.exception_handler(PipelinePromptError)
    async def _pipeline_prompt(request: Request, exc: PipelinePromptError) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content=error_body(
                "invalid_input", str(exc), request_id=_request_id(request)
            ),
        )

    @app.exception_handler(PipelineValidationError)
    async def _pipeline_validation(
        request: Request, exc: PipelineValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content=error_body(
                "invalid_input", str(exc), request_id=_request_id(request)
            ),
        )

    @app.exception_handler(PipelineBackendError)
    async def _pipeline_backend(request: Request, exc: PipelineBackendError) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content=error_body(
                "unsupported_backend", str(exc), request_id=_request_id(request)
            ),
        )

    @app.exception_handler(UnsupportedEditIntentError)
    async def _unsupported_edit_intent(
        request: Request, exc: UnsupportedEditIntentError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=400,
            content=error_body(
                "unsupported_edit_intent", str(exc), request_id=_request_id(request)
            ),
        )

    @app.exception_handler(ResourceNotFoundError)
    async def _not_found(request: Request, exc: ResourceNotFoundError) -> JSONResponse:
        return JSONResponse(
            status_code=404,
            content=error_body("not_found", str(exc), request_id=_request_id(request)),
        )

    @app.exception_handler(ServiceBusyError)
    async def _service_busy(request: Request, exc: ServiceBusyError) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content=error_body(
                "service_busy", str(exc), request_id=_request_id(request)
            ),
        )

    @app.exception_handler(WorkerTimeoutError)
    async def _worker_timeout(request: Request, exc: WorkerTimeoutError) -> JSONResponse:
        logger.warning("worker_timeout request_id=%s", _request_id(request))
        return JSONResponse(
            status_code=503,
            content=error_body(
                "worker_timeout",
                "Model worker timed out. Please retry.",
                request_id=_request_id(request),
            ),
        )

    @app.exception_handler(WorkerCrashedError)
    async def _worker_crashed(request: Request, exc: WorkerCrashedError) -> JSONResponse:
        logger.warning("worker_crashed request_id=%s error=%s", _request_id(request), exc)
        return JSONResponse(
            status_code=503,
            content=error_body(
                "worker_crashed",
                "Model worker exited unexpectedly. Please retry.",
                request_id=_request_id(request),
            ),
        )

    @app.exception_handler(ModelUnavailableError)
    async def _model_unavailable(
        request: Request, exc: ModelUnavailableError
    ) -> JSONResponse:
        message = str(exc)
        status = 404 if message.startswith("Unknown model") else 503
        code = "model_not_found" if status == 404 else "model_unavailable"
        return JSONResponse(
            status_code=status,
            content=error_body(code, message, request_id=_request_id(request)),
        )

    @app.exception_handler(ModelLoadError)
    async def _model_load(request: Request, exc: ModelLoadError) -> JSONResponse:
        return JSONResponse(
            status_code=503,
            content=error_body(
                "model_load_failed", str(exc), request_id=_request_id(request)
            ),
        )

    @app.exception_handler(ModelInferenceError)
    async def _model_inference(
        request: Request, exc: ModelInferenceError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content=error_body(
                "inference_failed", str(exc), request_id=_request_id(request)
            ),
        )

    @app.exception_handler(PixelForgeModelError)
    async def _model_error(request: Request, exc: PixelForgeModelError) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content=error_body(
                "model_error", str(exc), request_id=_request_id(request)
            ),
        )

    @app.exception_handler(RequestValidationError)
    async def _request_validation(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=error_body(
                "validation_error",
                "Request validation failed.",
                request_id=_request_id(request),
                details=exc.errors(),
            ),
        )

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled_error request_id=%s", _request_id(request))
        return JSONResponse(
            status_code=500,
            content=error_body(
                "internal_error",
                "An unexpected error occurred.",
                request_id=_request_id(request),
            ),
        )
