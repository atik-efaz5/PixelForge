"""HTTP middleware for request context and structured access logs."""

from __future__ import annotations

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from apps.backend.settings import get_settings

logger = logging.getLogger(__name__)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Attach a request id and emit lightweight structured access logs."""

    async def dispatch(
        self, request: Request, call_next: RequestResponseEndpoint
    ) -> Response:
        settings = get_settings()
        header = settings.request_id_header
        request_id = request.headers.get(header) or uuid.uuid4().hex[:16]
        request.state.request_id = request_id

        start = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception:
            duration_ms = (time.perf_counter() - start) * 1000.0
            logger.exception(
                "request_failed id=%s method=%s path=%s latency_ms=%.1f",
                request_id,
                request.method,
                request.url.path,
                duration_ms,
            )
            raise
        else:
            duration_ms = (time.perf_counter() - start) * 1000.0
            response.headers[header] = request_id
            logger.info(
                "request_complete id=%s method=%s path=%s status=%d latency_ms=%.1f",
                request_id,
                request.method,
                request.url.path,
                status_code,
                duration_ms,
            )
            return response
