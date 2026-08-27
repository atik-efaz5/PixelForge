"""In-process concurrency guards for expensive model operations."""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from typing import Iterator

from apps.backend.errors import ServiceBusyError
from apps.backend.settings import get_settings

logger = logging.getLogger(__name__)

_generation_slots = threading.BoundedSemaphore(value=1)
_slots_configured = False
_slots_lock = threading.Lock()


def _ensure_slots() -> None:
    global _generation_slots, _slots_configured
    with _slots_lock:
        if _slots_configured:
            return
        limit = max(1, get_settings().max_concurrent_generations)
        _generation_slots = threading.BoundedSemaphore(value=limit)
        _slots_configured = True


def reset_generation_slots() -> None:
    """Reconfigure semaphore from settings (tests only)."""
    global _slots_configured
    with _slots_lock:
        _slots_configured = False
    _ensure_slots()


@contextmanager
def generation_slot(*, operation: str, request_id: str | None = None) -> Iterator[None]:
    """Acquire a generation slot or reject when at capacity."""
    _ensure_slots()
    acquired = _generation_slots.acquire(blocking=False)
    if not acquired:
        logger.warning(
            "generation_rejected operation=%s request_id=%s reason=at_capacity",
            operation,
            request_id or "-",
        )
        raise ServiceBusyError(
            "Another generation request is in progress. Please retry shortly."
        )
    logger.info(
        "generation_start operation=%s request_id=%s",
        operation,
        request_id or "-",
    )
    try:
        yield
    finally:
        _generation_slots.release()
        logger.info(
            "generation_end operation=%s request_id=%s",
            operation,
            request_id or "-",
        )
