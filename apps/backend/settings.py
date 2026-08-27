"""Centralized runtime settings loaded from environment variables.

Deployment-specific values belong here — never hardcode secrets.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

_DEFAULT_CORS_ORIGINS = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return float(raw)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off"}


def _env_csv(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return tuple(origin.strip() for origin in raw.split(",") if origin.strip())


@dataclass(frozen=True)
class Settings:
    """Application-wide limits and deployment configuration."""

    # API
    api_host: str
    api_port: int
    cors_origins: tuple[str, ...]
    request_id_header: str

    # Upload / image bounds
    max_upload_bytes: int
    max_mask_upload_bytes: int
    min_image_dimension: int
    max_image_dimension: int
    max_image_pixels: int
    max_prompt_length: int
    max_instruction_length: int
    max_candidate_count: int
    max_morph_pixels: int

    # Concurrency
    max_concurrent_generations: int

    # Worker / subprocess
    worker_startup_timeout_sec: float
    worker_request_timeout_sec: float
    subprocess_timeout_sec: float
    persistent_moebius_enabled: bool

    # Client-visible operation timeout hint (seconds)
    operation_timeout_sec: float


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        api_host=os.environ.get("PIXELFORGE_API_HOST", "127.0.0.1").strip(),
        api_port=_env_int("PIXELFORGE_API_PORT", 8000),
        cors_origins=_env_csv("PIXELFORGE_CORS_ORIGINS", _DEFAULT_CORS_ORIGINS),
        request_id_header=os.environ.get(
            "PIXELFORGE_REQUEST_ID_HEADER", "X-Request-ID"
        ).strip()
        or "X-Request-ID",
        max_upload_bytes=_env_int("PIXELFORGE_MAX_UPLOAD_BYTES", 25 * 1024 * 1024),
        max_mask_upload_bytes=_env_int(
            "PIXELFORGE_MAX_MASK_UPLOAD_BYTES", 10 * 1024 * 1024
        ),
        min_image_dimension=_env_int("PIXELFORGE_MIN_IMAGE_DIMENSION", 8),
        max_image_dimension=_env_int("PIXELFORGE_MAX_IMAGE_DIMENSION", 4096),
        max_image_pixels=_env_int("PIXELFORGE_MAX_IMAGE_PIXELS", 16_777_216),
        max_prompt_length=_env_int("PIXELFORGE_MAX_PROMPT_LENGTH", 512),
        max_instruction_length=_env_int("PIXELFORGE_MAX_INSTRUCTION_LENGTH", 512),
        max_candidate_count=_env_int("PIXELFORGE_MAX_CANDIDATE_COUNT", 2),
        max_morph_pixels=_env_int("PIXELFORGE_MAX_MORPH_PIXELS", 64),
        max_concurrent_generations=_env_int("PIXELFORGE_MAX_CONCURRENT_GENERATIONS", 1),
        worker_startup_timeout_sec=_env_float("PIXELFORGE_WORKER_STARTUP_TIMEOUT", 180.0),
        worker_request_timeout_sec=_env_float("PIXELFORGE_WORKER_REQUEST_TIMEOUT", 600.0),
        subprocess_timeout_sec=_env_float("PIXELFORGE_SUBPROCESS_TIMEOUT", 600.0),
        persistent_moebius_enabled=_env_bool("PIXELFORGE_MOEBIUS_PERSISTENT_WORKER", True),
        operation_timeout_sec=_env_float("PIXELFORGE_OPERATION_TIMEOUT", 600.0),
    )


def reset_settings_cache() -> None:
    """Clear cached settings (tests only)."""
    get_settings.cache_clear()
