"""Pydantic schemas for JSON API responses."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"


class ModelInfo(BaseModel):
    id: str
    name: str
    role: str
    backend: str
    available: bool
    loaded: bool
    status: str


class ModelsResponse(BaseModel):
    models: list[ModelInfo]


class SegmentMetadata(BaseModel):
    """JSON metadata for ``POST /segment``. Mask pixels are returned as PNG."""

    confidence: float | None
    model: str
    backend: str
    method: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class InpaintMetadata(BaseModel):
    """JSON metadata for ``POST /inpaint``. Result pixels are returned as PNG."""

    model: str
    backend: str
    latency_ms: float
    memory_mb: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RemoveObjectMetadata(BaseModel):
    """JSON metadata for ``POST /remove-object``. Result pixels are returned as PNG."""

    model: str
    backend: str
    segmentation_model: str | None = None
    latency_ms: float
    segmentation_ms: float | None = None
    inpainting_ms: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class SelectByTextMetadata(BaseModel):
    """JSON metadata for ``POST /select-by-text``. Mask pixels are returned as PNG."""

    prompt: str
    model: str
    segmentation_model: str
    grounding_backend: str
    confidence: float | None
    method: str
    detection_index: int
    detection_count: int
    selected_label: str
    selected_box_xyxy: list[float]
    detections: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EditingCapabilityEntry(BaseModel):
    """Declared edit intents for one editing backend."""

    backend_id: str
    localized_inpaint: bool
    semantic_replace: bool
    global_instruction_edit: bool
    mask_conditioned_edit: bool
    accepts_text_instruction: bool
    accepts_reference_image: bool
    notes: str


class EditingCapabilitiesResponse(BaseModel):
    """JSON response for ``GET /capabilities``."""

    capabilities: list[EditingCapabilityEntry]


class EditByInstructionMetadata(BaseModel):
    """JSON metadata for ``POST /edit-by-instruction``. Result pixels are returned as PNG."""

    model: str
    backend: str
    instruction: str
    latency_ms: float
    memory_mb: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
