"""Deterministic model routing for PixelForge operations.

The router selects which registered adapter should handle a request. It does
not perform inference — adapters do.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from models.registry import get_adapter, known_models
from models.types import BackendType

# Models with local runtime validation in this repository (Phase 3–17).
RUNTIME_VALIDATED_MODELS = frozenset({"sam2", "grounding_dino", "moebius"})


class RoutingCapability(str, Enum):
    """High-level capability required to fulfill an operation."""

    OBJECT_SELECTION_POINT = "object_selection_point"
    OBJECT_SELECTION_TEXT = "object_selection_text"
    OBJECT_SELECTION_BOX = "object_selection_box"
    LOCALIZED_INPAINT = "localized_inpaint"
    MASKED_INPAINT = "masked_inpaint"
    GLOBAL_INSTRUCTION_EDIT = "global_instruction_edit"


class RoutingOperation(str, Enum):
    """User-facing pipeline operation (must not be silently remapped)."""

    SEGMENT_POINT = "segment_point"
    SELECT_BY_TEXT = "select_by_text"
    SEGMENT_BOX = "segment_box"
    INPAINT = "inpaint"
    EDIT_BY_INSTRUCTION = "edit_by_instruction"


class ExecutionPreference(str, Enum):
    LOCAL_FIRST = "local_first"
    CLOUD_FIRST = "cloud_first"
    FASTEST_AVAILABLE = "fastest_available"
    QUALITY_FIRST = "quality_first"


class QualityPreference(str, Enum):
    DEFAULT = "default"
    HIGH = "high"


class LatencyPreference(str, Enum):
    DEFAULT = "default"
    LOW = "low"


@dataclass(frozen=True)
class RoutingRequest:
    operation: RoutingOperation
    required_capability: RoutingCapability
    preferred_backend: str | None = None
    execution_preference: ExecutionPreference = ExecutionPreference.LOCAL_FIRST
    quality_preference: QualityPreference = QualityPreference.DEFAULT
    latency_preference: LatencyPreference = LatencyPreference.DEFAULT


@dataclass(frozen=True)
class RoutingFallback:
    model: str
    backend: str
    available: bool
    runtime_validated: bool
    reason: str


@dataclass(frozen=True)
class RoutingDecision:
    model: str
    backend: str
    reason: str
    available: bool
    operation: str
    required_capability: str
    runtime_validated: bool
    fallbacks: tuple[RoutingFallback, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "backend": self.backend,
            "reason": self.reason,
            "available": self.available,
            "operation": self.operation,
            "required_capability": self.required_capability,
            "runtime_validated": self.runtime_validated,
            "fallbacks": [
                {
                    "model": fb.model,
                    "backend": fb.backend,
                    "available": fb.available,
                    "runtime_validated": fb.runtime_validated,
                    "reason": fb.reason,
                }
                for fb in self.fallbacks
            ],
        }


class RoutingError(Exception):
    """No compatible or available backend for the routing request."""

    def __init__(
        self,
        message: str,
        *,
        fallbacks: tuple[RoutingFallback, ...] = (),
    ) -> None:
        super().__init__(message)
        self.fallbacks = fallbacks


# Capability → ordered model candidates (deterministic priority within a tier).
_CAPABILITY_CANDIDATES: dict[RoutingCapability, tuple[tuple[str, BackendType], ...]] = {
    RoutingCapability.OBJECT_SELECTION_POINT: (("sam2", BackendType.LOCAL_MPS),),
    RoutingCapability.OBJECT_SELECTION_TEXT: (("grounding_dino", BackendType.CPU),),
    RoutingCapability.OBJECT_SELECTION_BOX: (("sam2", BackendType.LOCAL_MPS),),
    RoutingCapability.LOCALIZED_INPAINT: (
        ("moebius", BackendType.LOCAL_MPS),
        ("pixelhacker", BackendType.CLOUD_GPU),
    ),
    RoutingCapability.MASKED_INPAINT: (("pixelhacker", BackendType.CLOUD_GPU),),
    RoutingCapability.GLOBAL_INSTRUCTION_EDIT: (
        ("instruct_pix2pix", BackendType.CLOUD_GPU),
    ),
}

_OPERATION_CAPABILITY: dict[RoutingOperation, RoutingCapability] = {
    RoutingOperation.SEGMENT_POINT: RoutingCapability.OBJECT_SELECTION_POINT,
    RoutingOperation.SELECT_BY_TEXT: RoutingCapability.OBJECT_SELECTION_TEXT,
    RoutingOperation.SEGMENT_BOX: RoutingCapability.OBJECT_SELECTION_BOX,
    RoutingOperation.INPAINT: RoutingCapability.LOCALIZED_INPAINT,
    RoutingOperation.EDIT_BY_INSTRUCTION: RoutingCapability.GLOBAL_INSTRUCTION_EDIT,
}

_BACKEND_TIER_LOCAL = 0
_BACKEND_TIER_CPU = 1
_BACKEND_TIER_CLOUD = 2

# Automatic routing may select non-validated cloud models when these preferences apply.
_CLOUD_AUTOMATIC_PREFERENCES = frozenset(
    {ExecutionPreference.CLOUD_FIRST, ExecutionPreference.QUALITY_FIRST}
)


def is_automatic_backend(backend: str | None) -> bool:
    if backend is None:
        return True
    key = backend.strip().lower()
    return key in {"", "auto", "automatic"}


def capability_for_operation(operation: RoutingOperation) -> RoutingCapability:
    return _OPERATION_CAPABILITY[operation]


def _backend_tier(backend: BackendType) -> int:
    if backend == BackendType.LOCAL_MPS:
        return _BACKEND_TIER_LOCAL
    if backend == BackendType.CPU:
        return _BACKEND_TIER_CPU
    return _BACKEND_TIER_CLOUD


def _probe_availability(
    model_id: str,
    *,
    availability_probe: Callable[[str], bool] | None = None,
) -> bool:
    if availability_probe is not None:
        return availability_probe(model_id)
    return get_adapter(model_id).is_available()


def _build_fallbacks(
    candidates: tuple[tuple[str, BackendType], ...],
    *,
    availability_probe: Callable[[str], bool] | None = None,
    exclude: str | None = None,
) -> tuple[RoutingFallback, ...]:
    fallbacks: list[RoutingFallback] = []
    for model_id, backend in candidates:
        if exclude and model_id == exclude:
            continue
        available = _probe_availability(model_id, availability_probe=availability_probe)
        fallbacks.append(
            RoutingFallback(
                model=model_id,
                backend=backend.value,
                available=available,
                runtime_validated=model_id in RUNTIME_VALIDATED_MODELS,
                reason=_candidate_reason(model_id, backend, available),
            )
        )
    return tuple(fallbacks)


def _candidate_reason(model_id: str, backend: BackendType, available: bool) -> str:
    validated = model_id in RUNTIME_VALIDATED_MODELS
    if available and validated:
        return f"{model_id} is available on {backend.value}."
    if available and not validated:
        return f"{model_id} is configured on {backend.value} but not runtime-validated locally."
    if not validated:
        return f"{model_id} ({backend.value}) is not runtime-validated and is unavailable."
    return f"{model_id} ({backend.value}) is unavailable in this environment."


def _rank_candidates(
    candidates: tuple[tuple[str, BackendType], ...],
    preference: ExecutionPreference,
) -> list[tuple[str, BackendType]]:
    """Return candidates sorted by execution preference (stable, deterministic)."""
    ranked: list[tuple[str, BackendType, tuple[int, ...]]] = []
    for index, (model_id, backend) in enumerate(candidates):
        tier = _backend_tier(backend)
        if preference == ExecutionPreference.LOCAL_FIRST:
            sort_key = (tier, index)
        elif preference == ExecutionPreference.CLOUD_FIRST:
            sort_key = (-tier, index)
        elif preference == ExecutionPreference.FASTEST_AVAILABLE:
            fast_rank = (
                0
                if model_id in RUNTIME_VALIDATED_MODELS
                and tier == _BACKEND_TIER_LOCAL
                else 1
            )
            sort_key = (fast_rank, tier, index)
        elif preference == ExecutionPreference.QUALITY_FIRST:
            sort_key = (-index, tier)
        else:
            sort_key = (tier, index)
        ranked.append((model_id, backend, sort_key))
    ranked.sort(key=lambda item: item[2])
    return [(model_id, backend) for model_id, backend, _ in ranked]


def route(
    request: RoutingRequest,
    *,
    availability_probe: Callable[[str], bool] | None = None,
) -> RoutingDecision:
    """Select a model/backend for ``request`` without performing inference."""
    if request.required_capability != capability_for_operation(request.operation):
        raise RoutingError(
            f"Operation {request.operation.value} requires capability "
            f"{capability_for_operation(request.operation).value}, not "
            f"{request.required_capability.value}."
        )

    candidates = _CAPABILITY_CANDIDATES.get(request.required_capability, ())
    if not candidates:
        raise RoutingError(
            f"No routing candidates registered for {request.required_capability.value}."
        )

    # Explicit backend — never substitute capabilities.
    if request.preferred_backend and not is_automatic_backend(request.preferred_backend):
        model_id = request.preferred_backend.strip().lower()
        allowed = {m for m, _ in candidates}
        if model_id not in allowed:
            raise RoutingError(
                f"Backend '{model_id}' does not support "
                f"{request.required_capability.value}.",
                fallbacks=_build_fallbacks(candidates, availability_probe=availability_probe),
            )
        backend = next(b for m, b in candidates if m == model_id)
        available = _probe_availability(model_id, availability_probe=availability_probe)
        if not available:
            raise RoutingError(
                f"Requested backend '{model_id}' is not available for "
                f"{request.required_capability.value}.",
                fallbacks=_build_fallbacks(
                    candidates,
                    availability_probe=availability_probe,
                    exclude=model_id,
                ),
            )
        return RoutingDecision(
            model=model_id,
            backend=backend.value,
            reason=(
                f"Explicit backend '{model_id}' requested for "
                f"{request.operation.value}."
            ),
            available=True,
            operation=request.operation.value,
            required_capability=request.required_capability.value,
            runtime_validated=model_id in RUNTIME_VALIDATED_MODELS,
            fallbacks=_build_fallbacks(
                candidates,
                availability_probe=availability_probe,
                exclude=model_id,
            ),
        )

    ranked = _rank_candidates(candidates, request.execution_preference)
    fallbacks = _build_fallbacks(candidates, availability_probe=availability_probe)

    for model_id, backend in ranked:
        available = _probe_availability(model_id, availability_probe=availability_probe)
        if not available:
            continue
        # Automatic routing prefers runtime-validated local paths unless the
        # preference explicitly ranks cloud candidates (cloud_first / quality_first).
        if (
            request.execution_preference not in _CLOUD_AUTOMATIC_PREFERENCES
            and model_id not in RUNTIME_VALIDATED_MODELS
            and backend == BackendType.CLOUD_GPU
        ):
            continue
        reason = _automatic_reason(
            request,
            model_id=model_id,
            backend=backend,
        )
        return RoutingDecision(
            model=model_id,
            backend=backend.value,
            reason=reason,
            available=True,
            operation=request.operation.value,
            required_capability=request.required_capability.value,
            runtime_validated=model_id in RUNTIME_VALIDATED_MODELS,
            fallbacks=tuple(
                fb for fb in fallbacks if fb.model != model_id
            ),
        )

    raise RoutingError(
        f"No available backend supports {request.required_capability.value}.",
        fallbacks=fallbacks,
    )


def _automatic_reason(
    request: RoutingRequest,
    *,
    model_id: str,
    backend: BackendType,
) -> str:
    cap = request.required_capability.value.replace("_", " ")
    if request.execution_preference == ExecutionPreference.LOCAL_FIRST:
        pref = "local-first preference"
    elif request.execution_preference == ExecutionPreference.CLOUD_FIRST:
        pref = "cloud-first preference"
    elif request.execution_preference == ExecutionPreference.FASTEST_AVAILABLE:
        pref = "fastest-available preference"
    else:
        pref = "quality-first preference"
    return (
        f"Selected {model_id} on {backend.value} because {cap} is required for "
        f"{request.operation.value} ({pref})."
    )


def list_routing_catalog(
    *,
    availability_probe: Callable[[str], bool] | None = None,
) -> dict[str, Any]:
    """Return capability mappings and per-model availability for ``GET /routing``."""
    capabilities: list[dict[str, Any]] = []
    for capability, candidates in _CAPABILITY_CANDIDATES.items():
        models: list[dict[str, Any]] = []
        for model_id, backend in candidates:
            available = _probe_availability(model_id, availability_probe=availability_probe)
            adapter = get_adapter(model_id)
            models.append(
                {
                    "model": model_id,
                    "backend": backend.value,
                    "available": available,
                    "runtime_validated": model_id in RUNTIME_VALIDATED_MODELS,
                    "status": adapter.status.value,
                    "display_name": adapter.model_name,
                }
            )
        capabilities.append(
            {
                "capability": capability.value,
                "models": models,
            }
        )

    return {
        "capabilities": capabilities,
        "operations": [
            {
                "operation": op.value,
                "required_capability": cap.value,
            }
            for op, cap in _OPERATION_CAPABILITY.items()
        ],
        "execution_preferences": [p.value for p in ExecutionPreference],
        "automatic_backend_aliases": ["auto", "automatic"],
        "known_models": list(known_models()),
    }
