"""Evaluation record types and metric status contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class MetricKind(str, Enum):
    """How a metric value was obtained."""

    MEASURED = "MEASURED"
    """Directly observed (e.g. wall-clock latency)."""

    COMPUTED = "COMPUTED"
    """Derived from inputs without an external reference."""

    REFERENCE_DEPENDENT = "REFERENCE_DEPENDENT"
    """Requires ground-truth or reference data."""

    UNAVAILABLE = "UNAVAILABLE"
    """Could not be computed (missing inputs or dependency)."""

    UNDEFINED = "UNDEFINED"
    """Formula undefined for this input (e.g. zero denominator)."""


@dataclass(frozen=True)
class MetricResult:
    """One named metric with explicit availability status."""

    name: str
    value: float | None
    kind: MetricKind
    unit: str | None = None
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "kind": self.kind.value,
            "unit": self.unit,
            "note": self.note,
        }


@dataclass
class StageTiming:
    """Latency for one pipeline stage."""

    stage: str
    latency_ms: float
    kind: MetricKind = MetricKind.MEASURED

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "latency_ms": self.latency_ms,
            "kind": self.kind.value,
        }


@dataclass
class EvaluationRecord:
    """Compact evaluation record for one experiment run."""

    experiment_id: str
    model: str
    backend: str
    operation: str
    image_shape: tuple[int, ...]
    mask_shape: tuple[int, int] | None
    mask_area: int | None
    timings: list[StageTiming] = field(default_factory=list)
    memory_mb: float | None = None
    seed: int | None = None
    timestamp: float | None = None
    model_commit: str | None = None
    environment: str | None = None
    metrics: list[MetricResult] = field(default_factory=list)
    reproducibility: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "model": self.model,
            "backend": self.backend,
            "operation": self.operation,
            "image_shape": list(self.image_shape),
            "mask_shape": list(self.mask_shape) if self.mask_shape else None,
            "mask_area": self.mask_area,
            "timings": [t.to_dict() for t in self.timings],
            "memory_mb": self.memory_mb,
            "seed": self.seed,
            "timestamp": self.timestamp,
            "model_commit": self.model_commit,
            "environment": self.environment,
            "metrics": [m.to_dict() for m in self.metrics],
            "reproducibility": self.reproducibility,
            "metadata": self.metadata,
        }


@dataclass
class ExperimentConfig:
    """Minimal experiment configuration for the runner."""

    experiment_id: str
    model: str
    backend: str
    operation: str
    environment: str | None = None
    model_commit: str | None = None
    seed: int | None = None
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "model": self.model,
            "backend": self.backend,
            "operation": self.operation,
            "environment": self.environment,
            "model_commit": self.model_commit,
            "seed": self.seed,
            "parameters": self.parameters,
        }
