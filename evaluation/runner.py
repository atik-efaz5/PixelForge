"""Lightweight experiment runner for staged evaluations."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.metrics import (
    f1,
    iou,
    mask_area_ratio,
    outside_mask_preservation_score,
    psnr,
    ssim_unavailable,
)
from evaluation.reproducibility import build_reproducibility_record, sha256_array
from evaluation.types import EvaluationRecord, ExperimentConfig, MetricResult, StageTiming


StageFn = Callable[[], Any]


@dataclass
class ExperimentInputs:
    """Optional references and artifacts for metric collection."""

    image: np.ndarray | None = None
    mask: np.ndarray | None = None
    output: np.ndarray | None = None
    reference_mask: np.ndarray | None = None
    reference_image: np.ndarray | None = None
    memory_mb: float | None = None


@dataclass
class ExperimentRunner:
    """Execute named stages, record latency, and emit an evaluation report."""

    config: ExperimentConfig
    device: str | None = None
    inputs: ExperimentInputs = field(default_factory=ExperimentInputs)

    def run_stages(self, stages: dict[str, StageFn]) -> EvaluationRecord:
        timings: list[StageTiming] = []
        stage_outputs: dict[str, Any] = {}

        for name, fn in stages.items():
            t0 = time.perf_counter()
            stage_outputs[name] = fn()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            timings.append(StageTiming(stage=name, latency_ms=round(elapsed_ms, 3)))

        metrics = self._collect_metrics()
        hashes = sha256_array(
            self.inputs.image,
            self.inputs.mask,
            self.inputs.output,
        )

        record = EvaluationRecord(
            experiment_id=self.config.experiment_id,
            model=self.config.model,
            backend=self.config.backend,
            operation=self.config.operation,
            image_shape=tuple(self.inputs.image.shape) if self.inputs.image is not None else (),
            mask_shape=(
                (int(self.inputs.mask.shape[0]), int(self.inputs.mask.shape[1]))
                if self.inputs.mask is not None
                else None
            ),
            mask_area=int(self.inputs.mask.sum()) if self.inputs.mask is not None else None,
            timings=timings,
            memory_mb=self.inputs.memory_mb,
            seed=self.config.seed,
            timestamp=time.time(),
            model_commit=self.config.model_commit,
            environment=self.config.environment,
            metrics=metrics,
            reproducibility=build_reproducibility_record(
                config=self.config,
                device=self.device,
                image_hash=hashes["image_sha256"],
                mask_hash=hashes["mask_sha256"],
                output_hash=hashes["output_sha256"],
            ),
            metadata={"stage_outputs": list(stage_outputs.keys())},
        )
        return record

    def _collect_metrics(self) -> list[MetricResult]:
        results: list[MetricResult] = []

        if self.inputs.mask is not None:
            results.append(mask_area_ratio(self.inputs.mask))

        if self.inputs.mask is not None and self.inputs.reference_mask is not None:
            results.extend(
                [
                    iou(self.inputs.mask, self.inputs.reference_mask),
                    f1(self.inputs.mask, self.inputs.reference_mask),
                ]
            )

        if (
            self.inputs.image is not None
            and self.inputs.output is not None
            and self.inputs.reference_image is not None
        ):
            results.append(psnr(self.inputs.reference_image, self.inputs.output))

        if (
            self.inputs.image is not None
            and self.inputs.output is not None
            and self.inputs.mask is not None
        ):
            results.append(
                outside_mask_preservation_score(
                    self.inputs.image,
                    self.inputs.output,
                    self.inputs.mask,
                )
            )

        results.append(ssim_unavailable())
        return results

    def write_report(self, record: EvaluationRecord, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return out
