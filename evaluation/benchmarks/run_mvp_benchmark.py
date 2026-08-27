#!/usr/bin/env python3
"""Run a small reproducible real-world benchmark of the local PixelForge MVP.

Generates three synthetic 512×512 cases, runs real SAM2 / Grounding DINO /
Moebius inference in their conda environments, and writes:

  - evaluation/reports/mvp_benchmark.json
  - docs/experiments/MVP_BENCHMARK.md  (from JSON when --write-doc)

Artifacts (PNGs) go to outputs/mvp_benchmark/ (gitignored).

Usage::

    python -m evaluation.benchmarks.run_mvp_benchmark
    python -m evaluation.benchmarks.run_mvp_benchmark --orchestrate
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from evaluation.benchmarks.cases import BenchmarkCase, all_cases
from evaluation.metrics import f1, iou, mask_area_ratio
from evaluation.reproducibility import build_reproducibility_record, sha256_array, sha256_file
from evaluation.types import ExperimentConfig, MetricKind, MetricResult
from models.registry import get_adapter, reset_registry
from models.types import InpaintParams, validate_image, validate_mask

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "mvp_benchmark"
REPORT_JSON = PROJECT_ROOT / "evaluation" / "reports" / "mvp_benchmark.json"
REPORT_MD = PROJECT_ROOT / "docs" / "experiments" / "MVP_BENCHMARK.md"
LOCKFILE = PROJECT_ROOT / "research" / "upstream" / "LOCKFILE.md"

SAM2_COMMIT = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
MOEBIUS_COMMIT = "b88d462bacb9af6e7128a3b4cc4a07418bedfd61"
GSA_COMMIT = "126abe633ffe333e16e4a0a4e946bc1003caf757"

WARM_REPEATS = 3


@dataclass
class BenchmarkReport:
    """Top-level machine-readable benchmark record."""

    benchmark_id: str = "mvp_local_benchmark_v1"
    phase: str = "16"
    status: str = "NOT_RUN"
    timestamp_utc: str = ""
    pixelforge_commit: str | None = None
    hardware: dict[str, Any] = field(default_factory=dict)
    environments: dict[str, Any] = field(default_factory=dict)
    model_commits: dict[str, str] = field(default_factory=dict)
    methodology: dict[str, Any] = field(default_factory=dict)
    test_cases: list[dict[str, Any]] = field(default_factory=list)
    sam2: dict[str, Any] = field(default_factory=dict)
    grounding_dino: dict[str, Any] = field(default_factory=dict)
    sam2_box: dict[str, Any] = field(default_factory=dict)
    moebius: dict[str, Any] = field(default_factory=dict)
    end_to_end: dict[str, Any] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "phase": self.phase,
            "status": self.status,
            "timestamp_utc": self.timestamp_utc,
            "pixelforge_commit": self.pixelforge_commit,
            "hardware": self.hardware,
            "environments": self.environments,
            "model_commits": self.model_commits,
            "methodology": self.methodology,
            "test_cases": self.test_cases,
            "sam2": self.sam2,
            "grounding_dino": self.grounding_dino,
            "sam2_box": self.sam2_box,
            "moebius": self.moebius,
            "end_to_end": self.end_to_end,
            "limitations": self.limitations,
        }


def _git_head() -> str | None:
    try:
        return (
            subprocess.run(
                ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except Exception:
        return None


def _env_python(env_name: str) -> Path:
    override = os.environ.get(f"PIXELFORGE_{env_name.upper().replace('-', '_')}_PYTHON", "").strip()
    if override:
        return Path(override).expanduser()
    return Path(f"/opt/anaconda3/envs/{env_name}/bin/python")


def _collect_hardware() -> dict[str, Any]:
    def sysctl(key: str) -> str | None:
        try:
            return (
                subprocess.run(
                    ["sysctl", "-n", key],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout.strip()
            )
        except Exception:
            return None

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": sysctl("machdep.cpu.brand_string"),
        "physical_cores": sysctl("hw.physicalcpu"),
        "logical_cores": sysctl("hw.logicalcpu"),
        "unified_memory_bytes": sysctl("hw.memsize"),
    }


def _peak_rss_mib() -> float | None:
    try:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # macOS reports bytes; Linux reports KiB.
        if sys.platform == "darwin":
            return round(rss / (1024 * 1024), 2)
        return round(rss / 1024, 2)
    except Exception:
        return None


def _mps_driver_mib() -> float | None:
    try:
        import torch

        if hasattr(torch, "mps") and hasattr(torch.mps, "driver_allocated_memory"):
            return round(torch.mps.driver_allocated_memory() / (1024 * 1024), 2)
    except Exception:
        return None
    return None


def _metric(name: str, value: float | None, kind: MetricKind, **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "name": name,
        "value": value,
        "kind": kind.value,
    }
    payload.update(extra)
    return payload


def _save_png(path: Path, arr: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if arr.ndim == 2:
        Image.fromarray((arr.astype(np.uint8) * 255), mode="L").save(path)
    else:
        Image.fromarray(arr, mode="RGB").save(path)
    return sha256_file(path)


def _prepare_workdir(work_dir: Path) -> dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)
    cases_meta: list[dict[str, Any]] = []
    for image, case in all_cases():
        img_path = work_dir / f"{case.case_id}.png"
        _save_png(img_path, image)
        meta = case.to_dict(image=image)
        meta["image_path"] = str(img_path)
        cases_meta.append(meta)
        if case.reference_mask is not None:
            ref_path = work_dir / f"{case.case_id}_reference_mask.png"
            _save_png(ref_path, case.reference_mask)
            meta["reference_mask_path"] = str(ref_path)
    manifest = {"cases": cases_meta}
    manifest_path = work_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _load_manifest(work_dir: Path) -> dict[str, Any]:
    return json.loads((work_dir / "manifest.json").read_text(encoding="utf-8"))


def _load_case_image(work_dir: Path, case_id: str) -> np.ndarray:
    path = work_dir / f"{case_id}.png"
    return validate_image(np.asarray(Image.open(path).convert("RGB")))


def _mask_metrics(mask: np.ndarray, reference: np.ndarray | None) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = [mask_area_ratio(mask).to_dict()]
    if reference is not None:
        metrics.extend(
            [
                iou(mask, reference).to_dict(),
                f1(mask, reference).to_dict(),
            ]
        )
    return metrics


def benchmark_sam2(work_dir: Path) -> dict[str, Any]:
    from models.adapters.sam2_adapter import SAM2Adapter

    reset_registry()
    adapter = get_adapter("sam2")
    assert isinstance(adapter, SAM2Adapter)

    if not adapter.is_available():
        return {"status": "UNAVAILABLE", "reason": "SAM2 adapter not available in this process"}

    load_t0 = time.perf_counter()
    adapter.load()
    load_ms = (time.perf_counter() - load_t0) * 1000.0

    results: list[dict[str, Any]] = []
    for idx, (image, case) in enumerate(all_cases()):
        image = _load_case_image(work_dir, case.case_id)
        x, y = case.point_xy

        cold_t0 = time.perf_counter()
        cold = adapter.segment_point(image, x, y)
        cold_ms = (time.perf_counter() - cold_t0) * 1000.0

        warm_ms: list[float] = []
        warm_result = cold
        for _ in range(WARM_REPEATS):
            t0 = time.perf_counter()
            warm_result = adapter.segment_point(image, x, y)
            warm_ms.append((time.perf_counter() - t0) * 1000.0)

        mask_path = work_dir / f"{case.case_id}_sam2_point_mask.png"
        mask_hash = _save_png(mask_path, warm_result.mask)

        entry: dict[str, Any] = {
            "case_id": case.case_id,
            "device": "mps",
            "model": warm_result.model,
            "method": warm_result.method,
            "model_load_ms": round(load_ms, 3) if idx == 0 else None,
            "first_segmentation_ms": round(cold_ms, 3),
            "warm_segmentation_ms": {
                "best": round(min(warm_ms), 3),
                "mean": round(float(np.mean(warm_ms)), 3),
                "samples": WARM_REPEATS,
            },
            "confidence": warm_result.confidence,
            "mask_area_pixels": int(warm_result.mask.sum()),
            "mask_area_ratio": round(warm_result.mask.sum() / warm_result.mask.size, 6),
            "mask_shape": list(warm_result.mask.shape),
            "mask_sha256": mask_hash,
            "metrics": _mask_metrics(warm_result.mask, case.reference_mask),
            "reproducibility": build_reproducibility_record(
                config=ExperimentConfig(
                    experiment_id=f"sam2_point_{case.case_id}",
                    model="sam2",
                    backend="LOCAL_MPS",
                    operation="segment_point",
                    environment=os.environ.get("CONDA_DEFAULT_ENV"),
                    model_commit=SAM2_COMMIT,
                    seed=16001,
                    parameters={"point_xy": [x, y]},
                ),
                device="mps",
                image_hash=sha256_array(image)["image_sha256"],
                mask_hash=mask_hash,
            ),
        }
        results.append(entry)

    adapter.unload()
    return {
        "status": "MEASURED",
        "device": "mps",
        "model": "SAM 2.1 Hiera-Tiny",
        "model_load_ms": round(load_ms, 3),
        "peak_rss_mib": _peak_rss_mib(),
        "mps_driver_mib": _mps_driver_mib(),
        "cases": results,
    }


def benchmark_grounding(work_dir: Path) -> dict[str, Any]:
    reset_registry()
    adapter = get_adapter("grounding_dino")
    if not adapter.is_available():
        return {"status": "UNAVAILABLE", "reason": "Grounding DINO not available"}

    load_t0 = time.perf_counter()
    adapter.load()
    load_ms = (time.perf_counter() - load_t0) * 1000.0

    results: list[dict[str, Any]] = []
    for idx, (_image, case) in enumerate(all_cases()):
        image = _load_case_image(work_dir, case.case_id)
        t0 = time.perf_counter()
        grounding = adapter.infer(image, case.text_prompt)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        detections = [
            {
                "index": det_idx,
                "label": det.label,
                "confidence": round(float(det.confidence), 6),
                "box_xyxy": [det.x1, det.y1, det.x2, det.y2],
            }
            for det_idx, det in enumerate(grounding.detections)
        ]
        selected = detections[0] if detections else None
        results.append(
            {
                "case_id": case.case_id,
                "device": "cpu",
                "model": grounding.model,
                "prompt": case.text_prompt,
                "model_load_ms": round(load_ms, 3) if idx == 0 else None,
                "grounding_latency_ms": round(latency_ms, 3),
                "detection_count": len(detections),
                "detections": detections,
                "selected_box_xyxy": selected["box_xyxy"] if selected else None,
                "selected_confidence": selected["confidence"] if selected else None,
                "metrics": [
                    _metric("detection_count", float(len(detections)), MetricKind.MEASURED),
                ],
                "reproducibility": build_reproducibility_record(
                    config=ExperimentConfig(
                        experiment_id=f"grounding_{case.case_id}",
                        model="grounding_dino",
                        backend="CPU",
                        operation="ground",
                        environment=os.environ.get("CONDA_DEFAULT_ENV"),
                        model_commit=GSA_COMMIT,
                        parameters={"prompt": case.text_prompt},
                    ),
                    device="cpu",
                    image_hash=sha256_array(image)["image_sha256"],
                ),
            }
        )

    adapter.unload()
    return {
        "status": "MEASURED",
        "device": "cpu",
        "model": "Grounding DINO SwinT OGC",
        "model_load_ms": round(load_ms, 3),
        "peak_rss_mib": _peak_rss_mib(),
        "cases": results,
    }


def benchmark_sam2_box(work_dir: Path, grounding_results: dict[str, Any]) -> dict[str, Any]:
    from models.adapters.sam2_adapter import SAM2Adapter

    reset_registry()
    adapter = get_adapter("sam2")
    assert isinstance(adapter, SAM2Adapter)
    if not adapter.is_available():
        return {"status": "UNAVAILABLE", "reason": "SAM2 not available"}

    adapter.load()
    cases_out: list[dict[str, Any]] = []

    grounding_by_id = {
        entry["case_id"]: entry for entry in grounding_results.get("cases", [])
    }

    for idx, (_image, case) in enumerate(all_cases()):
        g = grounding_by_id.get(case.case_id)
        if not g or not g.get("selected_box_xyxy"):
            cases_out.append(
                {
                    "case_id": case.case_id,
                    "status": "SKIPPED",
                    "reason": "no grounding detection",
                }
            )
            continue

        image = _load_case_image(work_dir, case.case_id)
        x1, y1, x2, y2 = [int(v) for v in g["selected_box_xyxy"]]

        t0 = time.perf_counter()
        seg = adapter.segment_box(image, x1, y1, x2, y2)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        mask_path = work_dir / f"{case.case_id}_sam2_box_mask.png"
        mask_hash = _save_png(mask_path, seg.mask)

        cases_out.append(
            {
                "case_id": case.case_id,
                "status": "MEASURED",
                "box_xyxy": [x1, y1, x2, y2],
                "segmentation_latency_ms": round(latency_ms, 3),
                "confidence": seg.confidence,
                "mask_area_pixels": int(seg.mask.sum()),
                "mask_area_ratio": round(seg.mask.sum() / seg.mask.size, 6),
                "mask_shape": list(seg.mask.shape),
                "mask_sha256": mask_hash,
                "metrics": _mask_metrics(seg.mask, case.reference_mask),
                "reproducibility": build_reproducibility_record(
                    config=ExperimentConfig(
                        experiment_id=f"sam2_box_{case.case_id}",
                        model="sam2",
                        backend="LOCAL_MPS",
                        operation="segment_box",
                        environment=os.environ.get("CONDA_DEFAULT_ENV"),
                        model_commit=SAM2_COMMIT,
                        parameters={"box_xyxy": [x1, y1, x2, y2]},
                    ),
                    device="mps",
                    image_hash=sha256_array(image)["image_sha256"],
                    mask_hash=mask_hash,
                ),
            }
        )

    adapter.unload()
    return {
        "status": "MEASURED",
        "device": "mps",
        "cases": cases_out,
        "peak_rss_mib": _peak_rss_mib(),
        "mps_driver_mib": _mps_driver_mib(),
    }


def benchmark_moebius(work_dir: Path, mask_source: str = "sam2_point") -> dict[str, Any]:
    reset_registry()
    adapter = get_adapter("moebius")
    if not adapter.is_available():
        return {"status": "UNAVAILABLE", "reason": "Moebius not available"}

    load_t0 = time.perf_counter()
    adapter.load()
    load_ms = (time.perf_counter() - load_t0) * 1000.0

    params = InpaintParams(num_steps=20)
    cases_out: list[dict[str, Any]] = []

    for idx, (_image, case) in enumerate(all_cases()):
        image = _load_case_image(work_dir, case.case_id)
        mask_path = work_dir / f"{case.case_id}_{mask_source}_mask.png"
        if not mask_path.is_file():
            cases_out.append(
                {
                    "case_id": case.case_id,
                    "status": "SKIPPED",
                    "reason": f"missing mask at {mask_path.name}",
                }
            )
            continue

        mask = validate_mask(
            np.asarray(Image.open(mask_path).convert("L")) >= 128,
            image=image,
        )

        cold_t0 = time.perf_counter()
        cold = adapter.infer(image, mask, params)
        cold_ms = (time.perf_counter() - cold_t0) * 1000.0

        warm_ms: list[float] = []
        warm = cold
        for _ in range(WARM_REPEATS):
            t0 = time.perf_counter()
            warm = adapter.infer(image, mask, params)
            warm_ms.append((time.perf_counter() - t0) * 1000.0)

        out_path = work_dir / f"{case.case_id}_moebius_output.png"
        output_hash = _save_png(out_path, warm.result)

        cases_out.append(
            {
                "case_id": case.case_id,
                "status": "MEASURED",
                "mask_source": mask_source,
                "mask_sha256": sha256_file(mask_path),
                "model_load_ms": round(load_ms, 3) if idx == 0 else None,
                "first_inference_ms": round(cold_ms, 3),
                "warm_inference_ms": {
                    "best": round(min(warm_ms), 3),
                    "mean": round(float(np.mean(warm_ms)), 3),
                    "samples": WARM_REPEATS,
                },
                "output_resolution": list(warm.result.shape),
                "adapter_latency_ms": warm.latency_ms,
                "memory_mb": warm.memory_mb,
                "output_sha256": output_hash,
                "metrics": [
                    mask_area_ratio(mask).to_dict(),
                    _metric(
                        "outside_mask_preservation",
                        None,
                        MetricKind.UNAVAILABLE,
                        note="not computed in benchmark harness",
                    ),
                ],
                "reproducibility": build_reproducibility_record(
                    config=ExperimentConfig(
                        experiment_id=f"moebius_{case.case_id}",
                        model="moebius",
                        backend="LOCAL_MPS",
                        operation="inpaint",
                        environment=os.environ.get("CONDA_DEFAULT_ENV"),
                        model_commit=MOEBIUS_COMMIT,
                        parameters={"num_steps": 20, "mask_source": mask_source},
                    ),
                    device="mps",
                    image_hash=sha256_array(image)["image_sha256"],
                    mask_hash=sha256_file(mask_path),
                    output_hash=output_hash,
                ),
            }
        )

    adapter.unload()
    return {
        "status": "MEASURED",
        "device": "mps",
        "model": "Moebius",
        "model_load_ms": round(load_ms, 3),
        "peak_rss_mib": _peak_rss_mib(),
        "mps_driver_mib": _mps_driver_mib(),
        "cases": cases_out,
    }


def benchmark_e2e(work_dir: Path) -> dict[str, Any]:
    """End-to-end timings using subprocess bridges where envs differ."""
    from apps.backend.isolated_runner import ground_via_isolated_env, inpaint_via_isolated_env
    from models.adapters.sam2_adapter import SAM2Adapter

    text_case = all_cases()[0][1]  # case1 — deterministic text path
    click_case = all_cases()[1][1]  # case2 — click path

    results: dict[str, Any] = {}

    # Text → Grounding DINO → SAM2 box → Moebius
    image_text = _load_case_image(work_dir, text_case.case_id)
    wall_t0 = time.perf_counter()
    stages: list[dict[str, Any]] = []

    g_t0 = time.perf_counter()
    grounding = ground_via_isolated_env(image_text, text_case.text_prompt)
    g_ms = (time.perf_counter() - g_t0) * 1000.0
    stages.append({"stage": "grounding_dino", "latency_ms": round(g_ms, 3), "device": "cpu"})

    if not grounding.detections:
        results["text_to_moebius"] = {
            "status": "FAILED",
            "reason": "no detections from grounding",
            "stages": stages,
        }
    else:
        det = grounding.detections[0]
        reset_registry()
        sam = get_adapter("sam2")
        assert isinstance(sam, SAM2Adapter)
        sam.load()
        s_t0 = time.perf_counter()
        seg = sam.segment_box(image_text, det.x1, det.y1, det.x2, det.y2)
        s_ms = (time.perf_counter() - s_t0) * 1000.0
        stages.append({"stage": "sam2_box", "latency_ms": round(s_ms, 3), "device": "mps"})
        sam.unload()

        mask_path = work_dir / "e2e_text_mask.png"
        _save_png(mask_path, seg.mask)

        m_t0 = time.perf_counter()
        inpaint = inpaint_via_isolated_env(image_text, seg.mask)
        m_ms = (time.perf_counter() - m_t0) * 1000.0
        stages.append({"stage": "moebius", "latency_ms": round(m_ms, 3), "device": "mps"})

        out_path = work_dir / "e2e_text_output.png"
        output_hash = _save_png(out_path, inpaint.result)
        total_ms = (time.perf_counter() - wall_t0) * 1000.0

        results["text_to_moebius"] = {
            "status": "MEASURED",
            "case_id": text_case.case_id,
            "prompt": text_case.text_prompt,
            "total_latency_ms": round(total_ms, 3),
            "stages": stages,
            "output_resolution": list(inpaint.result.shape),
            "output_sha256": output_hash,
            "mask_sha256": sha256_file(mask_path),
            "image_sha256": sha256_array(image_text)["image_sha256"],
        }

    # Click → SAM2 point → Moebius
    image_click = _load_case_image(work_dir, click_case.case_id)
    wall_t0 = time.perf_counter()
    stages = []

    reset_registry()
    sam = get_adapter("sam2")
    assert isinstance(sam, SAM2Adapter)
    sam.load()
    cx, cy = click_case.point_xy
    s_t0 = time.perf_counter()
    seg = sam.segment_point(image_click, cx, cy)
    s_ms = (time.perf_counter() - s_t0) * 1000.0
    stages.append({"stage": "sam2_point", "latency_ms": round(s_ms, 3), "device": "mps"})
    sam.unload()

    mask_path = work_dir / "e2e_click_mask.png"
    _save_png(mask_path, seg.mask)

    m_t0 = time.perf_counter()
    inpaint = inpaint_via_isolated_env(image_click, seg.mask)
    m_ms = (time.perf_counter() - m_t0) * 1000.0
    stages.append({"stage": "moebius", "latency_ms": round(m_ms, 3), "device": "mps"})
    total_ms = (time.perf_counter() - wall_t0) * 1000.0

    out_path = work_dir / "e2e_click_output.png"
    output_hash = _save_png(out_path, inpaint.result)

    results["click_to_moebius"] = {
        "status": "MEASURED",
        "case_id": click_case.case_id,
        "point_xy": list(click_case.point_xy),
        "total_latency_ms": round(total_ms, 3),
        "stages": stages,
        "output_resolution": list(inpaint.result.shape),
        "output_sha256": output_hash,
        "mask_sha256": sha256_file(mask_path),
        "image_sha256": sha256_array(image_click)["image_sha256"],
    }

    return {
        "status": "MEASURED",
        **results,
    }


def _run_phase(phase: str, work_dir: Path) -> dict[str, Any]:
    if phase == "sam2":
        return benchmark_sam2(work_dir)
    if phase == "grounding":
        return benchmark_grounding(work_dir)
    if phase == "sam2_box":
        grounding = json.loads((work_dir / "grounding.json").read_text(encoding="utf-8"))
        return benchmark_sam2_box(work_dir, grounding)
    if phase == "moebius":
        return benchmark_moebius(work_dir)
    if phase == "e2e":
        return benchmark_e2e(work_dir)
    raise ValueError(f"unknown phase: {phase}")


def orchestrate(work_dir: Path) -> BenchmarkReport:
    report = BenchmarkReport()
    report.timestamp_utc = datetime.now(timezone.utc).isoformat()
    report.pixelforge_commit = _git_head()
    report.hardware = _collect_hardware()
    report.model_commits = {
        "sam2": SAM2_COMMIT,
        "moebius": MOEBIUS_COMMIT,
        "grounding_dino": GSA_COMMIT,
    }
    report.methodology = {
        "image_source": "synthetic closed-form 512x512 scenes (no external dataset)",
        "warm_repeats": WARM_REPEATS,
        "measurement_only": True,
        "ground_truth_policy": (
            "IoU/F1 only for case1 where exact disc reference exists; "
            "otherwise mask_area_ratio and latency only"
        ),
    }

    manifest = _prepare_workdir(work_dir)
    report.test_cases = manifest["cases"]

    phases = [
        ("sam2", "pixelforge-sam2-v2"),
        ("grounding", "pixelforge-grounding-dino"),
        ("sam2_box", "pixelforge-sam2-v2"),
        ("moebius", "pixelforge-moebius"),
        ("e2e", "pixelforge-sam2-v2"),
    ]

    phase_results: dict[str, Any] = {}
    for phase, env_name in phases:
        python = _env_python(env_name)
        report.environments[phase] = {
            "conda_env": env_name,
            "python": str(python),
            "available": python.is_file(),
        }
        if not python.is_file():
            phase_results[phase] = {
                "status": "UNAVAILABLE",
                "reason": f"python not found: {python}",
            }
            continue

        out_file = work_dir / f"{phase}.json"
        cmd = [
            str(python),
            "-m",
            "evaluation.benchmarks.run_mvp_benchmark",
            "--phase",
            phase,
            "--work-dir",
            str(work_dir),
            "--output",
            str(out_file),
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT)
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=env,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            phase_results[phase] = {
                "status": "FAILED",
                "returncode": proc.returncode,
                "stderr": (proc.stderr or "")[-2000:],
                "stdout": (proc.stdout or "")[-2000:],
            }
        elif out_file.is_file():
            phase_results[phase] = json.loads(out_file.read_text(encoding="utf-8"))
        else:
            phase_results[phase] = {"status": "FAILED", "reason": "no output file"}

        if phase == "grounding" and out_file.is_file():
            (work_dir / "grounding.json").write_text(out_file.read_text(encoding="utf-8"))

    report.sam2 = phase_results.get("sam2", {})
    report.grounding_dino = phase_results.get("grounding", {})
    report.sam2_box = phase_results.get("sam2_box", {})
    report.moebius = phase_results.get("moebius", {})
    report.end_to_end = phase_results.get("e2e", {})

    statuses = [
        report.sam2.get("status"),
        report.grounding_dino.get("status"),
        report.moebius.get("status"),
    ]
    if all(s == "MEASURED" for s in statuses):
        report.status = "COMPLETE"
    elif any(s == "MEASURED" for s in statuses):
        report.status = "PARTIAL"
    else:
        report.status = "UNAVAILABLE"

    report.limitations = [
        "Measurement only — no optimization applied.",
        "Cross-env orchestration adds subprocess overhead to end-to-end timings.",
        "Grounding DINO quality metrics omitted (no ground-truth boxes).",
        "Moebius marked CONDITIONAL in prior validation; benchmark records actual behavior.",
        "Generated PNG artifacts are not committed (outputs/mvp_benchmark/).",
    ]
    return report


def write_markdown(report: BenchmarkReport, path: Path) -> None:
    data = report.to_dict()

    def fmt_ms(val: Any) -> str:
        return f"{val:.1f}" if isinstance(val, (int, float)) else str(val)

    lines = [
        "# MVP Local Pipeline Benchmark (Phase 16)",
        "",
        f"**Status:** {report.status}  ",
        f"**Timestamp (UTC):** {report.timestamp_utc}  ",
        f"**PixelForge commit:** `{report.pixelforge_commit}`  ",
        "",
        "> Measurement-only benchmark. Does not claim production readiness.",
        "",
        "## Methodology",
        "",
        "- Three synthetic 512×512 scenes generated from closed-form math (no external dataset, no RNG).",
        "- Real inference in isolated conda environments (`pixelforge-sam2-v2`, `pixelforge-grounding-dino`, `pixelforge-moebius`).",
        "- Three warm repetitions for steady-state latency (SAM2 point, Moebius inpaint).",
        "- Metrics: `mask_area_ratio` (COMPUTED), IoU/F1 only where exact reference mask exists (case 1).",
        "",
        "## Hardware",
        "",
        f"- Platform: {report.hardware.get('platform')}",
        f"- Processor: {report.hardware.get('processor')}",
        f"- Cores: {report.hardware.get('physical_cores')} physical / {report.hardware.get('logical_cores')} logical",
        "",
        "## Model commits",
        "",
    ]
    for name, sha in report.model_commits.items():
        lines.append(f"- **{name}:** `{sha}`")
    lines.extend(["", "## Test cases", ""])
    for case in report.test_cases:
        lines.append(
            f"### {case['case_id']}\n"
            f"- {case['description']}\n"
            f"- Point: `{case['point_xy']}` | Text: \"{case['text_prompt']}\"\n"
            f"- Image SHA-256: `{case['image_sha256']}`\n"
            f"- Reference mask: {'yes' if case['has_reference_mask'] else 'no'}\n"
        )

    lines.extend(["", "## Benchmark 1 — SAM2 point segmentation", ""])
    sam2 = data.get("sam2", {})
    if sam2.get("status") == "MEASURED":
        lines.append(f"- Model load: **{fmt_ms(sam2.get('model_load_ms'))} ms**")
        lines.append(f"- Peak RSS: {sam2.get('peak_rss_mib')} MiB | MPS driver: {sam2.get('mps_driver_mib')} MiB")
        for c in sam2.get("cases", []):
            lines.append(
                f"- **{c['case_id']}**: cold {fmt_ms(c['first_segmentation_ms'])} ms, "
                f"warm best {fmt_ms(c['warm_segmentation_ms']['best'])} ms, "
                f"area {c['mask_area_ratio']:.4f}, confidence {c['confidence']:.4f}"
            )
    else:
        lines.append(f"- Status: {sam2.get('status')} — {sam2.get('reason', sam2.get('stderr', ''))[:200]}")

    lines.extend(["", "## Benchmark 2 — Grounding DINO", ""])
    gd = data.get("grounding_dino", {})
    if gd.get("status") == "MEASURED":
        lines.append(f"- Model load: **{fmt_ms(gd.get('model_load_ms'))} ms**")
        for c in gd.get("cases", []):
            lines.append(
                f"- **{c['case_id']}**: {fmt_ms(c['grounding_latency_ms'])} ms, "
                f"{c['detection_count']} detection(s), "
                f"selected conf {c.get('selected_confidence')}"
            )
    else:
        lines.append(f"- Status: {gd.get('status')}")

    lines.extend(["", "## Benchmark 3 — SAM2 box (after grounding)", ""])
    sb = data.get("sam2_box", {})
    if sb.get("status") == "MEASURED":
        for c in sb.get("cases", []):
            if c.get("status") != "MEASURED":
                lines.append(f"- **{c['case_id']}**: skipped — {c.get('reason')}")
                continue
            lines.append(
                f"- **{c['case_id']}**: {fmt_ms(c['segmentation_latency_ms'])} ms, "
                f"area {c['mask_area_ratio']:.4f}, box `{c['box_xyxy']}`"
            )
    else:
        lines.append(f"- Status: {sb.get('status')}")

    lines.extend(["", "## Benchmark 4 — Moebius inpainting", ""])
    mob = data.get("moebius", {})
    if mob.get("status") == "MEASURED":
        lines.append(f"- Model load: **{fmt_ms(mob.get('model_load_ms'))} ms**")
        for c in mob.get("cases", []):
            if c.get("status") != "MEASURED":
                continue
            lines.append(
                f"- **{c['case_id']}**: cold {fmt_ms(c['first_inference_ms'])} ms, "
                f"warm best {fmt_ms(c['warm_inference_ms']['best'])} ms, "
                f"output {c['output_resolution']}"
            )
    else:
        lines.append(f"- Status: {mob.get('status')}")

    lines.extend(["", "## Benchmark 5 — End-to-end", ""])
    e2e = data.get("end_to_end", {})
    for key in ("text_to_moebius", "click_to_moebius"):
        entry = e2e.get(key, {})
        lines.append(f"### {key}")
        if entry.get("status") == "MEASURED":
            lines.append(f"- Total: **{fmt_ms(entry['total_latency_ms'])} ms**")
            for stage in entry.get("stages", []):
                lines.append(f"  - {stage['stage']}: {fmt_ms(stage['latency_ms'])} ms ({stage['device']})")
        else:
            lines.append(f"- Status: {entry.get('status', 'missing')}")

    lines.extend(["", "## Limitations", ""])
    for item in report.limitations:
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## Machine-readable report",
            "",
            f"See [`evaluation/reports/mvp_benchmark.json`](../../evaluation/reports/mvp_benchmark.json).",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orchestrate", action="store_true", help="Run all phases in conda envs")
    parser.add_argument("--phase", choices=["sam2", "grounding", "sam2_box", "moebius", "e2e"])
    parser.add_argument("--work-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--output", type=Path, help="Write single-phase JSON here")
    parser.add_argument("--write-doc", action="store_true", help="Also write MVP_BENCHMARK.md")
    args = parser.parse_args()

    if args.orchestrate:
        report = orchestrate(args.work_dir)
        REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
        REPORT_JSON.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        write_markdown(report, REPORT_MD)
        print(f"Wrote {REPORT_JSON}")
        print(f"Wrote {REPORT_MD}")
        print(f"Status: {report.status}")
        return 0 if report.status in {"COMPLETE", "PARTIAL"} else 1

    if not args.phase:
        parser.error("specify --phase or --orchestrate")

    if not (args.work_dir / "manifest.json").is_file():
        _prepare_workdir(args.work_dir)

    result = _run_phase(args.phase, args.work_dir)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    else:
        print(json.dumps(result, indent=2))
    ok_statuses = {"MEASURED", "COMPLETE", "PARTIAL"}
    if result.get("status") in ok_statuses:
        return 0
    if args.phase == "e2e" and result.get("text_to_moebius", {}).get("status") == "MEASURED":
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
