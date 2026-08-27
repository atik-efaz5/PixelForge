#!/usr/bin/env python3
"""Phase 20 — real-image quality evaluation harness.

Generates deterministic local test scenes (~512×512), runs real SAM2 /
Grounding DINO / Moebius inference, and writes:

  - evaluation/reports/real_image_quality.json
  - docs/experiments/REAL_IMAGE_QUALITY_EVALUATION.md

PNG artifacts go to outputs/real_image_eval/ (gitignored).

Usage::

    python -m evaluation.benchmarks.run_real_image_eval --orchestrate
    python -m evaluation.benchmarks.run_real_image_eval --phase sam2 --work-dir outputs/real_image_eval
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from evaluation.benchmarks.real_image_cases import RealImageCase, all_cases, inpaint_case_ids
from evaluation.benchmarks.run_mvp_benchmark import (
    GSA_COMMIT,
    MOEBIUS_COMMIT,
    SAM2_COMMIT,
    _collect_hardware,
    _env_python,
    _git_head,
    _metric,
    _mps_driver_mib,
    _peak_rss_mib,
    _save_png,
)
from evaluation.metrics import (
    boundary_difference_ratio,
    boundary_overlap,
    f1,
    iou,
    mask_area_ratio,
    outside_mask_preservation_score,
    precision,
    recall,
)
from evaluation.reproducibility import build_reproducibility_record, sha256_array, sha256_file
from evaluation.types import ExperimentConfig, MetricKind
from models.registry import get_adapter, reset_registry
from models.types import InpaintParams, validate_image, validate_mask

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "real_image_eval"
REPORT_JSON = PROJECT_ROOT / "evaluation" / "reports" / "real_image_quality.json"
REPORT_MD = PROJECT_ROOT / "docs" / "experiments" / "REAL_IMAGE_QUALITY_EVALUATION.md"

WARM_REPEATS = 1  # limited warm repeats per Phase 20 scope


@dataclass
class RealImageReport:
    evaluation_id: str = "real_image_quality_v1"
    phase: str = "20"
    status: str = "NOT_RUN"
    timestamp_utc: str = ""
    pixelforge_commit: str | None = None
    hardware: dict[str, Any] = field(default_factory=dict)
    environments: dict[str, Any] = field(default_factory=dict)
    model_commits: dict[str, str] = field(default_factory=dict)
    methodology: dict[str, Any] = field(default_factory=dict)
    test_cases: list[dict[str, Any]] = field(default_factory=list)
    selection: dict[str, Any] = field(default_factory=dict)
    inpainting: dict[str, Any] = field(default_factory=dict)
    preservation: dict[str, Any] = field(default_factory=dict)
    end_to_end: dict[str, Any] = field(default_factory=dict)
    qualitative_observations: list[dict[str, Any]] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    unavailable_metrics: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evaluation_id": self.evaluation_id,
            "phase": self.phase,
            "status": self.status,
            "timestamp_utc": self.timestamp_utc,
            "pixelforge_commit": self.pixelforge_commit,
            "hardware": self.hardware,
            "environments": self.environments,
            "model_commits": self.model_commits,
            "methodology": self.methodology,
            "test_cases": self.test_cases,
            "selection": self.selection,
            "inpainting": self.inpainting,
            "preservation": self.preservation,
            "end_to_end": self.end_to_end,
            "qualitative_observations": self.qualitative_observations,
            "limitations": self.limitations,
            "unavailable_metrics": self.unavailable_metrics,
        }


def _case_by_id(case_id: str) -> RealImageCase:
    for _image, case in all_cases():
        if case.case_id == case_id:
            return case
    raise KeyError(case_id)


def _reference_mask(case: RealImageCase) -> np.ndarray | None:
    return case.reference_mask


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
    (work_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _load_case_image(work_dir: Path, case_id: str) -> np.ndarray:
    return validate_image(np.asarray(Image.open(work_dir / f"{case_id}.png").convert("RGB")))


def _mask_metrics(mask: np.ndarray, reference: np.ndarray | None) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = [mask_area_ratio(mask).to_dict()]
    if reference is not None:
        metrics.extend(
            [
                iou(mask, reference).to_dict(),
                precision(mask, reference).to_dict(),
                recall(mask, reference).to_dict(),
                f1(mask, reference).to_dict(),
                boundary_overlap(mask, reference).to_dict(),
                boundary_difference_ratio(mask, reference).to_dict(),
            ]
        )
    return metrics


def _qualitative_notes(
    case: RealImageCase,
    *,
    mask: np.ndarray,
    metrics: list[dict[str, Any]],
    workflow: str,
) -> dict[str, Any]:
    iou_val = next((m["value"] for m in metrics if m.get("name") == "iou"), None)
    notes: dict[str, Any] = {
        "case_id": case.case_id,
        "category": case.category,
        "workflow": workflow,
        "target_object": case.target_object,
        "mask_area_ratio": round(mask.sum() / mask.size, 6),
        "reference_iou": iou_val,
        "obvious_artifacts": (
            "Not assessed visually in harness — see saved PNG artifacts."
        ),
        "boundary_quality": (
            f"Reference IoU {iou_val:.3f} (pixel-level, constructed GT)"
            if isinstance(iou_val, (int, float))
            else "No constructed reference — boundary quality not scored"
        ),
        "structure_preservation": "Subjective — not scored numerically",
        "semantic_plausibility": "Subjective — not scored numerically",
    }
    return notes


def benchmark_sam2_point(work_dir: Path) -> dict[str, Any]:
    from models.adapters.sam2_adapter import SAM2Adapter

    reset_registry()
    adapter = get_adapter("sam2")
    assert isinstance(adapter, SAM2Adapter)
    if not adapter.is_available():
        return {"status": "UNAVAILABLE", "reason": "SAM2 adapter not available"}

    load_t0 = time.perf_counter()
    adapter.load()
    load_ms = (time.perf_counter() - load_t0) * 1000.0

    qualitative: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for idx, (_image, case) in enumerate(all_cases()):
        image = _load_case_image(work_dir, case.case_id)
        x, y = case.point_xy

        cold_t0 = time.perf_counter()
        cold = adapter.segment_point(image, x, y)
        cold_ms = (time.perf_counter() - cold_t0) * 1000.0

        warm_ms: list[float] = []
        warm = cold
        for _ in range(WARM_REPEATS):
            t0 = time.perf_counter()
            warm = adapter.segment_point(image, x, y)
            warm_ms.append((time.perf_counter() - t0) * 1000.0)

        mask_path = work_dir / f"{case.case_id}_sam2_point_mask.png"
        mask_hash = _save_png(mask_path, warm.mask)
        ref = _reference_mask(case)
        metrics = _mask_metrics(warm.mask, ref)
        qualitative.append(
            _qualitative_notes(case, mask=warm.mask, metrics=metrics, workflow="click_sam2")
        )

        results.append(
            {
                "case_id": case.case_id,
                "workflow": "click_sam2",
                "device": "mps",
                "model": warm.model,
                "point_xy": [x, y],
                "model_load_ms": round(load_ms, 3) if idx == 0 else None,
                "cold_segmentation_ms": round(cold_ms, 3),
                "warm_segmentation_ms": {
                    "best": round(min(warm_ms), 3) if warm_ms else None,
                    "mean": round(float(np.mean(warm_ms)), 3) if warm_ms else None,
                    "samples": WARM_REPEATS,
                },
                "confidence": warm.confidence,
                "mask_area_pixels": int(warm.mask.sum()),
                "mask_area_ratio": round(warm.mask.sum() / warm.mask.size, 6),
                "mask_sha256": mask_hash,
                "metrics": metrics,
                "reproducibility": build_reproducibility_record(
                    config=ExperimentConfig(
                        experiment_id=f"real_sam2_point_{case.case_id}",
                        model="sam2",
                        backend="LOCAL_MPS",
                        operation="segment_point",
                        environment=os.environ.get("CONDA_DEFAULT_ENV"),
                        model_commit=SAM2_COMMIT,
                        seed=20001,
                        parameters={"point_xy": [x, y]},
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
        "workflow": "click_sam2",
        "device": "mps",
        "model_load_ms": round(load_ms, 3),
        "peak_rss_mib": _peak_rss_mib(),
        "mps_driver_mib": _mps_driver_mib(),
        "cases": results,
        "qualitative": qualitative,
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
                "workflow": "text_grounding",
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
                        experiment_id=f"real_grounding_{case.case_id}",
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
        "workflow": "text_grounding",
        "device": "cpu",
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
    grounding_by_id = {e["case_id"]: e for e in grounding_results.get("cases", [])}
    qualitative: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []

    for _idx, (_image, case) in enumerate(all_cases()):
        g = grounding_by_id.get(case.case_id)
        if not g or not g.get("selected_box_xyxy"):
            results.append(
                {
                    "case_id": case.case_id,
                    "workflow": "text_grounding_sam2_box",
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
        ref = _reference_mask(case)
        metrics = _mask_metrics(seg.mask, ref)
        qualitative.append(
            _qualitative_notes(
                case, mask=seg.mask, metrics=metrics, workflow="text_grounding_sam2_box"
            )
        )

        results.append(
            {
                "case_id": case.case_id,
                "workflow": "text_grounding_sam2_box",
                "status": "MEASURED",
                "box_xyxy": [x1, y1, x2, y2],
                "grounding_confidence": g.get("selected_confidence"),
                "segmentation_latency_ms": round(latency_ms, 3),
                "confidence": seg.confidence,
                "mask_area_pixels": int(seg.mask.sum()),
                "mask_area_ratio": round(seg.mask.sum() / seg.mask.size, 6),
                "mask_sha256": mask_hash,
                "metrics": metrics,
                "reproducibility": build_reproducibility_record(
                    config=ExperimentConfig(
                        experiment_id=f"real_sam2_box_{case.case_id}",
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
        "workflow": "text_grounding_sam2_box",
        "device": "mps",
        "cases": results,
        "qualitative": qualitative,
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
    allowed = inpaint_case_ids()
    cases_out: list[dict[str, Any]] = []
    preservation_out: list[dict[str, Any]] = []
    qualitative: list[dict[str, Any]] = []

    for idx, (_image, case) in enumerate(all_cases()):
        if case.case_id not in allowed:
            continue
        image = _load_case_image(work_dir, case.case_id)
        mask_path = work_dir / f"{case.case_id}_{mask_source}_mask.png"
        if not mask_path.is_file():
            cases_out.append(
                {
                    "case_id": case.case_id,
                    "status": "SKIPPED",
                    "reason": f"missing mask {mask_path.name}",
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
        preservation = outside_mask_preservation_score(image, warm.result, mask).to_dict()
        preservation_out.append(
            {
                "case_id": case.case_id,
                "metric": preservation,
                "note": "pixel-level preservation metric; not a human-quality score",
            }
        )

        qualitative.append(
            {
                "case_id": case.case_id,
                "workflow": "moebius_inpaint",
                "input_sha256": sha256_array(image)["image_sha256"],
                "mask_sha256": sha256_file(mask_path),
                "result_sha256": output_hash,
                "obvious_artifacts": "Inspect output PNG in outputs/real_image_eval/",
                "boundary_quality": "Subjective — not scored numerically",
                "structure_preservation": (
                    f"outside_mask_preservation={preservation.get('value')}"
                    if preservation.get("value") is not None
                    else "preservation unavailable"
                ),
                "semantic_plausibility": "Subjective — not scored numerically",
            }
        )

        cases_out.append(
            {
                "case_id": case.case_id,
                "status": "MEASURED",
                "mask_source": mask_source,
                "model_load_ms": round(load_ms, 3) if idx == 0 else None,
                "cold_inference_ms": round(cold_ms, 3),
                "warm_inference_ms": {
                    "best": round(min(warm_ms), 3) if warm_ms else None,
                    "mean": round(float(np.mean(warm_ms)), 3) if warm_ms else None,
                    "samples": WARM_REPEATS,
                },
                "output_resolution": list(warm.result.shape),
                "adapter_latency_ms": warm.latency_ms,
                "memory_mb": warm.memory_mb,
                "output_sha256": output_hash,
                "metrics": [preservation],
                "reproducibility": build_reproducibility_record(
                    config=ExperimentConfig(
                        experiment_id=f"real_moebius_{case.case_id}",
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
        "preservation": preservation_out,
        "qualitative": qualitative,
    }


def benchmark_e2e(work_dir: Path) -> dict[str, Any]:
    from apps.backend.isolated_runner import ground_via_isolated_env, inpaint_via_isolated_env
    from models.adapters.sam2_adapter import SAM2Adapter

    text_case = _case_by_id("real03_furniture")
    click_case = _case_by_id("real06_adjacent")
    results: dict[str, Any] = {}

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
            "case_id": text_case.case_id,
            "reason": "no detections",
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
        total_ms = (time.perf_counter() - wall_t0) * 1000.0
        out_path = work_dir / "e2e_text_output.png"
        output_hash = _save_png(out_path, inpaint.result)

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

    return {"status": "MEASURED", **results}


def _run_phase(phase: str, work_dir: Path) -> dict[str, Any]:
    if phase == "sam2":
        return benchmark_sam2_point(work_dir)
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


def orchestrate(work_dir: Path) -> RealImageReport:
    report = RealImageReport()
    report.timestamp_utc = datetime.now(timezone.utc).isoformat()
    report.pixelforge_commit = _git_head()
    report.hardware = _collect_hardware()
    report.model_commits = {
        "sam2": SAM2_COMMIT,
        "moebius": MOEBIUS_COMMIT,
        "grounding_dino": GSA_COMMIT,
    }
    report.methodology = {
        "image_source": (
            "deterministic locally generated 512×512 scenes (no external dataset downloads)"
        ),
        "categories": [
            "person",
            "vehicle",
            "furniture",
            "outdoor_scene",
            "textured_background",
            "adjacent_objects",
            "cluttered_scene",
        ],
        "warm_repeats": WARM_REPEATS,
        "inpaint_cases": sorted(inpaint_case_ids()),
        "ground_truth_policy": (
            "Constructed analytic reference masks for all cases; "
            "IoU/F1/boundary metrics are valid against these synthetic GT masks only"
        ),
        "preservation_note": (
            "outside_mask_preservation is a pixel-level metric, not human quality"
        ),
    }
    report.unavailable_metrics = [
        "lpips",
        "ssim_vs_reference_edit",
        "human_preference_score",
        "grounding_box_iou (no constructed box GT)",
    ]

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
            "evaluation.benchmarks.run_real_image_eval",
            "--phase",
            phase,
            "--work-dir",
            str(work_dir),
            "--output",
            str(out_file),
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT)
        proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env, capture_output=True, text=True)
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

    sam2 = phase_results.get("sam2", {})
    grounding = phase_results.get("grounding", {})
    sam2_box = phase_results.get("sam2_box", {})
    moebius = phase_results.get("moebius", {})

    report.selection = {
        "click_sam2": sam2,
        "text_grounding": grounding,
        "text_grounding_sam2_box": sam2_box,
    }
    report.inpainting = moebius
    report.preservation = {"cases": moebius.get("preservation", [])}
    report.end_to_end = phase_results.get("e2e", {})

    qualitative: list[dict[str, Any]] = []
    for block in (sam2, sam2_box, moebius):
        qualitative.extend(block.get("qualitative", []))
    report.qualitative_observations = qualitative

    statuses = [sam2.get("status"), grounding.get("status"), moebius.get("status")]
    if all(s == "MEASURED" for s in statuses):
        report.status = "COMPLETE"
    elif any(s == "MEASURED" for s in statuses):
        report.status = "PARTIAL"
    else:
        report.status = "UNAVAILABLE"

    report.limitations = [
        "Scenes are deterministic synthetic images, not photographs from a public dataset.",
        "Reference masks are constructed analytically — metrics measure agreement with designed targets.",
        "Subjective quality (semantic plausibility, visible artifacts) is recorded but not numerically scored.",
        "Moebius inpainting runs on a subset (3 cases) to limit inference cost.",
        "Cross-env subprocess orchestration adds overhead to end-to-end timings.",
        "Does not supersede or overwrite Phase 16 mvp_benchmark.json.",
        "Generated PNG artifacts are not committed (outputs/real_image_eval/).",
    ]
    return report


def write_markdown(report: RealImageReport, path: Path) -> None:
    data = report.to_dict()
    lines = [
        "# Real-Image Quality Evaluation (Phase 20)",
        "",
        f"**Status:** {report.status}  ",
        f"**Timestamp (UTC):** {report.timestamp_utc}  ",
        f"**PixelForge commit:** `{report.pixelforge_commit}`  ",
        "",
        "> Evaluation on deterministic local scenes. Does not claim photographic realism.",
        "",
        "## Hardware & environment",
        "",
        f"- Platform: {data['hardware'].get('platform')}",
        f"- Processor: {data['hardware'].get('processor')}",
        f"- Unified memory: {data['hardware'].get('unified_memory_bytes')} bytes",
        "",
        "## Test set",
        "",
        f"{len(data['test_cases'])} cases at 512×512 — categories: "
        + ", ".join(data["methodology"]["categories"]),
        "",
        "| Case | Category | SHA-256 (image) | Reference GT |",
        "|------|----------|-----------------|--------------|",
    ]
    for case in data["test_cases"]:
        lines.append(
            f"| `{case['case_id']}` | {case['category']} | `{case['image_sha256'][:16]}…` | "
            f"{'yes' if case['has_reference_mask'] else 'no'} |"
        )

    lines.extend(["", "## Selection results", ""])
    for workflow, block in data["selection"].items():
        lines.append(f"### {workflow} — {block.get('status', 'N/A')}")
        for entry in block.get("cases", [])[:3]:
            iou_val = next(
                (m.get("value") for m in entry.get("metrics", []) if m.get("name") == "iou"),
                None,
            )
            lines.append(
                f"- **{entry.get('case_id')}**: latency recorded; "
                f"IoU={iou_val if iou_val is not None else 'n/a'}; "
                f"mask_area_ratio={entry.get('mask_area_ratio', 'n/a')}"
            )
        if len(block.get("cases", [])) > 3:
            lines.append(f"- … and {len(block['cases']) - 3} more (see JSON)")
        lines.append("")

    lines.extend(["## Inpainting (Moebius)", ""])
    inp = data.get("inpainting", {})
    lines.append(f"Status: **{inp.get('status', 'N/A')}**")
    for entry in inp.get("cases", []):
        lines.append(
            f"- `{entry.get('case_id')}`: cold={entry.get('cold_inference_ms')} ms, "
            f"warm best={entry.get('warm_inference_ms', {}).get('best')} ms, "
            f"output `{str(entry.get('output_sha256', ''))[:16]}…`"
        )

    lines.extend(["", "## Preservation", ""])
    for entry in data.get("preservation", {}).get("cases", []):
        metric = entry.get("metric", {})
        lines.append(
            f"- `{entry.get('case_id')}`: outside_mask_preservation="
            f"{metric.get('value')} (pixel-level, not perceptual)"
        )

    lines.extend(["", "## End-to-end workflows", ""])
    e2e = data.get("end_to_end", {})
    for key in ("text_to_moebius", "click_to_moebius"):
        block = e2e.get(key, {})
        if block:
            lines.append(
                f"- **{key}** (`{block.get('case_id')}`): "
                f"total {block.get('total_latency_ms')} ms — stages: {block.get('stages')}"
            )

    lines.extend(["", "## Qualitative observations", ""])
    lines.append(
        "Subjective dimensions are listed in JSON without numeric scores. "
        "Inspect PNG artifacts under `outputs/real_image_eval/`."
    )

    lines.extend(["", "## Limitations", ""])
    for item in data.get("limitations", []):
        lines.append(f"- {item}")

    lines.extend(
        [
            "",
            "## Machine-readable report",
            "",
            "See [`evaluation/reports/real_image_quality.json`](../../evaluation/reports/real_image_quality.json).",
            "",
            "Phase 16 baseline: [`evaluation/reports/mvp_benchmark.json`](../../evaluation/reports/mvp_benchmark.json) (unchanged).",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orchestrate", action="store_true")
    parser.add_argument("--phase", choices=["sam2", "grounding", "sam2_box", "moebius", "e2e"])
    parser.add_argument("--work-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--write-doc", action="store_true")
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

    ok = result.get("status") in {"MEASURED", "COMPLETE", "PARTIAL"}
    if args.phase == "e2e" and result.get("text_to_moebius", {}).get("status") == "MEASURED":
        ok = True
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
