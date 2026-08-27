"""Unit tests for the experiment runner."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.runner import ExperimentInputs, ExperimentRunner
from evaluation.types import ExperimentConfig, MetricKind


def _rgb() -> np.ndarray:
    return np.zeros((16, 16, 3), dtype=np.uint8)


def _mask() -> np.ndarray:
    m = np.zeros((16, 16), dtype=bool)
    m[4:12, 4:12] = True
    return m


class TestExperimentRunner(unittest.TestCase):
    def test_run_stages_records_latency(self) -> None:
        config = ExperimentConfig(
            experiment_id="runner-demo",
            model="sam2",
            backend="LOCAL_MPS",
            operation="segment",
        )
        runner = ExperimentRunner(
            config=config,
            inputs=ExperimentInputs(image=_rgb(), mask=_mask()),
        )
        record = runner.run_stages(
            {
                "segment": lambda: "mask",
                "inpaint": lambda: "result",
            }
        )
        self.assertEqual(len(record.timings), 2)
        self.assertEqual(record.timings[0].stage, "segment")
        self.assertGreaterEqual(record.timings[0].latency_ms, 0.0)

    def test_reference_dependent_metrics_when_reference_present(self) -> None:
        pred = _mask()
        ref = pred.copy()
        ref[5, 5] = False
        config = ExperimentConfig(
            experiment_id="mask-ref",
            model="sam2",
            backend="LOCAL_MPS",
            operation="segment",
        )
        runner = ExperimentRunner(
            config=config,
            inputs=ExperimentInputs(
                image=_rgb(),
                mask=pred,
                reference_mask=ref,
            ),
        )
        record = runner.run_stages({"segment": lambda: None})
        names = {m.name: m for m in record.metrics}
        self.assertIn("iou", names)
        self.assertEqual(names["iou"].kind, MetricKind.REFERENCE_DEPENDENT)
        self.assertIsNotNone(names["iou"].value)

    def test_ssim_reported_unavailable_not_zero(self) -> None:
        config = ExperimentConfig(
            experiment_id="no-ssim",
            model="moebius",
            backend="LOCAL_MPS",
            operation="inpaint",
        )
        runner = ExperimentRunner(config=config, inputs=ExperimentInputs(image=_rgb()))
        record = runner.run_stages({})
        ssim = next(m for m in record.metrics if m.name == "ssim")
        self.assertEqual(ssim.kind, MetricKind.UNAVAILABLE)
        self.assertIsNone(ssim.value)

    def test_write_report_json(self) -> None:
        config = ExperimentConfig(
            experiment_id="report-json",
            model="moebius",
            backend="LOCAL_MPS",
            operation="inpaint",
        )
        runner = ExperimentRunner(
            config=config,
            inputs=ExperimentInputs(
                image=_rgb(),
                mask=_mask(),
                output=_rgb(),
                reference_image=_rgb(),
            ),
        )
        record = runner.run_stages({"inpaint": lambda: None})
        with tempfile.TemporaryDirectory() as tmp:
            path = runner.write_report(record, Path(tmp) / "report.json")
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["experiment_id"], "report-json")
            self.assertIn("metrics", payload)
            self.assertIn("reproducibility", payload)


if __name__ == "__main__":
    unittest.main()
