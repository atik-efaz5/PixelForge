"""Unit tests for evaluation metrics."""

from __future__ import annotations

import unittest

import numpy as np

from evaluation.metrics.image_metrics import mse, psnr, ssim_unavailable
from evaluation.metrics.mask_metrics import (
    boundary_difference_ratio,
    f1,
    iou,
    mask_area_ratio,
    precision,
    recall,
)
from evaluation.metrics.preservation import (
    outside_mask_mae,
    outside_mask_preservation_score,
)
from evaluation.types import MetricKind


def _mask(*, on: tuple[slice, slice] | None = (slice(2, 6), slice(2, 6))) -> np.ndarray:
    m = np.zeros((8, 8), dtype=bool)
    if on is not None:
        m[on] = True
    return m


def _rgb(fill: int = 100) -> np.ndarray:
    return np.full((8, 8, 3), fill, dtype=np.uint8)


class TestMaskMetrics(unittest.TestCase):
    def test_perfect_mask_iou_is_one(self) -> None:
        m = _mask()
        self.assertEqual(iou(m, m).value, 1.0)

    def test_disjoint_masks_iou_is_zero(self) -> None:
        a = _mask(on=(slice(0, 2), slice(0, 2)))
        b = _mask(on=(slice(6, 8), slice(6, 8)))
        self.assertEqual(iou(a, b).value, 0.0)

    def test_partial_overlap_iou(self) -> None:
        a = _mask()
        b = _mask(on=(slice(4, 8), slice(4, 8)))
        result = iou(a, b)
        self.assertIsNotNone(result.value)
        assert result.value is not None
        self.assertGreater(result.value, 0.0)
        self.assertLess(result.value, 1.0)

    def test_empty_masks_iou_undefined(self) -> None:
        empty = np.zeros((8, 8), dtype=bool)
        result = iou(empty, empty)
        self.assertIsNone(result.value)
        self.assertEqual(result.kind, MetricKind.REFERENCE_DEPENDENT)

    def test_precision_recall_f1_perfect(self) -> None:
        m = _mask()
        self.assertEqual(precision(m, m).value, 1.0)
        self.assertEqual(recall(m, m).value, 1.0)
        self.assertEqual(f1(m, m).value, 1.0)

    def test_precision_zero_denominator(self) -> None:
        pred = np.zeros((8, 8), dtype=bool)
        ref = _mask()
        result = precision(pred, ref)
        self.assertIsNone(result.value)
        self.assertIn("no positive predictions", result.note or "")

    def test_recall_zero_denominator(self) -> None:
        pred = _mask()
        ref = np.zeros((8, 8), dtype=bool)
        result = recall(pred, ref)
        self.assertIsNone(result.value)

    def test_mask_area_ratio(self) -> None:
        m = _mask()
        ratio = mask_area_ratio(m)
        self.assertIsNotNone(ratio.value)
        assert ratio.value is not None
        self.assertAlmostEqual(ratio.value, 16 / 64)

    def test_boundary_difference_ratio(self) -> None:
        a = _mask()
        b = _mask(on=(slice(3, 7), slice(3, 7)))
        result = boundary_difference_ratio(a, b)
        self.assertIsNotNone(result.value)


class TestImageMetrics(unittest.TestCase):
    def test_identical_images_infinite_psnr(self) -> None:
        img = _rgb(120)
        result = psnr(img, img)
        self.assertEqual(result.value, float("inf"))

    def test_different_images_finite_psnr(self) -> None:
        a = _rgb(0)
        b = _rgb(254)
        result = psnr(a, b)
        self.assertIsNotNone(result.value)
        assert result.value is not None
        self.assertGreaterEqual(result.value, 0.0)

    def test_mse_identical_zero(self) -> None:
        img = _rgb(50)
        self.assertEqual(mse(img, img).value, 0.0)

    def test_ssim_unavailable(self) -> None:
        result = ssim_unavailable()
        self.assertEqual(result.kind, MetricKind.UNAVAILABLE)
        self.assertIsNone(result.value)


class TestPreservationMetrics(unittest.TestCase):
    def test_identical_outside_preservation_is_one(self) -> None:
        img = _rgb(80)
        mask = _mask()
        edited = img.copy()
        edited[mask] = 200
        score = outside_mask_preservation_score(img, edited, mask)
        self.assertIsNotNone(score.value)
        assert score.value is not None
        self.assertAlmostEqual(score.value, 1.0)

    def test_changed_outside_lowers_preservation(self) -> None:
        img = _rgb(80)
        mask = _mask()
        edited = img.copy()
        edited[~mask] = 0
        score = outside_mask_preservation_score(img, edited, mask)
        self.assertIsNotNone(score.value)
        assert score.value is not None
        self.assertLess(score.value, 1.0)

    def test_full_mask_outside_mae_undefined(self) -> None:
        img = _rgb(10)
        mask = np.ones((8, 8), dtype=bool)
        result = outside_mask_mae(img, img, mask)
        self.assertIsNone(result.value)


if __name__ == "__main__":
    unittest.main()
