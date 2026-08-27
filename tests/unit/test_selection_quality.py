"""Unit tests for deterministic selection-quality heuristics."""

from __future__ import annotations

import unittest

import numpy as np

from pipelines.selection_quality import (
    ConfidenceTier,
    OVERSIZED_REJECT_RATIO,
    TINY_REJECT_RATIO,
    rank_candidates,
    score_mask_candidate,
)
from models.types import validate_mask


def _mask(h: int = 64, w: int = 64, *, region: tuple[slice, slice] | None = None) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    if region is not None:
        m[region] = True
    return validate_mask(m)


class TestSelectionQuality(unittest.TestCase):
    def test_tiny_mask_rejected(self) -> None:
        tiny = _mask(region=(slice(0, 1), slice(0, 1)))
        scored = score_mask_candidate(tiny, candidate_id="tiny", point_xy=(0, 0))
        self.assertTrue(scored.rejected)
        self.assertEqual(scored.rejection_reason, "mask too small")
        self.assertLess(scored.mask_area_ratio, TINY_REJECT_RATIO)

    def test_oversized_mask_rejected(self) -> None:
        big = np.ones((64, 64), dtype=bool)
        scored = score_mask_candidate(big, candidate_id="big")
        self.assertTrue(scored.rejected)
        self.assertEqual(scored.rejection_reason, "mask too large")
        self.assertGreater(scored.mask_area_ratio, OVERSIZED_REJECT_RATIO)

    def test_point_containment_required(self) -> None:
        mask = _mask(region=(slice(10, 30), slice(10, 30)))
        scored = score_mask_candidate(mask, candidate_id="off", point_xy=(50, 50))
        self.assertTrue(scored.rejected)
        self.assertEqual(scored.rejection_reason, "point outside mask")

    def test_point_containment_passes(self) -> None:
        mask = _mask(region=(slice(10, 30), slice(10, 30)))
        scored = score_mask_candidate(mask, candidate_id="hit", point_xy=(20, 20))
        self.assertFalse(scored.rejected)
        self.assertIn("contains prompt point", scored.reasons)

    def test_grounding_box_overlap_prefers_better_iou(self) -> None:
        good = _mask(region=(slice(10, 40), slice(10, 40)))
        poor = _mask(region=(slice(0, 10), slice(0, 10)))
        box = (8.0, 8.0, 42.0, 42.0)
        good_score = score_mask_candidate(good, candidate_id="good", box_xyxy=box)
        poor_score = score_mask_candidate(poor, candidate_id="poor", box_xyxy=box)
        self.assertGreater(good_score.score, poor_score.score)

    def test_deterministic_ranking(self) -> None:
        a = score_mask_candidate(
            _mask(region=(slice(12, 36), slice(12, 36))),
            candidate_id="a",
            point_xy=(24, 24),
            sam_confidence=0.8,
        )
        b = score_mask_candidate(
            _mask(region=(slice(14, 34), slice(14, 34))),
            candidate_id="b",
            point_xy=(24, 24),
            sam_confidence=0.7,
        )
        first = rank_candidates([a, b])
        second = rank_candidates([a, b])
        self.assertEqual(first.selected.candidate_id, second.selected.candidate_id)

    def test_confidence_tiers_assigned(self) -> None:
        strong = score_mask_candidate(
            _mask(region=(slice(10, 30), slice(10, 30))),
            candidate_id="strong",
            point_xy=(20, 20),
            box_xyxy=(5.0, 5.0, 35.0, 35.0),
            sam_confidence=0.95,
        )
        self.assertIn(strong.tier, {ConfidenceTier.HIGH, ConfidenceTier.MEDIUM, ConfidenceTier.LOW})
        self.assertFalse(strong.rejected)

    def test_rank_raises_when_all_rejected(self) -> None:
        tiny = score_mask_candidate(
            _mask(region=(slice(0, 1), slice(0, 1))),
            candidate_id="tiny",
            point_xy=(0, 0),
        )
        with self.assertRaises(ValueError):
            rank_candidates([tiny])


if __name__ == "__main__":
    unittest.main()
