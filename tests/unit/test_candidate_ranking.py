"""Unit tests for deterministic inpainting candidate ranking."""

from __future__ import annotations

import unittest

import numpy as np

from evaluation.candidate_ranking import rank_inpaint_candidates, score_inpaint_candidate
from models.types import validate_image, validate_mask


def _rgb(h: int = 32, w: int = 32) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.stack(
        [
            (20 + yy).astype(np.uint8),
            (30 + xx).astype(np.uint8),
            (40 + (yy + xx) // 2).astype(np.uint8),
        ],
        axis=-1,
    )
    return validate_image(img)


def _mask(h: int = 32, w: int = 32) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    m[10:20, 10:20] = True
    return validate_mask(m)


class TestCandidateRanking(unittest.TestCase):
    def test_rejects_invalid_output(self) -> None:
        original = _rgb()
        mask = _mask()
        bad = np.zeros((16, 16, 3), dtype=np.uint8)
        scored = score_inpaint_candidate(original, mask, "candidate_1", bad)
        self.assertTrue(scored.rejected)
        self.assertEqual(scored.validity_status, "rejected")

    def test_ranking_orders_by_score(self) -> None:
        original = _rgb()
        mask = _mask()
        better = original.copy()
        better[mask] = np.array([200, 120, 80], dtype=np.uint8)
        worse = original.copy()
        worse[mask] = np.array([200, 120, 80], dtype=np.uint8)
        worse[0, 0] = np.array([0, 0, 0], dtype=np.uint8)
        ranking = rank_inpaint_candidates(
            original,
            mask,
            [("candidate_2", worse), ("candidate_1", better)],
        )
        self.assertEqual(ranking.selected.candidate_id, "candidate_1")
        self.assertEqual(ranking.ordered[0].candidate_id, "candidate_1")

    def test_unavailable_metric_not_zeroed_without_note(self) -> None:
        original = _rgb()
        mask = np.zeros((32, 32), dtype=bool)
        mask[:, :] = True
        edited = original.copy()
        scored = score_inpaint_candidate(original, mask, "candidate_1", edited)
        preservation = scored.components["preservation_outside_mask"]
        self.assertFalse(preservation.available)
        self.assertIsNone(preservation.value)
        self.assertIsNotNone(preservation.note)

    def test_eval_record_contains_ranking_fields(self) -> None:
        original = _rgb()
        mask = _mask()
        edited = original.copy()
        edited[mask] = np.array([180, 100, 60], dtype=np.uint8)
        ranking = rank_inpaint_candidates(
            original, mask, [("candidate_1", edited)]
        )
        record = ranking.to_eval_record(candidate_count=1)
        self.assertEqual(record["candidate_count"], 1)
        self.assertIn("candidate_hashes", record)
        self.assertIn("scores", record)


if __name__ == "__main__":
    unittest.main()
