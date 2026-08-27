"""Unit tests for mask refinement and history."""

from __future__ import annotations

import unittest

import numpy as np

from models.types import validate_image, validate_mask
from pipelines.errors import PipelineValidationError
from pipelines.mask_history import MaskHistory
from pipelines.mask_refinement import (
    binary_dilate,
    binary_erode,
    refine_mask,
    require_non_empty_mask,
)
from pipelines.types import MaskRefinementOps


def _rgb(h: int = 16, w: int = 16) -> np.ndarray:
    return validate_image(np.zeros((h, w, 3), dtype=np.uint8))


def _mask(h: int = 16, w: int = 16, *, filled: bool = True) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    if filled:
        m[4:12, 4:12] = True
    return validate_mask(m)


class TestMorphology(unittest.TestCase):
    def test_dilation_expands_area(self) -> None:
        base = _mask()
        before = int(base.sum())
        dilated = binary_dilate(base, 1)
        self.assertGreater(int(dilated.sum()), before)

    def test_erosion_shrinks_area(self) -> None:
        base = _mask()
        before = int(base.sum())
        eroded = binary_erode(base, 1)
        self.assertLess(int(eroded.sum()), before)

    def test_dilation_zero_iterations_is_identity(self) -> None:
        base = _mask()
        np.testing.assert_array_equal(binary_dilate(base, 0), base)

    def test_erosion_zero_iterations_is_identity(self) -> None:
        base = _mask()
        np.testing.assert_array_equal(binary_erode(base, 0), base)


class TestRefineMask(unittest.TestCase):
    def test_add_remove_regions(self) -> None:
        img = _rgb()
        base = np.zeros((16, 16), dtype=bool)
        base[2:6, 2:6] = True
        add = np.zeros((16, 16), dtype=bool)
        add[10:12, 10:12] = True
        remove = np.zeros((16, 16), dtype=bool)
        remove[3:4, 3:4] = True

        refined = refine_mask(
            base,
            img,
            MaskRefinementOps(add=add, remove=remove),
        )
        self.assertTrue(refined[2, 2])
        self.assertFalse(refined[3, 3])
        self.assertTrue(refined[10, 10])

    def test_dilate_via_refine_mask(self) -> None:
        img = _rgb()
        base = _mask()
        dilated = refine_mask(base, img, MaskRefinementOps(dilate=2))
        self.assertGreater(int(dilated.sum()), int(base.sum()))

    def test_erode_via_refine_mask(self) -> None:
        img = _rgb()
        base = _mask()
        eroded = refine_mask(base, img, MaskRefinementOps(erode=2))
        self.assertLess(int(eroded.sum()), int(base.sum()))

    def test_dimension_mismatch_raises(self) -> None:
        img = _rgb(8, 8)
        mask = _mask(16, 16)
        with self.assertRaises(ValueError):
            refine_mask(mask, img)

    def test_empty_mask_rejected_for_inpaint(self) -> None:
        with self.assertRaises(PipelineValidationError):
            require_non_empty_mask(_mask(filled=False), stage="test")


class TestMaskHistory(unittest.TestCase):
    def test_undo_redo_roundtrip(self) -> None:
        history = MaskHistory(max_entries=10)
        m0 = _mask()
        m1 = m0.copy()
        m1[0, 0] = False
        m2 = m1.copy()
        m2[1, 1] = False

        history.push(m0)
        history.push(m1)
        current = m2.copy()

        undone = history.undo(current)
        self.assertIsNotNone(undone)
        np.testing.assert_array_equal(undone, m1)

        redone = history.redo(m2)
        self.assertIsNotNone(redone)
        np.testing.assert_array_equal(redone, m2)

    def test_undo_empty_returns_none(self) -> None:
        history = MaskHistory()
        self.assertIsNone(history.undo(_mask()))

    def test_redo_empty_returns_none(self) -> None:
        history = MaskHistory()
        self.assertIsNone(history.redo(_mask()))

    def test_reset_clears_history(self) -> None:
        history = MaskHistory()
        history.push(_mask())
        history.clear()
        self.assertFalse(history.can_undo())
        self.assertFalse(history.can_redo())

    def test_bounded_history(self) -> None:
        history = MaskHistory(max_entries=2)
        for i in range(4):
            m = _mask()
            m[0, i % 16] = True
            history.push(m)
        self.assertTrue(history.can_undo())
        current = _mask()
        first_undo = history.undo(current)
        assert first_undo is not None
        second_undo = history.undo(first_undo)
        assert second_undo is not None
        self.assertFalse(history.can_undo())


if __name__ == "__main__":
    unittest.main()
