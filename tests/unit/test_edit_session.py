"""Unit tests for edit session history."""

from __future__ import annotations

import unittest

import numpy as np

from models.types import validate_mask
from pipelines.edit_session import (
    EditOperation,
    EditSessionHistory,
    InpaintMetadata,
    SelectionMetadata,
)


def _mask(*, on: bool = True) -> np.ndarray:
    m = np.zeros((8, 8), dtype=bool)
    if on:
        m[2:6, 2:6] = True
    return validate_mask(m)


class TestEditSessionHistory(unittest.TestCase):
    def setUp(self) -> None:
        self.history = EditSessionHistory(max_entries=10)

    def test_reset_creates_original_entry(self) -> None:
        entry = self.history.reset("original://image")
        self.assertEqual(entry.operation, EditOperation.ORIGINAL)
        self.assertEqual(entry.original_ref, "original://image")
        self.assertIsNone(entry.mask)

    def test_append_advances_index(self) -> None:
        self.history.reset("original://image")
        self.history.append(
            EditOperation.SELECT,
            label="Object selected",
            mask=_mask(),
        )
        self.assertEqual(len(self.history.entries), 2)
        self.assertEqual(self.history.index, 1)

    def test_undo_redo_navigation(self) -> None:
        self.history.reset("original://image")
        self.history.append(EditOperation.SELECT, label="Selected", mask=_mask())
        current = self.history.undo()
        self.assertIsNotNone(current)
        assert current is not None
        self.assertEqual(current.operation, EditOperation.ORIGINAL)

        redone = self.history.redo()
        self.assertIsNotNone(redone)
        assert redone is not None
        self.assertEqual(redone.operation, EditOperation.SELECT)

    def test_branch_invalidation_after_new_operation(self) -> None:
        self.history.reset("original://image")
        self.history.append(EditOperation.SELECT, label="A", mask=_mask())
        self.history.append(EditOperation.MASK_EDIT, label="B", mask=_mask())
        self.history.undo()
        self.history.undo()
        self.history.append(EditOperation.MASK_RESET, label="C", mask=_mask(on=False))
        self.assertEqual(len(self.history.entries), 2)
        self.assertFalse(self.history.can_redo())
        self.assertEqual(self.history.current().operation, EditOperation.MASK_RESET)

    def test_result_accept_reject_preserves_original(self) -> None:
        original = "original://immutable"
        self.history.reset(original)
        self.history.append(
            EditOperation.INPAINT,
            label="Generated",
            mask=_mask(),
            result_ref="result://1",
            inpaint=InpaintMetadata(
                backend="moebius",
                latency_ms=12.0,
                candidate_count=2,
                selected_candidate_id="candidate_1",
            ),
        )
        accepted = self.history.append(
            EditOperation.RESULT_ACCEPTED,
            label="Accepted",
            mask=_mask(),
            result_ref="result://1",
            inpaint=InpaintMetadata(
                backend="moebius",
                selected_candidate_id="candidate_1",
            ),
        )
        self.assertEqual(accepted.original_ref, original)
        self.assertEqual(accepted.inpaint.selected_candidate_id, "candidate_1")

        self.history.append(
            EditOperation.RESULT_REJECTED,
            label="Discarded",
            mask=_mask(),
            result_ref=None,
        )
        self.assertEqual(self.history.current().operation, EditOperation.RESULT_REJECTED)
        self.assertEqual(self.history.original_ref, original)

    def test_go_to_snapshot(self) -> None:
        self.history.reset("original://image")
        self.history.append(EditOperation.SELECT, label="Sel", mask=_mask())
        self.history.append(EditOperation.MASK_EDIT, label="Edit", mask=_mask())
        snap = self.history.go_to(0)
        self.assertIsNotNone(snap)
        assert snap is not None
        self.assertEqual(snap.operation, EditOperation.ORIGINAL)

    def test_previous_accepted_result(self) -> None:
        self.history.reset("original://image")
        self.history.append(
            EditOperation.INPAINT,
            label="Gen1",
            mask=_mask(),
            result_ref="r1",
        )
        self.history.append(
            EditOperation.RESULT_ACCEPTED,
            label="Acc1",
            mask=_mask(),
            result_ref="r1",
        )
        self.history.append(EditOperation.MASK_EDIT, label="Refine", mask=_mask())
        prev = self.history.previous_accepted_result()
        self.assertIsNotNone(prev)
        assert prev is not None
        self.assertEqual(prev.result_ref, "r1")

    def test_selection_metadata_stored(self) -> None:
        self.history.reset("original://image")
        entry = self.history.append(
            EditOperation.SELECT,
            label="Text select",
            mask=_mask(),
            selection=SelectionMetadata(
                method="text",
                prompt="dog",
                label="dog",
                detection_index=0,
            ),
        )
        self.assertIsNotNone(entry.selection)
        assert entry.selection is not None
        self.assertEqual(entry.selection.prompt, "dog")

    def test_original_ref_immutable_across_entries(self) -> None:
        original = "original://keep"
        self.history.reset(original)
        for op in (
            EditOperation.SELECT,
            EditOperation.MASK_EDIT,
            EditOperation.INPAINT,
        ):
            self.history.append(op, label=op.value, mask=_mask(), result_ref="r")
        for entry in self.history.entries:
            self.assertEqual(entry.original_ref, original)


if __name__ == "__main__":
    unittest.main()
