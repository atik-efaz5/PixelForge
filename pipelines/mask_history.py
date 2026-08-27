"""Undo/redo stack for boolean H×W masks."""

from __future__ import annotations

import numpy as np

from models.types import MaskArray, validate_mask


class MaskHistory:
    """Bounded undo/redo history for mask edits."""

    def __init__(self, *, max_entries: int = 50) -> None:
        self._undo: list[MaskArray] = []
        self._redo: list[MaskArray] = []
        self._max_entries = max(1, max_entries)

    def can_undo(self) -> bool:
        return bool(self._undo)

    def can_redo(self) -> bool:
        return bool(self._redo)

    def clear(self) -> None:
        self._undo.clear()
        self._redo.clear()

    def push(self, mask: np.ndarray) -> None:
        """Record a mask snapshot before a new edit."""
        self._undo.append(validate_mask(mask).copy())
        self._redo.clear()
        if len(self._undo) > self._max_entries:
            self._undo.pop(0)

    def undo(self, current: np.ndarray) -> MaskArray | None:
        if not self._undo:
            return None
        self._redo.append(validate_mask(current).copy())
        return self._undo.pop().copy()

    def redo(self, current: np.ndarray) -> MaskArray | None:
        if not self._redo:
            return None
        self._undo.append(validate_mask(current).copy())
        return self._redo.pop().copy()

    def reset_to(self, mask: np.ndarray) -> MaskArray:
        """Replace history and return a copy of the given mask."""
        self.clear()
        return validate_mask(mask).copy()
