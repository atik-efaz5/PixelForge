"""In-memory edit session history (no persistence).

Represents meaningful editing milestones for a single user session. Pixel-level
mask undo/redo remains separate in :mod:`pipelines.mask_history`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import time
from typing import Any
from uuid import uuid4

import numpy as np

from models.types import MaskArray, validate_mask


class EditOperation(str, Enum):
    ORIGINAL = "ORIGINAL"
    SELECT = "SELECT"
    MASK_EDIT = "MASK_EDIT"
    MASK_RESET = "MASK_RESET"
    INPAINT = "INPAINT"
    RESULT_ACCEPTED = "RESULT_ACCEPTED"
    RESULT_REJECTED = "RESULT_REJECTED"


@dataclass(frozen=True)
class SelectionMetadata:
    method: str
    prompt: str | None = None
    label: str | None = None
    detection_index: int | None = None


@dataclass(frozen=True)
class InpaintMetadata:
    backend: str
    model: str | None = None
    latency_ms: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class EditSessionSnapshot:
    """One milestone in the edit session."""

    operation: EditOperation
    label: str
    original_ref: str
    mask: MaskArray | None = None
    ai_mask: MaskArray | None = None
    result_ref: str | None = None
    selection: SelectionMetadata | None = None
    inpaint: InpaintMetadata | None = None
    id: str = field(default_factory=lambda: uuid4().hex)
    timestamp: float = field(default_factory=time)

    def __post_init__(self) -> None:
        if self.mask is not None:
            self.mask = validate_mask(self.mask)
        if self.ai_mask is not None:
            self.ai_mask = validate_mask(self.ai_mask)


class EditSessionHistory:
    """Bounded linear history with undo/redo and branch truncation."""

    def __init__(self, *, max_entries: int = 30) -> None:
        self._entries: list[EditSessionSnapshot] = []
        self._index = -1
        self._max_entries = max(1, max_entries)
        self._original_ref: str | None = None

    @property
    def entries(self) -> tuple[EditSessionSnapshot, ...]:
        return tuple(self._entries)

    @property
    def index(self) -> int:
        return self._index

    @property
    def original_ref(self) -> str | None:
        return self._original_ref

    def can_undo(self) -> bool:
        return self._index > 0

    def can_redo(self) -> bool:
        return 0 <= self._index < len(self._entries) - 1

    def current(self) -> EditSessionSnapshot | None:
        if self._index < 0 or self._index >= len(self._entries):
            return None
        return self._entries[self._index]

    def reset(self, original_ref: str) -> EditSessionSnapshot:
        """Start a new session with the immutable original image reference."""
        self._entries.clear()
        self._index = -1
        self._original_ref = original_ref
        return self.append(
            EditOperation.ORIGINAL,
            label="Original",
            mask=None,
            ai_mask=None,
            result_ref=None,
        )

    def append(
        self,
        operation: EditOperation,
        *,
        label: str,
        mask: np.ndarray | None = None,
        ai_mask: np.ndarray | None = None,
        result_ref: str | None = None,
        selection: SelectionMetadata | None = None,
        inpaint: InpaintMetadata | None = None,
    ) -> EditSessionSnapshot:
        if self._original_ref is None:
            raise ValueError("Session not initialized; call reset() first.")

        if self._index < len(self._entries) - 1:
            self._entries = self._entries[: self._index + 1]

        entry = EditSessionSnapshot(
            operation=operation,
            label=label,
            original_ref=self._original_ref,
            mask=mask.copy() if mask is not None else None,
            ai_mask=ai_mask.copy() if ai_mask is not None else None,
            result_ref=result_ref,
            selection=selection,
            inpaint=inpaint,
        )
        self._entries.append(entry)
        self._index = len(self._entries) - 1
        if len(self._entries) > self._max_entries:
            dropped = self._entries.pop(0)
            self._index -= 1
            if dropped.original_ref != self._original_ref:
                pass  # caller owns URL lifecycle
        return entry

    def undo(self) -> EditSessionSnapshot | None:
        if not self.can_undo():
            return None
        self._index -= 1
        return self.current()

    def redo(self) -> EditSessionSnapshot | None:
        if not self.can_redo():
            return None
        self._index += 1
        return self.current()

    def go_to(self, index: int) -> EditSessionSnapshot | None:
        if index < 0 or index >= len(self._entries):
            return None
        self._index = index
        return self.current()

    def previous_accepted_result(self) -> EditSessionSnapshot | None:
        """Walk backward for the latest snapshot with an accepted result."""
        for idx in range(self._index - 1, -1, -1):
            entry = self._entries[idx]
            if entry.result_ref and entry.operation in {
                EditOperation.INPAINT,
                EditOperation.RESULT_ACCEPTED,
            }:
                return entry
        return None
