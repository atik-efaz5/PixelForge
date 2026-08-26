"""Application-level mask refinement. Output is always bool H×W."""

from __future__ import annotations

import numpy as np

from models.types import MaskArray, validate_mask
from pipelines.errors import PipelineValidationError
from pipelines.types import MaskRefinementOps


def refine_mask(
    mask: np.ndarray,
    image: np.ndarray,
    operations: MaskRefinementOps | None = None,
    *,
    add: np.ndarray | None = None,
    remove: np.ndarray | None = None,
    dilate: int = 0,
    erode: int = 0,
) -> MaskArray:
    """Apply lightweight boolean mask operations. No resizing."""
    out = validate_mask(mask, image=image)
    if operations is not None:
        add = operations.add if add is None else add
        remove = operations.remove if remove is None else remove
        dilate = operations.dilate if dilate == 0 else dilate
        erode = operations.erode if erode == 0 else erode

    if add is not None:
        region = validate_mask(add, image=image)
        out = out | region
    if remove is not None:
        region = validate_mask(remove, image=image)
        out = out & ~region
    if erode > 0:
        out = _binary_erode(out, erode)
    if dilate > 0:
        out = _binary_dilate(out, dilate)
    return validate_mask(out, image=image)


def _binary_dilate(mask: np.ndarray, iterations: int) -> np.ndarray:
    out = mask.astype(bool, copy=True)
    for _ in range(iterations):
        padded = np.pad(out, 1, mode="constant", constant_values=False)
        h, w = out.shape
        merged = np.zeros_like(out)
        for dy in range(3):
            for dx in range(3):
                merged |= padded[dy : dy + h, dx : dx + w]
        out = merged
    return out


def _binary_erode(mask: np.ndarray, iterations: int) -> np.ndarray:
    return ~_binary_dilate(~mask, iterations)


def require_non_empty_mask(mask: np.ndarray, *, stage: str) -> MaskArray:
    out = validate_mask(mask)
    if not bool(out.any()):
        raise PipelineValidationError(f"Mask is empty after {stage}; inpainting requires a region.")
    return out
