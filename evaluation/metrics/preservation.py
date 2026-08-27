"""Preservation metrics for localized image editing."""

from __future__ import annotations

import numpy as np

from evaluation.types import MetricKind, MetricResult


def _as_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must be H×W×3, got shape {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"image dtype must be uint8, got {image.dtype}")
    return image


def _as_bool_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError(f"mask must be H×W, got shape {mask.shape}")
    if mask.dtype == bool or mask.dtype == np.bool_:
        return mask.astype(bool, copy=False)
    raise ValueError(f"mask dtype must be bool, got {mask.dtype}")


def outside_mask_mae(
    original: np.ndarray,
    edited: np.ndarray,
    mask: np.ndarray,
) -> MetricResult:
    """Mean absolute difference outside the edit mask, normalized to [0, 1].

    Lower values indicate pixels outside the masked edit region were preserved
    more closely. This is **not** a perceptual quality score.
    """
    orig = _as_rgb(original)
    edit = _as_rgb(edited)
    m = _as_bool_mask(mask)
    if orig.shape[:2] != m.shape:
        raise ValueError("mask spatial dimensions must match image H×W")
    if orig.shape != edit.shape:
        raise ValueError("original and edited images must have the same shape")

    outside = ~m
    count = int(outside.sum())
    if count == 0:
        return MetricResult(
            name="outside_mask_mae",
            value=None,
            kind=MetricKind.COMPUTED,
            unit="normalized",
            note="no pixels outside mask",
        )

    diff = np.abs(orig.astype(np.float64) - edit.astype(np.float64))
    outside_diff = diff[outside]
    mae = float(outside_diff.mean() / 255.0)
    return MetricResult(
        name="outside_mask_mae",
        value=mae,
        kind=MetricKind.COMPUTED,
        unit="normalized",
        note="mean absolute difference outside mask; not a perceptual metric",
    )


def outside_mask_preservation_score(
    original: np.ndarray,
    edited: np.ndarray,
    mask: np.ndarray,
) -> MetricResult:
    """``1 - outside_mask_mae`` when defined. Higher ≈ more preserved outside mask."""
    mae = outside_mask_mae(original, edited, mask)
    if mae.value is None:
        return MetricResult(
            name="outside_mask_preservation",
            value=None,
            kind=MetricKind.COMPUTED,
            unit="ratio",
            note=mae.note,
        )
    return MetricResult(
        name="outside_mask_preservation",
        value=max(0.0, 1.0 - mae.value),
        kind=MetricKind.COMPUTED,
        unit="ratio",
        note="1 - normalized outside-mask MAE; not a perceptual metric",
    )
