"""Binary mask metric helpers (pure numpy)."""

from __future__ import annotations

import numpy as np

from evaluation.types import MetricKind, MetricResult


def _as_bool_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError(f"mask must be H×W, got shape {mask.shape}")
    if mask.dtype == bool or mask.dtype == np.bool_:
        return mask.astype(bool, copy=False)
    raise ValueError(f"mask dtype must be bool, got {mask.dtype}")


def _confusion(pred: np.ndarray, ref: np.ndarray) -> tuple[int, int, int, int]:
    pred_b = _as_bool_mask(pred)
    ref_b = _as_bool_mask(ref)
    if pred_b.shape != ref_b.shape:
        raise ValueError(
            f"mask shapes must match: pred {pred_b.shape} vs ref {ref_b.shape}"
        )
    tp = int(np.logical_and(pred_b, ref_b).sum())
    fp = int(np.logical_and(pred_b, np.logical_not(ref_b)).sum())
    fn = int(np.logical_and(np.logical_not(pred_b), ref_b).sum())
    tn = int(np.logical_and(np.logical_not(pred_b), np.logical_not(ref_b)).sum())
    return tp, fp, fn, tn


def mask_area_ratio(mask: np.ndarray) -> MetricResult:
    """Fraction of pixels marked True in the mask."""
    m = _as_bool_mask(mask)
    area = int(m.sum())
    total = m.size
    if total == 0:
        return MetricResult(
            name="mask_area_ratio",
            value=None,
            kind=MetricKind.UNDEFINED,
            unit="ratio",
            note="empty mask tensor",
        )
    return MetricResult(
        name="mask_area_ratio",
        value=area / total,
        kind=MetricKind.COMPUTED,
        unit="ratio",
    )


def iou(pred: np.ndarray, ref: np.ndarray) -> MetricResult:
    """Intersection over union. Requires reference mask."""
    tp, fp, fn, _tn = _confusion(pred, ref)
    union = tp + fp + fn
    if union == 0:
        return MetricResult(
            name="iou",
            value=None,
            kind=MetricKind.REFERENCE_DEPENDENT,
            unit="ratio",
            note="union is zero (both masks empty)",
        )
    return MetricResult(
        name="iou",
        value=tp / union,
        kind=MetricKind.REFERENCE_DEPENDENT,
        unit="ratio",
    )


def precision(pred: np.ndarray, ref: np.ndarray) -> MetricResult:
    """TP / (TP + FP). Requires reference mask."""
    tp, fp, _fn, _tn = _confusion(pred, ref)
    denom = tp + fp
    if denom == 0:
        return MetricResult(
            name="precision",
            value=None,
            kind=MetricKind.REFERENCE_DEPENDENT,
            unit="ratio",
            note="no positive predictions",
        )
    return MetricResult(
        name="precision",
        value=tp / denom,
        kind=MetricKind.REFERENCE_DEPENDENT,
        unit="ratio",
    )


def recall(pred: np.ndarray, ref: np.ndarray) -> MetricResult:
    """TP / (TP + FN). Requires reference mask."""
    tp, _fp, fn, _tn = _confusion(pred, ref)
    denom = tp + fn
    if denom == 0:
        return MetricResult(
            name="recall",
            value=None,
            kind=MetricKind.REFERENCE_DEPENDENT,
            unit="ratio",
            note="no positive reference pixels",
        )
    return MetricResult(
        name="recall",
        value=tp / denom,
        kind=MetricKind.REFERENCE_DEPENDENT,
        unit="ratio",
    )


def f1(pred: np.ndarray, ref: np.ndarray) -> MetricResult:
    """Harmonic mean of precision and recall. Requires reference mask."""
    p = precision(pred, ref)
    r = recall(pred, ref)
    if p.value is None or r.value is None:
        note = p.note or r.note or "precision or recall undefined"
        return MetricResult(
            name="f1",
            value=None,
            kind=MetricKind.REFERENCE_DEPENDENT,
            unit="ratio",
            note=note,
        )
    denom = p.value + r.value
    if denom == 0:
        return MetricResult(
            name="f1",
            value=None,
            kind=MetricKind.REFERENCE_DEPENDENT,
            unit="ratio",
            note="precision and recall are both zero",
        )
    return MetricResult(
        name="f1",
        value=2 * p.value * r.value / denom,
        kind=MetricKind.REFERENCE_DEPENDENT,
        unit="ratio",
    )


def _binary_dilate(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
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


def _mask_boundary(mask: np.ndarray, *, width: int = 1) -> np.ndarray:
    m = _as_bool_mask(mask)
    if width < 1:
        return np.zeros_like(m)
    dilated = _binary_dilate(m, width)
    eroded = ~_binary_dilate(~m, width)
    return np.logical_and(dilated, np.logical_not(eroded))


def boundary_overlap(pred: np.ndarray, ref: np.ndarray, *, width: int = 1) -> MetricResult:
    """IoU on morphological boundary bands. Requires reference mask."""
    pred_b = _mask_boundary(pred, width=width)
    ref_b = _mask_boundary(ref, width=width)
    return iou(pred_b, ref_b)


def boundary_difference_ratio(
    pred: np.ndarray, ref: np.ndarray, *, width: int = 1
) -> MetricResult:
    """Symmetric difference of boundary bands divided by union of boundary pixels."""
    pred_b = _mask_boundary(pred, width=width)
    ref_b = _mask_boundary(ref, width=width)
    xor = np.logical_xor(pred_b, ref_b)
    union = np.logical_or(pred_b, ref_b)
    union_count = int(union.sum())
    if union_count == 0:
        return MetricResult(
            name="boundary_difference_ratio",
            value=None,
            kind=MetricKind.REFERENCE_DEPENDENT,
            unit="ratio",
            note="no boundary pixels in either mask",
        )
    return MetricResult(
        name="boundary_difference_ratio",
        value=int(xor.sum()) / union_count,
        kind=MetricKind.REFERENCE_DEPENDENT,
        unit="ratio",
    )
