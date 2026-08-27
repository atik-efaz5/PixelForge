"""RGB image metrics (pure numpy where possible)."""

from __future__ import annotations

import numpy as np

from evaluation.types import MetricKind, MetricResult


def _as_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"image must be H×W×3, got shape {image.shape}")
    if image.dtype != np.uint8:
        raise ValueError(f"image dtype must be uint8, got {image.dtype}")
    return image


def mse(a: np.ndarray, b: np.ndarray) -> MetricResult:
    """Mean squared error on uint8 RGB images."""
    img_a = _as_rgb(a)
    img_b = _as_rgb(b)
    if img_a.shape != img_b.shape:
        raise ValueError(f"image shapes must match: {img_a.shape} vs {img_b.shape}")
    diff = img_a.astype(np.float64) - img_b.astype(np.float64)
    value = float(np.mean(diff * diff))
    return MetricResult(
        name="mse",
        value=value,
        kind=MetricKind.COMPUTED,
        unit="uint8^2",
    )


def psnr(a: np.ndarray, b: np.ndarray, *, max_val: float = 255.0) -> MetricResult:
    """Peak signal-to-noise ratio: 10 * log10(MAX^2 / MSE). Requires reference image."""
    mse_result = mse(a, b)
    if mse_result.value is None:
        return MetricResult(
            name="psnr",
            value=None,
            kind=MetricKind.REFERENCE_DEPENDENT,
            unit="dB",
            note="mse undefined",
        )
    if mse_result.value == 0:
        return MetricResult(
            name="psnr",
            value=float("inf"),
            kind=MetricKind.REFERENCE_DEPENDENT,
            unit="dB",
            note="images are identical",
        )
    value = 10.0 * np.log10((max_val * max_val) / mse_result.value)
    return MetricResult(
        name="psnr",
        value=float(value),
        kind=MetricKind.REFERENCE_DEPENDENT,
        unit="dB",
    )


def ssim_unavailable() -> MetricResult:
    """SSIM is not computed — scikit-image/scipy are not project dependencies."""
    return MetricResult(
        name="ssim",
        value=None,
        kind=MetricKind.UNAVAILABLE,
        unit="ratio",
        note="SSIM requires scikit-image; not installed in PixelForge evaluation stack",
    )


def lpips_unavailable() -> MetricResult:
    """LPIPS is not computed — torch/LPIPS are not evaluation dependencies."""
    return MetricResult(
        name="lpips",
        value=None,
        kind=MetricKind.UNAVAILABLE,
        unit="distance",
        note="LPIPS not available without additional model dependencies",
    )
