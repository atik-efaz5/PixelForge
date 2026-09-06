"""Crop around the inpaint mask, run Moebius at 512, then composite back.

Moebius is only validated at 512×512. Shrinking a whole shop photo to 512 makes the
object a few pixels; the upscaled fill looks like a ghost. Infer on a padded crop
around the mask instead so the hole uses the 512 budget. Restore feathers a dilated
mask so SAM’s tight edge does not leave a bag outline. Restore uses a hard fill core
plus a short feather so contact shadows are replaced instead of blended into a stain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter

from models.types import validate_image, validate_mask

DEFAULT_MOEBIUS_SIZE = 512
CROP_MARGIN_RATIO = 0.25
CROP_MARGIN_MIN_PX = 64
WORK_MASK_DILATE = 3
COMPOSITE_HALO_MIN = 40
COMPOSITE_HALO_MAX = 80
COMPOSITE_HALO_RATIO = 0.03
COMPOSITE_PHOTO_HALO_RATIO = 0.012
COMPOSITE_FEATHER_PX = 6
UNIFORM_NEIGHBOR_BAND_PX = 24
UNIFORM_NEAR_MAX = 18.0
UNIFORM_NEAR_FRACTION = 0.70
UNIFORM_MIN_SAMPLES = 64
UNIFORM_FILL_DILATE_PX = 6
UNIFORM_FILL_FEATHER_PX = 2


@dataclass(frozen=True)
class MoebiusWorkGeometry:
    orig_h: int
    orig_w: int
    crop_y0: int
    crop_x0: int
    crop_h: int
    crop_w: int
    scaled_h: int
    scaled_w: int
    pad_top: int
    pad_left: int
    work_size: int
    skipped: bool


def prepare_moebius_work(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    size: int = DEFAULT_MOEBIUS_SIZE,
) -> tuple[np.ndarray, np.ndarray, MoebiusWorkGeometry]:
    """Return a square ``size`` RGB/mask pair for the masked crop when the photo is large."""
    image = validate_image(image)
    mask = validate_mask(mask, image=image)
    orig_h, orig_w = image.shape[:2]
    if max(orig_h, orig_w) <= size:
        geom = MoebiusWorkGeometry(
            orig_h=orig_h,
            orig_w=orig_w,
            crop_y0=0,
            crop_x0=0,
            crop_h=orig_h,
            crop_w=orig_w,
            scaled_h=orig_h,
            scaled_w=orig_w,
            pad_top=0,
            pad_left=0,
            work_size=size,
            skipped=True,
        )
        work_mask = _dilate_mask(mask, _work_dilate_iterations(geom))
        return image, work_mask, geom

    y0, x0, y1, x1 = _expand_bbox(*_mask_bbox(mask), orig_h, orig_w)
    crop_rgb = image[y0:y1, x0:x1]
    crop_mask = mask[y0:y1, x0:x1]
    crop_h, crop_w = crop_rgb.shape[:2]

    if max(crop_h, crop_w) > size:
        scale = size / max(crop_h, crop_w)
        scaled_w = max(1, min(size, round(crop_w * scale)))
        scaled_h = max(1, min(size, round(crop_h * scale)))
        rgb = np.asarray(
            Image.fromarray(crop_rgb, mode="RGB").resize(
                (scaled_w, scaled_h), resample=Image.Resampling.LANCZOS
            ),
            dtype=np.uint8,
        )
        work_mask_small = _max_pool_mask(crop_mask, scaled_h, scaled_w)
    else:
        scaled_h, scaled_w = crop_h, crop_w
        rgb = crop_rgb
        work_mask_small = crop_mask.astype(bool, copy=False)

    work_image, work_mask, pad_top, pad_left = _pad_to_square(rgb, work_mask_small, size)
    geom = MoebiusWorkGeometry(
        orig_h=orig_h,
        orig_w=orig_w,
        crop_y0=y0,
        crop_x0=x0,
        crop_h=crop_h,
        crop_w=crop_w,
        scaled_h=scaled_h,
        scaled_w=scaled_w,
        pad_top=pad_top,
        pad_left=pad_left,
        work_size=size,
        skipped=False,
    )
    work_mask = _dilate_mask(work_mask, _work_dilate_iterations(geom))
    return work_image, work_mask, geom


def restore_moebius_result(
    original: np.ndarray,
    mask: np.ndarray,
    generated: np.ndarray,
    geom: MoebiusWorkGeometry,
) -> np.ndarray:
    """Paste the 512 crop fill back with a dilated, Gaussian-feathered mask."""
    original = validate_image(original)
    mask = validate_mask(mask, image=original)
    generated = validate_image(generated)

    if geom.skipped:
        filled = _resize_rgb(generated, geom.orig_w, geom.orig_h)
    else:
        square = _resize_rgb(generated, geom.work_size, geom.work_size)
        y0, x0 = geom.pad_top, geom.pad_left
        cropped = square[y0 : y0 + geom.scaled_h, x0 : x0 + geom.scaled_w]
        crop_rgb = _resize_rgb(cropped, geom.crop_w, geom.crop_h)
        filled = original.copy()
        y, x = geom.crop_y0, geom.crop_x0
        filled[y : y + geom.crop_h, x : x + geom.crop_w] = crop_rgb

    return _feathered_composite(original, filled, mask, geom)


def try_uniform_neighbor_fill(
    image: np.ndarray, mask: np.ndarray
) -> np.ndarray | None:
    """If pixels around the mask are almost one color (white paper), fill with that color.

    Returns ``None`` when neighbors vary (photos, marble, scenery) so Moebius still runs.
    """
    image = validate_image(image)
    mask = validate_mask(mask, image=image)
    if not bool(mask.any()):
        return None
    band = _dilate_mask(mask, UNIFORM_NEIGHBOR_BAND_PX) & np.logical_not(mask)
    if int(band.sum()) < UNIFORM_MIN_SAMPLES:
        return None
    samples = image[band].reshape(-1, 3).astype(np.float32)
    median = np.median(samples, axis=0)
    dist = np.max(np.abs(samples - median), axis=1)
    near = dist <= UNIFORM_NEAR_MAX
    if float(near.mean()) < UNIFORM_NEAR_FRACTION:
        return None
    near_samples = samples[near]
    if near_samples.shape[0] < UNIFORM_MIN_SAMPLES:
        return None
    color = np.rint(np.median(near_samples, axis=0)).clip(0, 255).astype(np.uint8)
    hole = _dilate_mask(mask, UNIFORM_FILL_DILATE_PX)
    solid = np.empty_like(image)
    solid[:, :] = color
    alpha_img = Image.fromarray((hole.astype(np.uint8) * 255), mode="L")
    alpha = np.asarray(
        alpha_img.filter(ImageFilter.GaussianBlur(radius=UNIFORM_FILL_FEATHER_PX)),
        dtype=np.float32,
    ) / 255.0
    alpha = alpha[..., np.newaxis]
    blended = image.astype(np.float32) * (1.0 - alpha) + solid.astype(np.float32) * alpha
    return np.clip(np.rint(blended), 0, 255).astype(np.uint8)


def composite_halo_px(geom: MoebiusWorkGeometry) -> int:
    """Original-pixel dilation covering the object plus its contact shadow."""
    return _halo_from_hw(
        geom.crop_h, geom.crop_w, orig_h=geom.orig_h, orig_w=geom.orig_w
    )


def _halo_from_hw(
    crop_h: int,
    crop_w: int,
    *,
    orig_h: int = 0,
    orig_w: int = 0,
) -> int:
    crop_term = int(round(COMPOSITE_HALO_RATIO * max(crop_h, crop_w)))
    photo_term = (
        int(round(COMPOSITE_PHOTO_HALO_RATIO * max(orig_h, orig_w)))
        if orig_h > 0 and orig_w > 0
        else 0
    )
    halo = max(COMPOSITE_HALO_MIN, crop_term, photo_term)
    return min(COMPOSITE_HALO_MAX, halo)


def _crop_scale(geom: MoebiusWorkGeometry) -> float:
    if geom.scaled_h <= 0 or geom.scaled_w <= 0:
        return 1.0
    return max(geom.crop_h / geom.scaled_h, geom.crop_w / geom.scaled_w, 1.0)


def _work_dilate_iterations(geom: MoebiusWorkGeometry) -> int:
    halo = composite_halo_px(geom)
    scale = _crop_scale(geom)
    work_px = int(round(halo / scale)) if scale > 0 else halo
    return max(WORK_MASK_DILATE, work_px)


def _feathered_composite(
    original: np.ndarray,
    filled: np.ndarray,
    mask: np.ndarray,
    geom: MoebiusWorkGeometry,
) -> np.ndarray:
    halo = composite_halo_px(geom)
    dilated = _dilate_mask(mask, halo)
    alpha_img = Image.fromarray((dilated.astype(np.uint8) * 255), mode="L")
    alpha = np.asarray(
        alpha_img.filter(ImageFilter.GaussianBlur(radius=COMPOSITE_FEATHER_PX)),
        dtype=np.float32,
    ) / 255.0
    alpha = alpha[..., np.newaxis]
    blended = original.astype(np.float32) * (1.0 - alpha) + filled.astype(np.float32) * alpha
    return np.clip(np.rint(blended), 0, 255).astype(np.uint8)


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        h, w = mask.shape
        return 0, 0, h, w
    return int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1


def _expand_bbox(
    y0: int,
    x0: int,
    y1: int,
    x1: int,
    height: int,
    width: int,
    *,
    ratio: float = CROP_MARGIN_RATIO,
    min_px: int = CROP_MARGIN_MIN_PX,
) -> tuple[int, int, int, int]:
    box_h = y1 - y0
    box_w = x1 - x0
    margin_y = max(min_px, int(round(box_h * ratio)))
    margin_x = max(min_px, int(round(box_w * ratio)))
    tent_h = min(height, box_h + 2 * margin_y)
    tent_w = min(width, box_w + 2 * margin_x)
    extra = _halo_from_hw(
        tent_h, tent_w, orig_h=height, orig_w=width
    ) + COMPOSITE_FEATHER_PX
    margin_y = max(margin_y, extra)
    margin_x = max(margin_x, extra)
    return (
        max(0, y0 - margin_y),
        max(0, x0 - margin_x),
        min(height, y1 + margin_y),
        min(width, x1 + margin_x),
    )


def _max_pool_mask(mask: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Downscale so every source True maps to a dest True (NEAREST can drop pixels)."""
    src_h, src_w = mask.shape
    if (src_h, src_w) == (out_h, out_w):
        return mask.astype(bool, copy=False)
    out = np.zeros((out_h, out_w), dtype=bool)
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return out
    dy = np.minimum((ys * out_h) // src_h, out_h - 1)
    dx = np.minimum((xs * out_w) // src_w, out_w - 1)
    out[dy, dx] = True
    return out


def _dilate_mask(mask: np.ndarray, iterations: int) -> np.ndarray:
    out = mask.astype(bool, copy=True)
    if iterations <= 0:
        return out
    for _ in range(iterations):
        padded = np.pad(out, 1, mode="constant", constant_values=False)
        grown = np.zeros_like(out)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                grown |= padded[1 + dy : 1 + dy + out.shape[0], 1 + dx : 1 + dx + out.shape[1]]
        out = grown
    return out


def _pad_to_square(
    rgb: np.ndarray, mask: np.ndarray, size: int
) -> tuple[np.ndarray, np.ndarray, int, int]:
    height, width = rgb.shape[:2]
    pad_top = (size - height) // 2
    pad_left = (size - width) // 2
    work_rgb = np.zeros((size, size, 3), dtype=np.uint8)
    work_mask = np.zeros((size, size), dtype=bool)
    work_rgb[pad_top : pad_top + height, pad_left : pad_left + width] = rgb
    work_mask[pad_top : pad_top + height, pad_left : pad_left + width] = mask
    return work_rgb, work_mask, pad_top, pad_left


def _resize_rgb(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if image.shape[0] == height and image.shape[1] == width:
        return image
    resized = Image.fromarray(image, mode="RGB").resize(
        (width, height), resample=Image.Resampling.LANCZOS
    )
    return np.asarray(resized, dtype=np.uint8)
