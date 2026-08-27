"""Shared request validation helpers for the FastAPI layer."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image, UnidentifiedImageError

from apps.backend.errors import InvalidInputError
from models.types import pil_rgb_to_array, validate_image, validate_mask

# Upload bounds (documented for API clients).
MIN_IMAGE_DIMENSION = 8
MAX_IMAGE_PIXELS = 16_777_216  # 4096×4096
MASK_INPAINT_THRESHOLD = 128


def normalize_upload_image(raw: bytes, *, content_type: str | None = None) -> np.ndarray:
    """Decode bytes to H×W×3 uint8 RGB. Supports RGB, RGBA, grayscale, palette."""
    if not raw:
        raise InvalidInputError("Image upload is empty.")
    if content_type and not content_type.startswith("image/"):
        raise InvalidInputError(f"Expected an image upload, got {content_type}.")

    try:
        pil = Image.open(io.BytesIO(raw))
        pil.load()
    except UnidentifiedImageError as exc:
        raise InvalidInputError("Unsupported or unrecognized image format.") from exc
    except Exception as exc:
        raise InvalidInputError("Could not decode image upload.") from exc

    if pil.width < MIN_IMAGE_DIMENSION or pil.height < MIN_IMAGE_DIMENSION:
        raise InvalidInputError(
            f"Image is too small ({pil.width}×{pil.height}). "
            f"Minimum dimension is {MIN_IMAGE_DIMENSION}px."
        )
    if pil.width * pil.height > MAX_IMAGE_PIXELS:
        raise InvalidInputError("Image exceeds the maximum allowed pixel count.")

    return pil_rgb_to_array(pil)


def decode_mask_bytes(raw: bytes, *, image: np.ndarray | None = None) -> np.ndarray:
    """Decode mask PNG bytes to bool H×W (white/255 = inpaint)."""
    if not raw:
        raise InvalidInputError("Mask upload is empty.")
    try:
        pil = Image.open(io.BytesIO(raw))
        pil.load()
    except UnidentifiedImageError as exc:
        raise InvalidInputError("Unsupported or unrecognized mask format.") from exc
    except Exception as exc:
        raise InvalidInputError("Could not decode mask upload.") from exc

    gray = np.asarray(pil.convert("L"))
    if gray.ndim != 2:
        raise InvalidInputError("Mask must decode to a single-channel image.")
    mask = gray >= MASK_INPAINT_THRESHOLD
    if image is not None:
        validate_mask(mask, image=image)
    else:
        validate_mask(mask)
    return mask


def require_nonempty_mask(mask: np.ndarray, *, context: str = "inpaint") -> np.ndarray:
    """Reject masks with no inpaint pixels."""
    mask = validate_mask(mask)
    if not bool(mask.any()):
        raise InvalidInputError(f"Mask has no inpaint region for {context}.")
    return mask


def validate_point(image: np.ndarray, x: int, y: int) -> None:
    """Ensure integer point lies within image bounds."""
    image = validate_image(image)
    h, w = image.shape[:2]
    if not isinstance(x, int) or not isinstance(y, int):
        raise InvalidInputError("Point coordinates must be integers.")
    if not (0 <= x < w and 0 <= y < h):
        raise InvalidInputError(
            f"Point ({x}, {y}) is outside image bounds ({w}×{h})."
        )


def validate_inpaint_params_fields(
    *,
    num_steps: int | None,
    guidance_scale: float | None,
    strength: float | None,
    noise_offset: float | None,
    image_size: int | None,
    seed: int | None = None,
) -> None:
    if num_steps is not None and not (1 <= num_steps <= 200):
        raise InvalidInputError("num_steps must be between 1 and 200.")
    if guidance_scale is not None and not (0.0 <= guidance_scale <= 30.0):
        raise InvalidInputError("guidance_scale must be between 0 and 30.")
    if strength is not None and not (0.0 <= strength <= 1.0):
        raise InvalidInputError("strength must be between 0 and 1.")
    if noise_offset is not None and not (0.0 <= noise_offset <= 1.0):
        raise InvalidInputError("noise_offset must be between 0 and 1.")
    if image_size is not None and not (64 <= image_size <= 4096):
        raise InvalidInputError("image_size must be between 64 and 4096.")
    if seed is not None and not (0 <= seed <= 2**31 - 1):
        raise InvalidInputError("seed must be a non-negative 31-bit integer.")


def validate_candidate_count_field(candidate_count: int | None) -> int:
    """Validate optional candidate_count form field (defaults to 1)."""
    if candidate_count is None:
        return 1
    try:
        from pipelines.candidate_seeds import validate_candidate_count

        return validate_candidate_count(candidate_count)
    except ValueError as exc:
        raise InvalidInputError(str(exc)) from exc
