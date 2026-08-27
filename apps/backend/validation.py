"""Shared request validation helpers for the FastAPI layer."""

from __future__ import annotations

import io
import re
from pathlib import PurePosixPath

import numpy as np
from PIL import Image, UnidentifiedImageError

from apps.backend.errors import InvalidInputError
from apps.backend.settings import get_settings
from models.types import pil_rgb_to_array, validate_image, validate_mask

# Re-export bounds from settings for backward-compatible imports/tests.
_settings = get_settings()
MIN_IMAGE_DIMENSION = _settings.min_image_dimension
MAX_IMAGE_PIXELS = _settings.max_image_pixels
MAX_IMAGE_DIMENSION = _settings.max_image_dimension
MASK_INPAINT_THRESHOLD = 128

_IMAGE_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"RIFF", "image/webp"),  # WebP: RIFF....WEBP
)
_UNSAFE_FILENAME = re.compile(r"[^\w.\-]+")


def validate_upload_bytes(raw: bytes, *, max_bytes: int, label: str = "upload") -> bytes:
    """Reject empty or oversized raw uploads before decode."""
    if not raw:
        raise InvalidInputError(f"{label} is empty.")
    if len(raw) > max_bytes:
        raise InvalidInputError(
            f"{label} exceeds maximum size ({max_bytes} bytes)."
        )
    return raw


def sanitize_filename(name: str | None) -> str:
    """Strip path components and unsafe characters from upload filenames."""
    if not name:
        return "upload"
    base = PurePosixPath(name.replace("\\", "/")).name
    cleaned = _UNSAFE_FILENAME.sub("_", base).strip("._")
    return cleaned or "upload"


def verify_image_magic(raw: bytes) -> None:
    """Basic content sniffing for supported image formats."""
    if raw.startswith(_IMAGE_MAGIC[0][0]):
        return
    if raw.startswith(_IMAGE_MAGIC[1][0]):
        return
    if len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return
    raise InvalidInputError(
        "Unsupported image content. Upload PNG, JPEG, or WebP."
    )


def verify_mask_magic(raw: bytes) -> None:
    """Mask uploads must be PNG (lossless single-channel)."""
    if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
        raise InvalidInputError("Mask must be a PNG image.")


def validate_text_field(
    value: str | None,
    *,
    field_name: str,
    max_length: int,
    required: bool = True,
) -> str:
    text = (value or "").strip()
    if not text:
        if required:
            raise InvalidInputError(f"{field_name} is empty.")
        return ""
    if len(text) > max_length:
        raise InvalidInputError(
            f"{field_name} exceeds maximum length ({max_length} characters)."
        )
    return text


def normalize_upload_image(raw: bytes, *, content_type: str | None = None) -> np.ndarray:
    """Decode bytes to H×W×3 uint8 RGB. Supports RGB, RGBA, grayscale, palette."""
    settings = get_settings()
    validate_upload_bytes(raw, max_bytes=settings.max_upload_bytes, label="Image upload")
    if content_type and not content_type.startswith("image/"):
        raise InvalidInputError(f"Expected an image upload, got {content_type}.")
    verify_image_magic(raw)

    # Decompression-bomb guard (PIL pixel cap).
    Image.MAX_IMAGE_PIXELS = settings.max_image_pixels

    try:
        pil = Image.open(io.BytesIO(raw))
        pil.load()
    except UnidentifiedImageError as exc:
        raise InvalidInputError("Unsupported or unrecognized image format.") from exc
    except Image.DecompressionBombError as exc:
        raise InvalidInputError("Image exceeds the maximum allowed pixel count.") from exc
    except Exception as exc:
        raise InvalidInputError("Could not decode image upload.") from exc

    if pil.width < settings.min_image_dimension or pil.height < settings.min_image_dimension:
        raise InvalidInputError(
            f"Image is too small ({pil.width}×{pil.height}). "
            f"Minimum dimension is {settings.min_image_dimension}px."
        )
    if pil.width > settings.max_image_dimension or pil.height > settings.max_image_dimension:
        raise InvalidInputError(
            f"Image exceeds maximum dimension ({settings.max_image_dimension}px)."
        )
    if pil.width * pil.height > settings.max_image_pixels:
        raise InvalidInputError("Image exceeds the maximum allowed pixel count.")

    return pil_rgb_to_array(pil)


def decode_mask_bytes(raw: bytes, *, image: np.ndarray | None = None) -> np.ndarray:
    """Decode mask PNG bytes to bool H×W (white/255 = inpaint)."""
    settings = get_settings()
    validate_upload_bytes(raw, max_bytes=settings.max_mask_upload_bytes, label="Mask upload")
    verify_mask_magic(raw)
    Image.MAX_IMAGE_PIXELS = settings.max_image_pixels

    try:
        pil = Image.open(io.BytesIO(raw))
        pil.load()
    except UnidentifiedImageError as exc:
        raise InvalidInputError("Unsupported or unrecognized mask format.") from exc
    except Image.DecompressionBombError as exc:
        raise InvalidInputError("Mask exceeds the maximum allowed pixel count.") from exc
    except Exception as exc:
        raise InvalidInputError("Could not decode mask upload.") from exc

    gray = np.asarray(pil.convert("L"))
    if gray.ndim != 2:
        raise InvalidInputError("Mask must decode to a single-channel image.")
    if gray.shape[0] > settings.max_image_dimension or gray.shape[1] > settings.max_image_dimension:
        raise InvalidInputError(
            f"Mask exceeds maximum dimension ({settings.max_image_dimension}px)."
        )
    mask = gray >= MASK_INPAINT_THRESHOLD
    try:
        if image is not None:
            validate_mask(mask, image=image)
        else:
            validate_mask(mask)
    except ValueError as exc:
        raise InvalidInputError(str(exc)) from exc
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


def validate_morph_amount(value: int, *, field_name: str) -> int:
    settings = get_settings()
    if not isinstance(value, int):
        raise InvalidInputError(f"{field_name} must be an integer.")
    if value < 0 or value > settings.max_morph_pixels:
        raise InvalidInputError(
            f"{field_name} must be between 0 and {settings.max_morph_pixels}."
        )
    return value


def validate_inpaint_params_fields(
    *,
    num_steps: int | None,
    guidance_scale: float | None,
    strength: float | None,
    noise_offset: float | None,
    image_size: int | None,
    seed: int | None = None,
) -> None:
    settings = get_settings()
    if num_steps is not None and not (1 <= num_steps <= 200):
        raise InvalidInputError("num_steps must be between 1 and 200.")
    if guidance_scale is not None and not (0.0 <= guidance_scale <= 30.0):
        raise InvalidInputError("guidance_scale must be between 0 and 30.")
    if strength is not None and not (0.0 <= strength <= 1.0):
        raise InvalidInputError("strength must be between 0 and 1.")
    if noise_offset is not None and not (0.0 <= noise_offset <= 1.0):
        raise InvalidInputError("noise_offset must be between 0 and 1.")
    if image_size is not None and not (
        settings.min_image_dimension <= image_size <= settings.max_image_dimension
    ):
        raise InvalidInputError(
            f"image_size must be between {settings.min_image_dimension} "
            f"and {settings.max_image_dimension}."
        )
    if seed is not None and not (0 <= seed <= 2**31 - 1):
        raise InvalidInputError("seed must be a non-negative 31-bit integer.")


def validate_instruction_edit_params_fields(
    *,
    num_steps: int | None,
    guidance_text: float | None,
    guidance_image: float | None,
    resolution: int | None,
) -> None:
    settings = get_settings()
    if num_steps is not None and not (1 <= num_steps <= 200):
        raise InvalidInputError("num_steps must be between 1 and 200.")
    if guidance_text is not None and not (0.0 <= guidance_text <= 30.0):
        raise InvalidInputError("guidance_text must be between 0 and 30.")
    if guidance_image is not None and not (0.0 <= guidance_image <= 30.0):
        raise InvalidInputError("guidance_image must be between 0 and 30.")
    if resolution is not None and not (
        settings.min_image_dimension <= resolution <= settings.max_image_dimension
    ):
        raise InvalidInputError(
            f"resolution must be between {settings.min_image_dimension} "
            f"and {settings.max_image_dimension}."
        )


def validate_candidate_count_field(candidate_count: int | None) -> int:
    """Validate optional candidate_count form field (defaults to 1)."""
    settings = get_settings()
    if candidate_count is None:
        return 1
    try:
        from pipelines.candidate_seeds import validate_candidate_count

        count = validate_candidate_count(candidate_count)
    except ValueError as exc:
        raise InvalidInputError(str(exc)) from exc
    if count > settings.max_candidate_count:
        raise InvalidInputError(
            f"candidate_count cannot exceed {settings.max_candidate_count}."
        )
    return count


def validate_detection_index(detection_index: int, *, detection_count: int | None = None) -> int:
    if not isinstance(detection_index, int):
        raise InvalidInputError("detection_index must be an integer.")
    if detection_index < 0:
        raise InvalidInputError("detection_index must be non-negative.")
    if detection_count is not None and detection_index >= detection_count:
        raise InvalidInputError(
            f"detection_index {detection_index} is out of range."
        )
    return detection_index
