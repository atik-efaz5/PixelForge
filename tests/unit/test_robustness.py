"""Unit tests for upload validation and API hardening helpers."""

from __future__ import annotations

import io
import unittest

import numpy as np
from PIL import Image

from apps.backend.errors import error_body
from apps.backend.validation import (
    MIN_IMAGE_DIMENSION,
    decode_mask_bytes,
    normalize_upload_image,
    require_nonempty_mask,
    validate_inpaint_params_fields,
    validate_point,
)
from apps.backend.errors import InvalidInputError
from models.types import validate_image


def _png_bytes(mode: str, size: tuple[int, int], color=0) -> bytes:
    buf = io.BytesIO()
    Image.new(mode, size, color=color).save(buf, format="PNG")
    return buf.getvalue()


class TestNormalizeUploadImage(unittest.TestCase):
    def test_rejects_empty_bytes(self) -> None:
        with self.assertRaises(InvalidInputError) as ctx:
            normalize_upload_image(b"")
        self.assertIn("empty", str(ctx.exception).lower())

    def test_rejects_corrupted_bytes(self) -> None:
        with self.assertRaises(InvalidInputError):
            normalize_upload_image(b"not-an-image")

    def test_accepts_rgba_and_returns_rgb(self) -> None:
        raw = _png_bytes("RGBA", (16, 16), color=(255, 0, 0, 128))
        arr = normalize_upload_image(raw)
        self.assertEqual(arr.shape, (16, 16, 3))
        self.assertEqual(arr.dtype, np.uint8)

    def test_accepts_grayscale(self) -> None:
        raw = _png_bytes("L", (16, 16), color=128)
        arr = normalize_upload_image(raw)
        self.assertEqual(arr.shape, (16, 16, 3))

    def test_rejects_too_small(self) -> None:
        raw = _png_bytes("RGB", (4, 4))
        with self.assertRaises(InvalidInputError) as ctx:
            normalize_upload_image(raw)
        self.assertIn(str(MIN_IMAGE_DIMENSION), str(ctx.exception))

    def test_rejects_bad_content_type(self) -> None:
        raw = _png_bytes("RGB", (16, 16))
        with self.assertRaises(InvalidInputError):
            normalize_upload_image(raw, content_type="text/plain")

    def test_wide_aspect_ratio_ok(self) -> None:
        raw = _png_bytes("RGB", (64, 8))
        arr = normalize_upload_image(raw)
        self.assertEqual(arr.shape[:2], (8, 64))


class TestMaskValidation(unittest.TestCase):
    def test_rejects_empty_mask_upload(self) -> None:
        with self.assertRaises(InvalidInputError):
            decode_mask_bytes(b"")

    def test_rejects_dimension_mismatch(self) -> None:
        image = validate_image(np.zeros((16, 16, 3), dtype=np.uint8))
        mask_png = _png_bytes("L", (8, 8))
        with self.assertRaises(InvalidInputError):
            decode_mask_bytes(mask_png, image=image)

    def test_rejects_empty_inpaint_region(self) -> None:
        mask = np.zeros((8, 8), dtype=bool)
        with self.assertRaises(InvalidInputError):
            require_nonempty_mask(mask)

    def test_mask_png_roundtrip_lossless(self) -> None:
        mask = np.zeros((16, 16), dtype=bool)
        mask[4:12, 4:12] = True
        buf = io.BytesIO()
        Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(buf, format="PNG")
        decoded = decode_mask_bytes(buf.getvalue())
        np.testing.assert_array_equal(decoded, mask)


class TestPointValidation(unittest.TestCase):
    def test_rejects_out_of_bounds(self) -> None:
        image = validate_image(np.zeros((32, 32, 3), dtype=np.uint8))
        with self.assertRaises(InvalidInputError):
            validate_point(image, 32, 0)
        with self.assertRaises(InvalidInputError):
            validate_point(image, -1, 0)

    def test_accepts_valid_point(self) -> None:
        image = validate_image(np.zeros((32, 32, 3), dtype=np.uint8))
        validate_point(image, 0, 0)
        validate_point(image, 31, 31)


class TestInpaintParamValidation(unittest.TestCase):
    def test_rejects_invalid_num_steps(self) -> None:
        with self.assertRaises(InvalidInputError):
            validate_inpaint_params_fields(
                num_steps=0,
                guidance_scale=None,
                strength=None,
                noise_offset=None,
                image_size=None,
            )

    def test_rejects_invalid_strength(self) -> None:
        with self.assertRaises(InvalidInputError):
            validate_inpaint_params_fields(
                num_steps=None,
                guidance_scale=None,
                strength=1.5,
                noise_offset=None,
                image_size=None,
            )


class TestErrorBody(unittest.TestCase):
    def test_nested_and_legacy_fields(self) -> None:
        body = error_body("invalid_input", "bad upload")
        self.assertEqual(body["error"]["code"], "invalid_input")
        self.assertEqual(body["error"]["message"], "bad upload")
        self.assertEqual(body["code"], "invalid_input")
        self.assertEqual(body["message"], "bad upload")


if __name__ == "__main__":
    unittest.main()
