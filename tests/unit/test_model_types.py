"""Contracts for images, masks, backends, and result records."""

from __future__ import annotations

import unittest

import numpy as np

from models.types import (
    BackendType,
    InpaintingResult,
    ModelStatus,
    SegmentationResult,
    validate_image,
    validate_mask,
)


class TestModelTypes(unittest.TestCase):
    def test_backend_enum_values(self) -> None:
        self.assertEqual(
            {m.value for m in BackendType},
            {"LOCAL_MPS", "CLOUD_GPU", "CPU", "UNAVAILABLE"},
        )

    def test_model_status_enum_values(self) -> None:
        self.assertEqual(
            {m.value for m in ModelStatus},
            {"READY", "UNAVAILABLE", "LOADING", "ERROR"},
        )

    def test_valid_image_contract(self) -> None:
        image = np.zeros((8, 12, 3), dtype=np.uint8)
        out = validate_image(image)
        self.assertEqual(out.shape, (8, 12, 3))
        self.assertEqual(out.dtype, np.uint8)
        self.assertIs(out, image)

    def test_image_rejects_wrong_rank(self) -> None:
        with self.assertRaisesRegex(ValueError, "H×W×3"):
            validate_image(np.zeros((8, 12), dtype=np.uint8))

    def test_image_rejects_wrong_dtype(self) -> None:
        with self.assertRaisesRegex(ValueError, "uint8"):
            validate_image(np.zeros((8, 12, 3), dtype=np.float32))

    def test_valid_mask_contract(self) -> None:
        mask = np.zeros((8, 12), dtype=bool)
        mask[2, 3] = True
        out = validate_mask(mask)
        self.assertEqual(out.shape, (8, 12))
        self.assertEqual(out.dtype, np.bool_)
        self.assertTrue(bool(out[2, 3]))

    def test_mask_must_match_image_hw(self) -> None:
        image = np.zeros((8, 12, 3), dtype=np.uint8)
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_mask(np.zeros((4, 4), dtype=bool), image=image)

    def test_mask_rejects_uint8(self) -> None:
        with self.assertRaisesRegex(ValueError, "bool"):
            validate_mask(np.zeros((8, 12), dtype=np.uint8))

    def test_segmentation_result_validates_mask(self) -> None:
        mask = np.zeros((4, 4), dtype=bool)
        result = SegmentationResult(
            mask=mask,
            confidence=0.9,
            model="SAM 2.1 Hiera-Tiny",
            method="point",
        )
        self.assertEqual(result.mask.dtype, np.bool_)
        self.assertEqual(result.confidence, 0.9)

    def test_inpainting_result_validates_image(self) -> None:
        rgb = np.zeros((4, 4, 3), dtype=np.uint8)
        result = InpaintingResult(
            result=rgb,
            latency_ms=12.5,
            memory_mb=None,
            model="Moebius",
            backend=BackendType.LOCAL_MPS,
        )
        self.assertEqual(result.result.dtype, np.uint8)
        self.assertIs(result.backend, BackendType.LOCAL_MPS)


if __name__ == "__main__":
    unittest.main()
