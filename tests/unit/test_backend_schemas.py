"""Unit tests for backend Pydantic schemas and media helpers."""

from __future__ import annotations

import io
import unittest

import numpy as np
from PIL import Image

from apps.backend.schemas import HealthResponse, ModelsResponse, SegmentMetadata
from apps.backend.services import (
    decode_upload_mask,
    image_to_png_bytes,
    mask_to_png_bytes,
    parse_inpaint_params,
)
from models.types import validate_image, validate_mask


class TestSchemas(unittest.TestCase):
    def test_health_response(self) -> None:
        payload = HealthResponse()
        self.assertEqual(payload.status, "ok")

    def test_models_response_shape(self) -> None:
        payload = ModelsResponse(
            models=[
                {
                    "id": "sam2",
                    "name": "SAM",
                    "role": "segmentation",
                    "backend": "LOCAL_MPS",
                    "available": True,
                    "loaded": False,
                    "status": "UNAVAILABLE",
                }
            ]
        )
        self.assertEqual(payload.models[0].id, "sam2")

    def test_segment_metadata(self) -> None:
        meta = SegmentMetadata(
            confidence=0.9,
            model="SAM",
            backend="LOCAL_MPS",
            method="point",
            metadata={"latency_ms": 1.0},
        )
        self.assertEqual(meta.confidence, 0.9)


class TestMediaHelpers(unittest.IsolatedAsyncioTestCase):
    async def test_mask_png_contract_white_inpaint(self) -> None:
        arr = np.zeros((8, 8), dtype=np.uint8)
        arr[2:6, 2:6] = 255
        buf = io.BytesIO()
        Image.fromarray(arr, mode="L").save(buf, format="PNG")
        buf.seek(0)

        class _Upload:
            async def read(self) -> bytes:
                return buf.getvalue()

        mask = await decode_upload_mask(_Upload())
        self.assertEqual(mask.shape, (8, 8))
        self.assertTrue(mask[3, 3])
        self.assertFalse(mask[0, 0])

    def test_mask_roundtrip_png(self) -> None:
        mask = np.zeros((16, 16), dtype=bool)
        mask[4:12, 4:12] = True
        png = mask_to_png_bytes(mask)
        decoded = np.asarray(Image.open(io.BytesIO(png)).convert("L"))
        self.assertTrue((decoded[mask] == 255).all())

    def test_image_png_roundtrip(self) -> None:
        img = validate_image(np.zeros((4, 4, 3), dtype=np.uint8))
        png = image_to_png_bytes(img)
        out = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"))
        self.assertEqual(out.shape, (4, 4, 3))

    def test_parse_inpaint_params_none_when_empty(self) -> None:
        self.assertIsNone(parse_inpaint_params())

    def test_parse_inpaint_params_partial(self) -> None:
        params = parse_inpaint_params(num_steps=10)
        self.assertIsNotNone(params)
        assert params is not None
        self.assertEqual(params.num_steps, 10)


if __name__ == "__main__":
    unittest.main()
