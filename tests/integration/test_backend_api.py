"""FastAPI integration tests with mocked pipeline (no real inference)."""

from __future__ import annotations

import io
import unittest
from unittest.mock import MagicMock

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from apps.backend.dependencies import get_editing_service
from apps.backend.main import create_app
from apps.backend.services import ImageEditingService
from models.types import (
    BackendType,
    InpaintingResult,
    SegmentationResult,
    validate_image,
    validate_mask,
)
from pipelines.errors import PipelineBackendError, PipelineValidationError
from pipelines.types import ImageEditPipelineResult, PipelineLatency


def _png_bytes(h: int = 32, w: int = 32) -> bytes:
    img = Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _photo_png(h: int = 32, w: int = 32) -> bytes:
    yy, xx = np.ogrid[:h, :w]
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[..., 0] = (xx * 7 + yy * 3) % 256
    arr[..., 1] = (xx * 5 + yy * 11) % 256
    arr[..., 2] = (xx * 13 + yy * 2) % 256
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _mask_png(h: int = 32, w: int = 32) -> bytes:
    arr = np.zeros((h, w), dtype=np.uint8)
    arr[8:24, 8:24] = 255
    img = Image.fromarray(arr, mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


class TestBackendAPI(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = MagicMock()
        self.service = ImageEditingService(pipeline=self.pipeline)
        self.app = create_app()
        self.app.dependency_overrides[get_editing_service] = lambda: self.service
        self.client = TestClient(self.app)

    def tearDown(self) -> None:
        self.app.dependency_overrides.clear()

    def test_health(self) -> None:
        resp = self.client.get("/health")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["version"], "0.1.0")
        self.assertIn("max_upload_bytes", body)
        self.assertIn("max_concurrent_generations", body)

    def test_models(self) -> None:
        resp = self.client.get("/models")
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIn("models", body)
        self.assertGreaterEqual(len(body["models"]), 3)

    def test_invalid_image(self) -> None:
        resp = self.client.post(
            "/segment",
            files={"image": ("bad.txt", b"not-an-image", "text/plain")},
            data={"x": "5", "y": "5"},
        )
        self.assertEqual(resp.status_code, 400)

    def test_invalid_point(self) -> None:
        self.pipeline.segment.side_effect = PipelineValidationError("Point out of bounds")
        resp = self.client.post(
            "/segment",
            files={"image": ("img.png", _png_bytes(), "image/png")},
            data={"x": "999", "y": "999"},
        )
        self.assertEqual(resp.status_code, 400)

    def test_invalid_mask_dimensions(self) -> None:
        resp = self.client.post(
            "/inpaint",
            files={
                "image": ("img.png", _png_bytes(32, 32), "image/png"),
                "mask": ("mask.png", _mask_png(16, 16), "image/png"),
            },
            data={"backend": "moebius"},
        )
        self.assertEqual(resp.status_code, 400)

    def test_unsupported_backend(self) -> None:
        self.pipeline.inpaint.side_effect = PipelineBackendError("Unsupported backend")
        resp = self.client.post(
            "/inpaint",
            files={
                "image": ("img.png", _photo_png(), "image/png"),
                "mask": ("mask.png", _mask_png(), "image/png"),
            },
            data={"backend": "unknown"},
        )
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.json()["error"]["code"], "unsupported_backend")

    def test_segment_success_mocked(self) -> None:
        mask = validate_mask(np.zeros((32, 32), dtype=bool))
        mask[10:20, 10:20] = True
        self.pipeline.segment.return_value = SegmentationResult(
            mask=mask,
            confidence=0.95,
            model="Fake SAM",
            method="point",
            metadata={"latency_ms": 3.0},
        )
        resp = self.client.post(
            "/segment",
            files={"image": ("img.png", _png_bytes(), "image/png")},
            data={"x": "16", "y": "16"},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers["content-type"], "image/png")
        self.assertIn("x-pf-model", resp.headers)
        self.assertEqual(resp.headers["x-pf-confidence"], "0.95")
        self.pipeline.segment.assert_called_once()

    def test_inpaint_success_mocked(self) -> None:
        result_img = validate_image(np.full((32, 32, 3), 128, dtype=np.uint8))
        self.pipeline.inpaint.return_value = InpaintingResult(
            result=result_img,
            latency_ms=21.0,
            memory_mb=50.0,
            model="Fake Moebius",
            backend=BackendType.LOCAL_MPS,
            metadata={},
        )
        resp = self.client.post(
            "/inpaint",
            files={
                "image": ("img.png", _photo_png(), "image/png"),
                "mask": ("mask.png", _mask_png(), "image/png"),
            },
            data={"backend": "moebius"},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers["content-type"], "image/png")
        self.assertEqual(resp.headers["x-pf-latency-ms"], "21.0")
        self.pipeline.inpaint.assert_called_once()

    def test_remove_object_success_mocked(self) -> None:
        result_img = validate_image(np.full((32, 32, 3), 64, dtype=np.uint8))
        mask = validate_mask(np.zeros((32, 32), dtype=bool))
        mask[8:24, 8:24] = True
        self.pipeline.remove_object.return_value = ImageEditPipelineResult(
            result=result_img,
            mask=mask,
            segmentation=SegmentationResult(
                mask=mask,
                confidence=0.9,
                model="Fake SAM",
                method="point",
                metadata={},
            ),
            inpainting=InpaintingResult(
                result=result_img,
                latency_ms=30.0,
                memory_mb=None,
                model="Fake Moebius",
                backend=BackendType.LOCAL_MPS,
                metadata={},
            ),
            selected_model="Fake Moebius",
            backend=BackendType.LOCAL_MPS,
            latency=PipelineLatency(
                segmentation_ms=5.0,
                inpainting_ms=30.0,
                total_ms=40.0,
            ),
            metadata={"prompt_xy": [16, 16]},
        )
        resp = self.client.post(
            "/remove-object",
            files={"image": ("img.png", _png_bytes(), "image/png")},
            data={"x": "16", "y": "16", "backend": "moebius"},
        )
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers["content-type"], "image/png")
        self.assertEqual(resp.headers["x-pf-latency-ms"], "40.0")
        self.pipeline.remove_object.assert_called_once()


if __name__ == "__main__":
    unittest.main()
