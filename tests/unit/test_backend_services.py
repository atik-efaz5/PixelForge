"""Unit tests for ImageEditingService delegation."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import numpy as np

from apps.backend.services import ImageEditingService
from models.types import (
    BackendType,
    InpaintingResult,
    SegmentationResult,
    validate_image,
    validate_mask,
)
from pipelines.types import ImageEditPipelineResult, MaskRefinementOps, PipelineLatency


def _rgb() -> np.ndarray:
    return validate_image(np.zeros((32, 32, 3), dtype=np.uint8))


def _mask() -> np.ndarray:
    m = np.zeros((32, 32), dtype=bool)
    m[8:24, 8:24] = True
    return validate_mask(m)


class TestImageEditingService(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = MagicMock()
        self.service = ImageEditingService(pipeline=self.pipeline)

    def test_segment_delegates_to_pipeline(self) -> None:
        expected = SegmentationResult(
            mask=_mask(),
            confidence=0.8,
            model="SAM",
            method="point",
            metadata={},
        )
        self.pipeline.segment.return_value = expected
        out = self.service.segment(_rgb(), 10, 12)
        self.pipeline.segment.assert_called_once()
        self.assertEqual(out.model, "SAM")

    def test_inpaint_delegates_to_pipeline(self) -> None:
        expected = InpaintingResult(
            result=_rgb(),
            latency_ms=5.0,
            memory_mb=None,
            model="Moebius",
            backend=BackendType.LOCAL_MPS,
            metadata={},
        )
        self.pipeline.inpaint.return_value = expected
        out = self.service.inpaint(_rgb(), _mask(), backend="moebius")
        self.pipeline.inpaint.assert_called_once()
        self.assertEqual(out.model, "Moebius")

    def test_remove_object_delegates_to_pipeline(self) -> None:
        expected = ImageEditPipelineResult(
            result=_rgb(),
            mask=_mask(),
            segmentation=None,
            inpainting=None,
            selected_model="Moebius",
            backend=BackendType.LOCAL_MPS,
            latency=PipelineLatency(total_ms=10.0),
            metadata={},
        )
        self.pipeline.remove_object.return_value = expected
        refinement = MaskRefinementOps(dilate=1)
        out = self.service.remove_object(
            _rgb(), 5, 6, backend="moebius", refinement=refinement
        )
        self.pipeline.remove_object.assert_called_once()
        self.assertEqual(out.selected_model, "Moebius")

    def test_list_models_returns_entries(self) -> None:
        models = self.service.list_models()
        ids = {entry["id"] for entry in models}
        self.assertIn("sam2", ids)
        self.assertIn("moebius", ids)
        self.assertIn("pixelhacker", ids)


if __name__ == "__main__":
    unittest.main()
