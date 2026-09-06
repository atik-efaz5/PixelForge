"""Unit tests for ImageEditingService delegation."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from apps.backend.services import ImageEditingService
from models.errors import ModelLoadError
from models.types import (
    BackendType,
    BoundingBox,
    GroundingResult,
    InpaintingResult,
    SegmentationResult,
    TextSelectionResult,
    validate_image,
    validate_mask,
)
from pipelines.types import ImageEditPipelineResult, MaskRefinementOps, PipelineLatency


def _rgb() -> np.ndarray:
    return validate_image(np.zeros((32, 32, 3), dtype=np.uint8))


def _photo() -> np.ndarray:
    """Non-uniform RGB so document solid-fill does not short-circuit Moebius."""
    yy, xx = np.ogrid[:32, :32]
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[..., 0] = (xx * 7 + yy * 3) % 256
    img[..., 1] = (xx * 5 + yy * 11) % 256
    img[..., 2] = (xx * 13 + yy * 2) % 256
    return validate_image(img)


def _mask() -> np.ndarray:
    m = np.zeros((32, 32), dtype=bool)
    m[8:24, 8:24] = True
    return validate_mask(m)


def _page() -> np.ndarray:
    img = np.full((64, 64, 3), 250, dtype=np.uint8)
    img[16:48, 16:48] = (40, 70, 110)
    return validate_image(img)


def _page_mask() -> np.ndarray:
    m = np.zeros((64, 64), dtype=bool)
    m[16:48, 16:48] = True
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
        out = self.service.inpaint(_photo(), _mask(), backend="moebius")
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

    def test_select_smart_text_falls_back_to_isolated_grounding(self) -> None:
        self.pipeline.select_smart.side_effect = ModelLoadError(
            "Grounding DINO failed to load."
        )
        detection = BoundingBox(4.0, 4.0, 20.0, 20.0, 0.9, "black bag")
        grounding = GroundingResult(
            detections=[detection],
            model="Grounding DINO",
            prompt="black bag",
            backend=BackendType.CPU,
        )
        text = TextSelectionResult(
            mask=_mask(),
            segmentation=SegmentationResult(
                mask=_mask(),
                confidence=0.95,
                model="SAM",
                method="box",
                metadata={},
            ),
            grounding=grounding,
            selected_detection=detection,
            metadata={"prompt": "black bag"},
        )
        self.pipeline.select_from_grounding.return_value = text
        image = _rgb()
        with patch(
            "apps.backend.services.ground_via_isolated_env", return_value=grounding
        ) as isolated:
            out = self.service.select_smart(
                image, text_prompt="black bag", selection_mode="smart"
            )
        isolated.assert_called_once()
        self.pipeline.select_from_grounding.assert_called_once()
        self.assertEqual(out.method, "text")
        self.assertEqual(out.selected_detection.label, "black bag")
        self.assertTrue(out.metadata.get("isolated_grounding"))

    def test_select_smart_point_does_not_use_isolated_grounding(self) -> None:
        self.pipeline.select_smart.side_effect = ModelLoadError("SAM failed to load.")
        with patch("apps.backend.services.ground_via_isolated_env") as isolated:
            with self.assertRaises(ModelLoadError):
                self.service.select_smart(_rgb(), x=8, y=8, selection_mode="point")
        isolated.assert_not_called()
        self.pipeline.select_from_grounding.assert_not_called()

    def test_inpaint_auto_falls_back_to_isolated_moebius(self) -> None:
        self.pipeline.inpaint.side_effect = ModelLoadError("Moebius failed to load.")
        expected = InpaintingResult(
            result=_rgb(),
            latency_ms=20.0,
            memory_mb=None,
            model="Moebius",
            backend=BackendType.LOCAL_MPS,
            metadata={"isolated_env": True},
        )
        with patch(
            "apps.backend.services.inpaint_via_isolated_env", return_value=expected
        ) as isolated:
            out = self.service.inpaint(_photo(), _mask(), backend="auto")
        isolated.assert_called_once()
        self.assertEqual(out.model, "Moebius")

    def test_inpaint_pixelhacker_does_not_use_isolated_moebius(self) -> None:
        self.pipeline.inpaint.side_effect = ModelLoadError(
            "PixelHacker failed to load."
        )
        with patch("apps.backend.services.inpaint_via_isolated_env") as isolated:
            with self.assertRaises(ModelLoadError):
                self.service.inpaint(_photo(), _mask(), backend="pixelhacker")
        isolated.assert_not_called()

    def test_inpaint_uniform_page_skips_pipeline(self) -> None:
        with patch("apps.backend.services.inpaint_via_isolated_env") as isolated:
            out = self.service.inpaint(_page(), _page_mask(), backend="moebius")
        self.pipeline.inpaint.assert_not_called()
        isolated.assert_not_called()
        self.assertEqual(out.model, "solid_fill")
        self.assertTrue(out.metadata.get("solid_fill"))
        np.testing.assert_allclose(out.result[32, 32], (250, 250, 250), atol=2)
        np.testing.assert_array_equal(out.result[4, 4], _page()[4, 4])

    def test_edit_localized_uniform_page_skips_pipeline(self) -> None:
        with patch("apps.backend.services.inpaint_via_isolated_env") as isolated:
            out = self.service.edit_localized(_page(), _page_mask(), backend="auto")
        self.pipeline.edit_localized.assert_not_called()
        isolated.assert_not_called()
        self.assertEqual(out.model, "solid_fill")

    def test_generate_candidates_uniform_page_skips_pipeline(self) -> None:
        out = self.service.generate_candidates(_page(), _page_mask(), count=1)
        self.pipeline.generate_candidates.assert_not_called()
        self.assertEqual(out.model, "solid_fill")
        self.assertEqual(out.selected_candidate_id, "candidate_1")
        self.assertEqual(len(out.candidates), 1)
        np.testing.assert_allclose(
            out.candidates[0].result[32, 32], (250, 250, 250), atol=2
        )


if __name__ == "__main__":
    unittest.main()
