"""Unit tests for smart selection pipeline paths (mocked adapters)."""

from __future__ import annotations

import unittest

import numpy as np

from models.adapters.base import GroundingAdapter, SegmentationAdapter
from models.errors import ModelInferenceError
from models.types import (
    BackendType,
    BoundingBox,
    GroundingResult,
    SegmentationResult,
    validate_image,
    validate_mask,
)
from pipelines.errors import PipelinePromptError, PipelineValidationError
from pipelines.orchestration.image_edit_pipeline import ImageEditPipeline


def _rgb(h: int = 64, w: int = 64) -> np.ndarray:
    return validate_image(np.zeros((h, w, 3), dtype=np.uint8))


def _mask(h: int = 64, w: int = 64) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    m[20:40, 20:40] = True
    return validate_mask(m)


class FakeGroundingAdapter(GroundingAdapter):
    backend_type = BackendType.CPU
    model_name = "Fake Grounding DINO"

    def __init__(self, detections: list[BoundingBox] | None = None) -> None:
        super().__init__()
        self.available = True
        self._detections = (
            list(detections)
            if detections is not None
            else [
                BoundingBox(10.0, 10.0, 30.0, 30.0, 0.9, "dog"),
                BoundingBox(35.0, 35.0, 55.0, 55.0, 0.7, "cat"),
            ]
        )

    def is_available(self) -> bool:
        return self.available

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def infer(self, image: np.ndarray, text_prompt: str, /) -> GroundingResult:
        return GroundingResult(
            detections=list(self._detections),
            model=self.model_name,
            prompt=text_prompt,
            backend=self.backend_type,
        )


class FakeSmartSegmentationAdapter(SegmentationAdapter):
    backend_type = BackendType.LOCAL_MPS
    model_name = "Fake SAM"

    def __init__(self) -> None:
        super().__init__()
        self.available = True

    def is_available(self) -> bool:
        return self.available

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def infer(self, image: np.ndarray, x: int, y: int, /) -> SegmentationResult:
        return SegmentationResult(
            mask=_mask(),
            confidence=0.9,
            model=self.model_name,
            method="point",
        )

    def segment_point_candidates(
        self, image: np.ndarray, x: int, y: int
    ) -> list[SegmentationResult]:
        small = np.zeros((64, 64), dtype=bool)
        small[30, 30] = True
        large = np.zeros((64, 64), dtype=bool)
        large[10:50, 10:50] = True
        return [
            SegmentationResult(
                mask=validate_mask(small),
                confidence=0.6,
                model=self.model_name,
                method="point",
                metadata={"multimask_index": 0},
            ),
            SegmentationResult(
                mask=validate_mask(large),
                confidence=0.85,
                model=self.model_name,
                method="point",
                metadata={"multimask_index": 1},
            ),
        ]

    def segment_box(
        self, image: np.ndarray, x1: int, y1: int, x2: int, y2: int
    ) -> SegmentationResult:
        m = np.zeros(image.shape[:2], dtype=bool)
        m[y1:y2, x1:x2] = True
        return SegmentationResult(
            mask=validate_mask(m),
            confidence=0.88,
            model=self.model_name,
            method="box",
        )


class TestSmartSelectionPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = ImageEditPipeline(
            segmentation_provider=lambda: FakeSmartSegmentationAdapter(),
            grounding_provider=lambda _name: FakeGroundingAdapter(),
        )

    def test_explicit_point_mode(self) -> None:
        result = self.pipeline.select_smart(
            _rgb(), x=32, y=32, selection_mode="point"
        )
        self.assertEqual(result.method, "point")
        self.assertEqual(result.selection_mode, "point")
        self.assertTrue(result.mask.any())

    def test_explicit_text_mode(self) -> None:
        result = self.pipeline.select_smart(
            _rgb(), text_prompt="dog", selection_mode="text"
        )
        self.assertEqual(result.method, "text")
        self.assertIsNotNone(result.grounding)

    def test_smart_mode_prefers_text_when_prompt_given(self) -> None:
        result = self.pipeline.select_smart(
            _rgb(), x=32, y=32, text_prompt="dog", selection_mode="smart"
        )
        self.assertEqual(result.method, "text")

    def test_smart_mode_uses_point_without_prompt(self) -> None:
        result = self.pipeline.select_smart(_rgb(), x=32, y=32, selection_mode="smart")
        self.assertEqual(result.method, "point")

    def test_smart_point_picks_larger_valid_candidate(self) -> None:
        result = self.pipeline.select_smart(_rgb(), x=32, y=32, selection_mode="point")
        self.assertGreater(int(result.mask.sum()), 1)
        self.assertIn("selected", result.ranking)

    def test_smart_text_ranks_detections(self) -> None:
        result = self.pipeline.select_smart(
            _rgb(), text_prompt="dog", selection_mode="text"
        )
        self.assertGreaterEqual(len(result.ranking.get("candidates", [])), 1)
        self.assertIn(result.confidence_tier, {"HIGH", "MEDIUM", "LOW"})

    def test_smart_requires_input(self) -> None:
        with self.assertRaises(PipelineValidationError):
            self.pipeline.select_smart(_rgb(), selection_mode="smart")

    def test_text_mode_requires_prompt(self) -> None:
        with self.assertRaises(PipelinePromptError):
            self.pipeline.select_smart(_rgb(), selection_mode="text")

    def test_no_detections_raises(self) -> None:
        pipeline = ImageEditPipeline(
            segmentation_provider=lambda: FakeSmartSegmentationAdapter(),
            grounding_provider=lambda _name: FakeGroundingAdapter(detections=[]),
        )
        with self.assertRaises(ModelInferenceError):
            pipeline.select_smart(_rgb(), text_prompt="dog", selection_mode="text")


if __name__ == "__main__":
    unittest.main()
