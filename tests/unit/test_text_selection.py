"""Unit tests for text-guided object selection (mocked grounding + SAM 2)."""

from __future__ import annotations

import unittest

import numpy as np

from models.adapters.base import GroundingAdapter, SegmentationAdapter
from models.errors import ModelInferenceError, ModelUnavailableError
from models.types import (
    BackendType,
    BoundingBox,
    GroundingResult,
    SegmentationResult,
    validate_image,
    validate_mask,
)
from pipelines.errors import PipelineBackendError, PipelinePromptError, PipelineValidationError
from pipelines.orchestration.image_edit_pipeline import ImageEditPipeline


def _rgb(h: int = 64, w: int = 64) -> np.ndarray:
    return validate_image(np.zeros((h, w, 3), dtype=np.uint8))


def _mask(h: int = 64, w: int = 64, *, filled: bool = True) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    if filled:
        m[20:40, 20:40] = True
    return validate_mask(m)


class FakeGroundingAdapter(GroundingAdapter):
    backend_type = BackendType.CPU
    model_name = "Fake Grounding DINO"

    def __init__(self, detections: list[BoundingBox] | None = None) -> None:
        super().__init__()
        self.available = True
        self.infer_calls: list[str] = []
        self._detections = detections if detections is not None else [
            BoundingBox(10.0, 10.0, 30.0, 30.0, 0.9, "dog"),
        ]

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def is_available(self) -> bool:
        return self.available

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def infer(self, image: np.ndarray, text_prompt: str, /) -> GroundingResult:
        self.infer_calls.append(text_prompt)
        return GroundingResult(
            detections=list(self._detections),
            model=self.model_name,
            prompt=text_prompt,
            backend=self.backend_type,
            metadata={"latency_ms": 8.0},
        )


class FakeBoxSegmentationAdapter(SegmentationAdapter):
    backend_type = BackendType.LOCAL_MPS
    model_name = "Fake SAM"

    def __init__(self) -> None:
        super().__init__()
        self.available = True
        self.empty_mask = False
        self.point_calls: list[tuple[int, int]] = []
        self.box_calls: list[tuple[int, int, int, int]] = []

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def is_available(self) -> bool:
        return self.available

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def infer(self, image: np.ndarray, x: int, y: int, /) -> SegmentationResult:
        self.point_calls.append((x, y))
        return SegmentationResult(
            mask=_mask(image.shape[0], image.shape[1], filled=not self.empty_mask),
            confidence=0.85,
            model=self.model_name,
            method="point",
            metadata={},
        )

    def segment_box(
        self, image: np.ndarray, x1: int, y1: int, x2: int, y2: int, /
    ) -> SegmentationResult:
        self.box_calls.append((x1, y1, x2, y2))
        return SegmentationResult(
            mask=_mask(image.shape[0], image.shape[1], filled=not self.empty_mask),
            confidence=0.88,
            model=self.model_name,
            method="box",
            metadata={"box_xyxy": [x1, y1, x2, y2]},
        )


class TestTextPromptValidation(unittest.TestCase):
    def setUp(self) -> None:
        self.grounding = FakeGroundingAdapter()
        self.seg = FakeBoxSegmentationAdapter()
        self.pipeline = ImageEditPipeline(
            segmentation_provider=lambda: self.seg,
            grounding_provider=lambda _name: self.grounding,
        )

    def test_empty_prompt_rejected(self) -> None:
        with self.assertRaises(PipelinePromptError):
            self.pipeline.select_by_text(_rgb(), "")
        with self.assertRaises(PipelinePromptError):
            self.pipeline.select_by_text(_rgb(), "   ")

    def test_whitespace_prompt_is_stripped(self) -> None:
        result = self.pipeline.select_by_text(_rgb(), "  dog  ")
        self.assertEqual(self.grounding.infer_calls, ["dog"])
        self.assertEqual(result.grounding.prompt, "dog")


class TestGroundingResultContract(unittest.TestCase):
    def test_select_by_text_returns_mask_and_metadata(self) -> None:
        grounding = FakeGroundingAdapter()
        seg = FakeBoxSegmentationAdapter()
        pipeline = ImageEditPipeline(
            segmentation_provider=lambda: seg,
            grounding_provider=lambda _name: grounding,
        )
        result = pipeline.select_by_text(_rgb(), "dog")
        self.assertTrue(result.mask.any())
        self.assertEqual(result.selected_detection.label, "dog")
        self.assertEqual(result.segmentation.method, "box")
        self.assertEqual(result.grounding.model, "Fake Grounding DINO")
        self.assertEqual(result.metadata["detection_count"], 1)

    def test_no_detections_raises(self) -> None:
        grounding = FakeGroundingAdapter(detections=[])
        seg = FakeBoxSegmentationAdapter()
        pipeline = ImageEditPipeline(
            segmentation_provider=lambda: seg,
            grounding_provider=lambda _name: grounding,
        )
        with self.assertRaises(ModelInferenceError):
            pipeline.select_by_text(_rgb(), "dog")


class TestMultipleDetections(unittest.TestCase):
    def setUp(self) -> None:
        self.detections = [
            BoundingBox(5.0, 5.0, 20.0, 20.0, 0.95, "dog"),
            BoundingBox(30.0, 30.0, 50.0, 50.0, 0.7, "dog"),
        ]
        self.grounding = FakeGroundingAdapter(self.detections)
        self.seg = FakeBoxSegmentationAdapter()
        self.pipeline = ImageEditPipeline(
            segmentation_provider=lambda: self.seg,
            grounding_provider=lambda _name: self.grounding,
        )

    def test_detection_index_selects_box(self) -> None:
        result = self.pipeline.select_by_text(_rgb(), "dog", detection_index=1)
        self.assertEqual(self.seg.box_calls, [(30, 30, 50, 50)])
        self.assertEqual(result.selected_detection.confidence, 0.7)
        self.assertEqual(result.metadata["detection_index"], 1)
        self.assertEqual(result.metadata["detection_count"], 2)

    def test_out_of_range_detection_index(self) -> None:
        with self.assertRaises(PipelineValidationError):
            self.pipeline.select_by_text(_rgb(), "dog", detection_index=2)

    def test_select_from_grounding_without_rerunning_grounding(self) -> None:
        grounding = GroundingResult(
            detections=self.detections,
            model="cached",
            prompt="dog",
            backend=BackendType.CPU,
        )
        result = self.pipeline.select_from_grounding(_rgb(), grounding, detection_index=0)
        self.assertEqual(self.grounding.infer_calls, [])
        self.assertEqual(self.seg.box_calls, [(5, 5, 20, 20)])
        self.assertEqual(result.grounding.model, "cached")


class TestBoxToMaskContract(unittest.TestCase):
    def test_segment_box_produces_bool_hw_mask(self) -> None:
        seg = FakeBoxSegmentationAdapter()
        pipeline = ImageEditPipeline(segmentation_provider=lambda: seg)
        image = _rgb()
        detection = BoundingBox(12.0, 14.0, 40.0, 42.0, 0.8, "car")
        result = pipeline.select_from_grounding(
            image,
            GroundingResult(
                detections=[detection],
                model="Fake Grounding DINO",
                prompt="car",
                backend=BackendType.CPU,
            ),
        )
        self.assertEqual(result.mask.dtype, bool)
        self.assertEqual(result.mask.shape, image.shape[:2])
        self.assertEqual(seg.box_calls, [(12, 14, 40, 42)])

    def test_empty_box_mask_raises(self) -> None:
        seg = FakeBoxSegmentationAdapter()
        seg.empty_mask = True
        pipeline = ImageEditPipeline(
            segmentation_provider=lambda: seg,
            grounding_provider=lambda _name: FakeGroundingAdapter(),
        )
        with self.assertRaises(ModelInferenceError):
            pipeline.select_by_text(_rgb(), "dog")


class TestUnknownGroundingBackend(unittest.TestCase):
    def test_unsupported_backend(self) -> None:
        pipeline = ImageEditPipeline(
            segmentation_provider=lambda: FakeBoxSegmentationAdapter(),
            grounding_provider=lambda _name: FakeGroundingAdapter(),
        )
        with self.assertRaises(PipelineBackendError):
            pipeline.select_by_text(_rgb(), "dog", grounding_backend="clipseg")

    def test_unavailable_grounding_backend(self) -> None:
        grounding = FakeGroundingAdapter()
        grounding.available = False
        pipeline = ImageEditPipeline(
            segmentation_provider=lambda: FakeBoxSegmentationAdapter(),
            grounding_provider=lambda _name: grounding,
        )
        with self.assertRaises(ModelUnavailableError):
            pipeline.select_by_text(_rgb(), "dog")


if __name__ == "__main__":
    unittest.main()
