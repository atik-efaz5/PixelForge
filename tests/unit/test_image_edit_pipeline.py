"""Unit tests for the image-editing pipeline (mocked adapters only)."""

from __future__ import annotations

import unittest
import numpy as np

from models.adapters.base import InpaintingAdapter, SegmentationAdapter
from models.errors import ModelInferenceError, ModelUnavailableError
from models.types import (
    BackendType,
    InpaintingResult,
    SegmentationResult,
    validate_image,
    validate_mask,
)
from pipelines.errors import PipelineBackendError, PipelineValidationError
from pipelines.mask_refinement import refine_mask, require_non_empty_mask
from pipelines.orchestration.image_edit_pipeline import ImageEditPipeline
from pipelines.types import MaskRefinementOps


def _rgb(h: int = 64, w: int = 64) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.stack(
        [
            (30 + yy).astype(np.uint8),
            (40 + xx).astype(np.uint8),
            (50 + (yy + xx) // 2).astype(np.uint8),
        ],
        axis=-1,
    )
    return validate_image(img)


def _mask(h: int = 64, w: int = 64, *, filled: bool = True) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    if filled:
        m[20:40, 20:40] = True
    return validate_mask(m)


class FakeSegmentationAdapter(SegmentationAdapter):
    backend_type = BackendType.LOCAL_MPS
    model_name = "Fake SAM"

    def __init__(self) -> None:
        super().__init__()
        self.infer_calls: list[tuple[int, int]] = []
        self.load_calls = 0
        self.unload_calls = 0
        self.available = True
        self.empty_mask = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def is_available(self) -> bool:
        return self.available

    def load(self) -> None:
        self.load_calls += 1
        self._loaded = True

    def unload(self) -> None:
        self.unload_calls += 1
        self._loaded = False

    def infer(self, image: np.ndarray, x: int, y: int, /) -> SegmentationResult:
        self.infer_calls.append((x, y))
        mask = _mask(image.shape[0], image.shape[1], filled=not self.empty_mask)
        return SegmentationResult(
            mask=mask,
            confidence=0.9,
            model=self.model_name,
            method="point",
            metadata={"latency_ms": 12.5},
        )


class FakeInpaintingAdapter(InpaintingAdapter):
    backend_type = BackendType.LOCAL_MPS
    model_name = "Fake Moebius"

    def __init__(self) -> None:
        super().__init__()
        self.infer_calls: list[np.ndarray] = []
        self.load_calls = 0
        self.unload_calls = 0
        self.available = True

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def is_available(self) -> bool:
        return self.available

    def load(self) -> None:
        self.load_calls += 1
        self._loaded = True

    def unload(self) -> None:
        self.unload_calls += 1
        self._loaded = False

    def infer(self, image: np.ndarray, mask: np.ndarray, params=None, /) -> InpaintingResult:
        self.infer_calls.append(mask.copy())
        out = image.copy()
        out[mask] = 255
        return InpaintingResult(
            result=out,
            latency_ms=42.0,
            memory_mb=100.0,
            model=self.model_name,
            backend=self.backend_type,
            metadata={},
        )


class TestImageContracts(unittest.TestCase):
    def test_valid_image_contract(self) -> None:
        img = _rgb()
        self.assertEqual(img.dtype, np.uint8)
        self.assertEqual(img.shape, (64, 64, 3))

    def test_invalid_image_shape(self) -> None:
        with self.assertRaises(ValueError):
            validate_image(np.zeros((8, 8), dtype=np.uint8))

    def test_invalid_image_dtype(self) -> None:
        with self.assertRaises(ValueError):
            validate_image(np.zeros((8, 8, 3), dtype=np.float32))


class TestMaskContracts(unittest.TestCase):
    def test_valid_mask_contract(self) -> None:
        m = _mask()
        self.assertEqual(m.dtype, bool)
        self.assertEqual(m.shape, (64, 64))

    def test_mismatched_image_mask_dimensions(self) -> None:
        img = _rgb(32, 32)
        mask = _mask(64, 64)
        with self.assertRaises(ValueError):
            refine_mask(mask, img)

    def test_empty_mask_rejection(self) -> None:
        with self.assertRaises(PipelineValidationError):
            require_non_empty_mask(_mask(filled=False), stage="inpaint")

    def test_mask_refinement_add_remove(self) -> None:
        img = _rgb()
        base = np.zeros((64, 64), dtype=bool)
        base[10:20, 10:20] = True
        add = np.zeros((64, 64), dtype=bool)
        add[30:35, 30:35] = True
        remove = np.zeros((64, 64), dtype=bool)
        remove[12:14, 12:14] = True

        refined = refine_mask(
            base,
            img,
            MaskRefinementOps(add=add, remove=remove),
        )
        self.assertTrue(refined[10, 10])
        self.assertFalse(refined[12, 12])
        self.assertTrue(refined[30, 30])

        dilated = refine_mask(base, img, MaskRefinementOps(dilate=1))
        self.assertGreater(int(dilated.sum()), int(base.sum()))


class TestPipelineValidation(unittest.TestCase):
    def setUp(self) -> None:
        self.seg = FakeSegmentationAdapter()
        self.inpaint = FakeInpaintingAdapter()
        self.pipeline = ImageEditPipeline(
            segmentation_provider=lambda: self.seg,
            inpaint_provider=lambda _name: self.inpaint,
        )

    def test_invalid_point_rejection(self) -> None:
        img = _rgb()
        with self.assertRaises(PipelineValidationError):
            self.pipeline.segment(img, -1, 5)
        with self.assertRaises(PipelineValidationError):
            self.pipeline.segment(img, 5, 100)

    def test_unsupported_backend(self) -> None:
        img = _rgb()
        mask = _mask()
        with self.assertRaises(PipelineBackendError):
            self.pipeline.inpaint(img, mask, backend="unknown-model")

    def test_unavailable_backend(self) -> None:
        self.inpaint.available = False
        with self.assertRaises(ModelUnavailableError):
            self.pipeline.inpaint(_rgb(), _mask(), backend="moebius")


class TestPipelineAdapterInteraction(unittest.TestCase):
    def setUp(self) -> None:
        self.seg = FakeSegmentationAdapter()
        self.inpaint = FakeInpaintingAdapter()
        self.pipeline = ImageEditPipeline(
            segmentation_provider=lambda: self.seg,
            inpaint_provider=lambda name: self.inpaint,
            unload_between_stages=True,
        )

    def test_segment_loads_and_unloads(self) -> None:
        img = _rgb()
        result = self.pipeline.segment(img, 32, 32)
        self.assertEqual(self.seg.load_calls, 1)
        self.assertEqual(self.seg.unload_calls, 1)
        self.assertFalse(self.seg.is_loaded)
        self.assertTrue(result.mask.any())

    def test_inpaint_loads_and_unloads(self) -> None:
        img = _rgb()
        mask = _mask()
        result = self.pipeline.inpaint(img, mask, backend="moebius")
        self.assertEqual(self.inpaint.load_calls, 1)
        self.assertEqual(self.inpaint.unload_calls, 1)
        self.assertFalse(self.inpaint.is_loaded)
        self.assertEqual(result.result.shape, img.shape)

    def test_remove_object_workflow(self) -> None:
        img = _rgb()
        out = self.pipeline.remove_object(img, 32, 32, backend="moebius")
        self.assertEqual(self.seg.infer_calls, [(32, 32)])
        self.assertEqual(len(self.inpaint.infer_calls), 1)
        self.assertEqual(out.result.shape, img.shape)
        self.assertEqual(out.selected_model, "Fake Moebius")
        self.assertEqual(out.backend, BackendType.LOCAL_MPS)
        self.assertIsNotNone(out.latency.total_ms)
        self.assertIsNotNone(out.segmentation)
        self.assertIsNotNone(out.inpainting)

    def test_empty_segmentation_raises(self) -> None:
        self.seg.empty_mask = True
        with self.assertRaises(ModelInferenceError):
            self.pipeline.segment(_rgb(), 10, 10)

    def test_refinement_applied_in_remove_object(self) -> None:
        img = _rgb()
        remove = np.zeros((64, 64), dtype=bool)
        remove[25:35, 25:35] = True
        before_area = int(_mask().sum())
        out = self.pipeline.remove_object(
            img,
            32,
            32,
            backend="moebius",
            refinement=MaskRefinementOps(remove=remove),
        )
        self.assertLess(int(out.mask.sum()), before_area)


if __name__ == "__main__":
    unittest.main()
