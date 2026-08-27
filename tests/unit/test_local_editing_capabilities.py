"""Unit tests for localized editing capability contract (mocked adapters)."""

from __future__ import annotations

import unittest

import numpy as np

from apps.backend.schemas import EditingCapabilitiesResponse, EditingCapabilityEntry
from apps.backend.services import ImageEditingService
from models.adapters.base import InpaintingAdapter
from models.types import (
    BackendType,
    InpaintingResult,
    validate_image,
    validate_mask,
)
from pipelines.editing_capabilities import (
    BACKEND_EDIT_CAPABILITIES,
    EditIntent,
    localized_edit_supported,
    supports_intent,
)
from pipelines.errors import PipelineBackendError, UnsupportedEditIntentError
from pipelines.orchestration.image_edit_pipeline import ImageEditPipeline


def _rgb(h: int = 32, w: int = 32) -> np.ndarray:
    return validate_image(np.zeros((h, w, 3), dtype=np.uint8))


def _mask(h: int = 32, w: int = 32) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    m[8:24, 8:24] = True
    return validate_mask(m)


class FakeInpaintingAdapter(InpaintingAdapter):
    backend_type = BackendType.LOCAL_MPS
    model_name = "Fake Moebius"

    def __init__(self) -> None:
        super().__init__()
        self.available = True
        self.infer_calls: list[np.ndarray] = []

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def is_available(self) -> bool:
        return self.available

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def infer(self, image: np.ndarray, mask: np.ndarray, params=None, /) -> InpaintingResult:
        self.infer_calls.append(mask.copy())
        out = image.copy()
        out[mask] = 200
        return InpaintingResult(
            result=out,
            latency_ms=10.0,
            memory_mb=None,
            model=self.model_name,
            backend=self.backend_type,
            metadata={},
        )


class TestEditingCapabilityRegistry(unittest.TestCase):
    def test_moebius_declares_localized_inpaint_only(self) -> None:
        cap = BACKEND_EDIT_CAPABILITIES["moebius"]
        self.assertIn(EditIntent.LOCALIZED_INPAINT, cap.supported_intents)
        self.assertNotIn(EditIntent.SEMANTIC_REPLACE, cap.supported_intents)
        self.assertFalse(cap.accepts_text_instruction)
        self.assertFalse(cap.accepts_reference_image)

    def test_localized_edit_supported_for_moebius(self) -> None:
        self.assertTrue(localized_edit_supported("moebius"))
        self.assertFalse(supports_intent("moebius", EditIntent.SEMANTIC_REPLACE))
        self.assertFalse(localized_edit_supported("unknown"))


class TestEditLocalizedPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.inpaint = FakeInpaintingAdapter()
        self.pipeline = ImageEditPipeline(
            inpaint_provider=lambda _name: self.inpaint,
        )

    def test_edit_localized_delegates_to_inpaint(self) -> None:
        img = _rgb()
        mask = _mask()
        result = self.pipeline.edit_localized(img, mask, backend="moebius")
        self.assertEqual(len(self.inpaint.infer_calls), 1)
        self.assertEqual(result.model, "Fake Moebius")

    def test_edit_localized_rejects_instruction(self) -> None:
        with self.assertRaises(UnsupportedEditIntentError):
            self.pipeline.edit_localized(
                _rgb(),
                _mask(),
                backend="moebius",
                instruction="replace with a red chair",
            )
        self.assertEqual(len(self.inpaint.infer_calls), 0)

    def test_edit_localized_rejects_whitespace_instruction(self) -> None:
        result = self.pipeline.edit_localized(
            _rgb(),
            _mask(),
            backend="moebius",
            instruction="   ",
        )
        self.assertEqual(len(self.inpaint.infer_calls), 1)
        self.assertEqual(result.model, "Fake Moebius")

    def test_edit_localized_unsupported_backend(self) -> None:
        with self.assertRaises(PipelineBackendError):
            self.pipeline.edit_localized(_rgb(), _mask(), backend="not-a-backend")


class TestEditingCapabilitiesService(unittest.TestCase):
    def test_list_editing_capabilities_shape(self) -> None:
        service = ImageEditingService(pipeline=ImageEditPipeline())
        rows = service.list_editing_capabilities()
        self.assertGreaterEqual(len(rows), 1)
        moebius = next(r for r in rows if r["backend_id"] == "moebius")
        self.assertTrue(moebius["localized_inpaint"])
        self.assertFalse(moebius["semantic_replace"])
        self.assertFalse(moebius["accepts_text_instruction"])

    def test_capabilities_response_schema(self) -> None:
        entry = EditingCapabilityEntry(
            backend_id="moebius",
            localized_inpaint=True,
            semantic_replace=False,
            accepts_text_instruction=False,
            accepts_reference_image=False,
            notes="mask only",
        )
        payload = EditingCapabilitiesResponse(capabilities=[entry])
        self.assertEqual(payload.capabilities[0].backend_id, "moebius")


if __name__ == "__main__":
    unittest.main()
