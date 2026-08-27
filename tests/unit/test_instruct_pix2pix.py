"""Unit tests for InstructPix2Pix adapter and global instruction editing (mocked)."""

from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock

import numpy as np

from apps.backend.schemas import EditByInstructionMetadata, EditingCapabilityEntry
from apps.backend.services import ImageEditingService, parse_instruction_edit_params
from models.adapters.base import InstructionEditAdapter
from models.adapters.instruct_pix2pix_adapter import InstructPix2PixAdapter
from models.errors import ModelInferenceError, ModelUnavailableError
from models.registry import get_adapter, known_models, reset_registry
from models.types import (
    BackendType,
    InstructionEditParams,
    InstructionEditResult,
    validate_image,
)
from pipelines.editing_capabilities import (
    BACKEND_EDIT_CAPABILITIES,
    EditIntent,
    global_instruction_edit_supported,
    supports_intent,
)
from pipelines.errors import PipelineBackendError, PipelinePromptError, UnsupportedEditIntentError
from pipelines.orchestration.image_edit_pipeline import ImageEditPipeline


def _rgb(h: int = 32, w: int = 32) -> np.ndarray:
    return validate_image(np.zeros((h, w, 3), dtype=np.uint8))


class FakeInstructionEditAdapter(InstructionEditAdapter):
    backend_type = BackendType.CLOUD_GPU
    model_name = "Fake InstructPix2Pix"

    def __init__(self) -> None:
        super().__init__()
        self.available = True
        self.infer_calls: list[tuple[str, InstructionEditParams | None]] = []

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def is_available(self) -> bool:
        return self.available

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def infer(
        self,
        image: np.ndarray,
        instruction: str,
        params: InstructionEditParams | None = None,
        /,
    ) -> InstructionEditResult:
        self.infer_calls.append((instruction, params))
        out = image.copy()
        out[:, :, 0] = 255
        return InstructionEditResult(
            result=out,
            latency_ms=25.0,
            memory_mb=None,
            model=self.model_name,
            backend=self.backend_type,
            instruction=instruction,
            metadata={},
        )


class TestInstructPix2PixAdapter(unittest.TestCase):
    def setUp(self) -> None:
        reset_registry()
        self._endpoint_env = os.environ.pop("PIXELFORGE_INSTRUCT_PIX2PIX_ENDPOINT", None)

    def tearDown(self) -> None:
        reset_registry()
        if self._endpoint_env is None:
            os.environ.pop("PIXELFORGE_INSTRUCT_PIX2PIX_ENDPOINT", None)
        else:
            os.environ["PIXELFORGE_INSTRUCT_PIX2PIX_ENDPOINT"] = self._endpoint_env

    def test_registry_includes_instruct_pix2pix(self) -> None:
        self.assertIn("instruct_pix2pix", known_models())
        adapter = get_adapter("instruct_pix2pix")
        self.assertIsInstance(adapter, InstructPix2PixAdapter)
        self.assertIs(adapter.backend_type, BackendType.CLOUD_GPU)

    def test_unavailable_without_endpoint(self) -> None:
        adapter = InstructPix2PixAdapter(endpoint="")
        self.assertFalse(adapter.is_available())
        with self.assertRaisesRegex(ModelUnavailableError, "not configured"):
            adapter.infer(_rgb(), "make it sunset")

    def test_endpoint_configured_but_infer_not_implemented(self) -> None:
        adapter = InstructPix2PixAdapter(endpoint="https://example.invalid/edit")
        self.assertTrue(adapter.is_available())
        with self.assertRaisesRegex(ModelInferenceError, "not implemented"):
            adapter.infer(_rgb(), "make it sunset")

    def test_empty_instruction_rejected(self) -> None:
        adapter = InstructPix2PixAdapter(endpoint="https://example.invalid/edit")
        with self.assertRaises(ValueError):
            adapter.infer(_rgb(), "   ")

    def test_remote_request_shape(self) -> None:
        adapter = InstructPix2PixAdapter(endpoint="https://example.invalid/edit")
        payload = adapter.build_remote_request(
            _rgb(),
            "make the sky sunset",
            InstructionEditParams(num_steps=50),
        )
        self.assertEqual(payload["endpoint"], "https://example.invalid/edit")
        self.assertEqual(payload["instruction"], "make the sky sunset")
        self.assertEqual(payload["num_steps"], 50)
        self.assertEqual(payload["guidance_text"], 7.5)


class TestInstructionEditCapabilities(unittest.TestCase):
    def test_instruct_pix2pix_declares_global_instruction_only(self) -> None:
        cap = BACKEND_EDIT_CAPABILITIES["instruct_pix2pix"]
        self.assertIn(EditIntent.GLOBAL_INSTRUCTION_EDIT, cap.supported_intents)
        self.assertNotIn(EditIntent.MASK_CONDITIONED_EDIT, cap.supported_intents)
        self.assertTrue(cap.accepts_text_instruction)
        self.assertFalse(cap.accepts_mask)

    def test_global_instruction_supported(self) -> None:
        self.assertTrue(global_instruction_edit_supported("instruct_pix2pix"))
        self.assertFalse(supports_intent("instruct_pix2pix", EditIntent.MASK_CONDITIONED_EDIT))


class TestEditByInstructionPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.editor = FakeInstructionEditAdapter()
        self.pipeline = ImageEditPipeline(
            instruction_edit_provider=lambda _name: self.editor,
        )

    def test_edit_by_instruction_delegates(self) -> None:
        result = self.pipeline.edit_by_instruction(
            _rgb(), "make the sky sunset", backend="instruct_pix2pix"
        )
        self.assertEqual(len(self.editor.infer_calls), 1)
        self.assertEqual(result.instruction, "make the sky sunset")
        self.assertEqual(result.model, "Fake InstructPix2Pix")

    def test_empty_instruction_rejected(self) -> None:
        with self.assertRaises(PipelinePromptError):
            self.pipeline.edit_by_instruction(_rgb(), "   ", backend="instruct_pix2pix")
        self.assertEqual(len(self.editor.infer_calls), 0)

    def test_mask_rejected(self) -> None:
        mask = np.zeros((32, 32), dtype=bool)
        with self.assertRaises(UnsupportedEditIntentError):
            self.pipeline.edit_by_instruction(
                _rgb(),
                "edit",
                backend="instruct_pix2pix",
                mask=mask,
            )

    def test_unavailable_backend_raises(self) -> None:
        self.editor.available = False
        with self.assertRaises(ModelUnavailableError):
            self.pipeline.edit_by_instruction(_rgb(), "edit", backend="instruct_pix2pix")

    def test_unsupported_backend_raises(self) -> None:
        with self.assertRaises(PipelineBackendError):
            self.pipeline.edit_by_instruction(_rgb(), "edit", backend="moebius")


class TestEditByInstructionService(unittest.TestCase):
    def test_service_delegates(self) -> None:
        pipeline = MagicMock()
        expected = InstructionEditResult(
            result=_rgb(),
            latency_ms=1.0,
            memory_mb=None,
            model="InstructPix2Pix",
            backend=BackendType.CLOUD_GPU,
            instruction="sunset",
            metadata={},
        )
        pipeline.edit_by_instruction.return_value = expected
        service = ImageEditingService(pipeline=pipeline)
        out = service.edit_by_instruction(_rgb(), "sunset")
        pipeline.edit_by_instruction.assert_called_once()
        self.assertEqual(out.instruction, "sunset")

    def test_parse_instruction_edit_params_none_when_empty(self) -> None:
        self.assertIsNone(parse_instruction_edit_params())

    def test_parse_instruction_edit_params_partial(self) -> None:
        params = parse_instruction_edit_params(num_steps=80)
        self.assertIsNotNone(params)
        assert params is not None
        self.assertEqual(params.num_steps, 80)


class TestEditByInstructionSchema(unittest.TestCase):
    def test_metadata_schema(self) -> None:
        meta = EditByInstructionMetadata(
            model="InstructPix2Pix",
            backend="CLOUD_GPU",
            instruction="make it sunset",
            latency_ms=12.5,
            metadata={},
        )
        self.assertEqual(meta.instruction, "make it sunset")

    def test_capability_entry_includes_global_instruction(self) -> None:
        entry = EditingCapabilityEntry(
            backend_id="instruct_pix2pix",
            localized_inpaint=False,
            semantic_replace=False,
            global_instruction_edit=True,
            mask_conditioned_edit=False,
            accepts_text_instruction=True,
            accepts_reference_image=False,
            notes="cloud only",
        )
        self.assertTrue(entry.global_instruction_edit)


if __name__ == "__main__":
    unittest.main()
