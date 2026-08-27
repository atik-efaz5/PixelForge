"""Unit tests for multi-candidate inpainting pipeline and API helpers."""

from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock

import numpy as np

from apps.backend.services import (
    build_inpaint_candidates_metadata,
    build_multipart_inpaint_response,
    parse_inpaint_params,
)
from apps.backend.validation import validate_candidate_count_field
from apps.backend.errors import InvalidInputError
from models.adapters.base import InpaintingAdapter
from models.types import (
    BackendType,
    InpaintCandidate,
    InpaintCandidatesResult,
    InpaintingResult,
    InpaintParams,
    validate_image,
    validate_mask,
)
from pipelines.candidate_seeds import derive_candidate_seeds
from pipelines.orchestration.image_edit_pipeline import ImageEditPipeline


def _rgb(h: int = 32, w: int = 32) -> np.ndarray:
    return validate_image(np.full((h, w, 3), 120, dtype=np.uint8))


def _mask(h: int = 32, w: int = 32) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    m[8:24, 8:24] = True
    return validate_mask(m)


class SeedAwareFakeInpaint(InpaintingAdapter):
    backend_type = BackendType.LOCAL_MPS
    model_name = "Fake Moebius"

    def __init__(self) -> None:
        super().__init__()
        self._loaded = True
        self.calls: list[int | None] = []

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def is_available(self) -> bool:
        return True

    def load(self) -> None:
        self._loaded = True

    def unload(self) -> None:
        self._loaded = False

    def infer(self, image: np.ndarray, mask: np.ndarray, params=None, /) -> InpaintingResult:
        seed = None if params is None else params.seed
        self.calls.append(seed)
        out = image.copy()
        fill = 80 if seed is None else 80 + (seed % 100)
        out[mask] = fill
        return InpaintingResult(
            result=out,
            latency_ms=10.0,
            memory_mb=None,
            model=self.model_name,
            backend=self.backend_type,
            metadata={"seed": seed},
        )


class TestCandidateGenerationPipeline(unittest.TestCase):
    def test_generate_two_candidates_uses_distinct_seeds(self) -> None:
        adapter = SeedAwareFakeInpaint()
        pipeline = ImageEditPipeline(
            inpaint_provider=lambda _name: adapter,
            unload_between_stages=False,
        )
        result = pipeline.generate_candidates(
            _rgb(),
            _mask(),
            backend="moebius",
            count=2,
            params=InpaintParams(seed=5),
        )
        self.assertEqual(len(result.candidates), 2)
        self.assertEqual(adapter.calls, derive_candidate_seeds(5, 2))
        self.assertNotEqual(
            result.candidates[0].output_hash,
            result.candidates[1].output_hash,
        )

    def test_inpaint_single_result_unchanged(self) -> None:
        adapter = SeedAwareFakeInpaint()
        pipeline = ImageEditPipeline(
            inpaint_provider=lambda _name: adapter,
            unload_between_stages=False,
        )
        single = pipeline.inpaint(_rgb(), _mask(), backend="moebius")
        self.assertEqual(single.model, "Fake Moebius")
        self.assertEqual(len(adapter.calls), 1)

    def test_generate_candidates_rejects_invalid_count(self) -> None:
        pipeline = ImageEditPipeline(
            inpaint_provider=lambda _name: SeedAwareFakeInpaint(),
            unload_between_stages=False,
        )
        with self.assertRaises(ValueError):
            pipeline.generate_candidates(_rgb(), _mask(), count=3)


class TestCandidateApiHelpers(unittest.TestCase):
    def test_validate_candidate_count_field(self) -> None:
        self.assertEqual(validate_candidate_count_field(None), 1)
        self.assertEqual(validate_candidate_count_field(2), 2)
        with self.assertRaises(InvalidInputError):
            validate_candidate_count_field(5)

    def test_parse_inpaint_params_seed(self) -> None:
        params = parse_inpaint_params(seed=123)
        assert params is not None
        self.assertEqual(params.seed, 123)

    def test_multipart_response_contains_metadata_and_pngs(self) -> None:
        candidate = InpaintCandidate(
            candidate_id="candidate_1",
            result=_rgb(),
            seed=1,
            latency_ms=1.0,
            memory_mb=None,
            model="Fake",
            backend=BackendType.LOCAL_MPS,
            output_hash="abc",
            validity_status="valid",
            generation_params={"seed": 1},
            metadata={"candidate_score": 0.8, "score_components": {}},
        )
        result = InpaintCandidatesResult(
            candidates=[candidate],
            selected_candidate_id="candidate_1",
            ranking={"selected_candidate_id": "candidate_1"},
            model="Fake",
            backend=BackendType.LOCAL_MPS,
            metadata={},
        )
        metadata = build_inpaint_candidates_metadata(result)
        body, content_type = build_multipart_inpaint_response(
            metadata,
            [("candidate_1", b"\x89PNG\r\n\x1a\n")],
        )
        self.assertIn("multipart/form-data", content_type)
        payload = body.decode("latin-1")
        self.assertIn("candidate_count", payload)
        self.assertEqual(metadata["candidate_count"], 1)


class TestCandidateServiceDelegation(unittest.TestCase):
    def test_generate_candidates_delegates(self) -> None:
        from apps.backend.services import ImageEditingService

        pipeline = MagicMock()
        service = ImageEditingService(pipeline=pipeline)
        expected = InpaintCandidatesResult(
            candidates=[],
            selected_candidate_id="candidate_1",
            ranking={},
            model="Fake",
            backend=BackendType.LOCAL_MPS,
            metadata={},
        )
        pipeline.generate_candidates.return_value = expected
        out = service.generate_candidates(_rgb(), _mask(), count=2)
        pipeline.generate_candidates.assert_called_once()
        self.assertEqual(out.model, "Fake")


if __name__ == "__main__":
    unittest.main()
