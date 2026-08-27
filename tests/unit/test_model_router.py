"""Unit tests for deterministic model routing (no inference)."""

from __future__ import annotations

import unittest

from models.router import (
    ExecutionPreference,
    RoutingCapability,
    RoutingError,
    RoutingOperation,
    RoutingRequest,
    is_automatic_backend,
    route,
)


def _availability(available: set[str]):
    def probe(model_id: str) -> bool:
        return model_id in available

    return probe


class TestModelRouter(unittest.TestCase):
    def test_localized_inpaint_routes_to_moebius(self) -> None:
        decision = route(
            RoutingRequest(
                operation=RoutingOperation.INPAINT,
                required_capability=RoutingCapability.LOCALIZED_INPAINT,
                preferred_backend="auto",
            ),
            availability_probe=_availability({"moebius"}),
        )
        self.assertEqual(decision.model, "moebius")
        self.assertEqual(decision.backend, "LOCAL_MPS")
        self.assertTrue(decision.available)

    def test_text_selection_routes_to_grounding_dino(self) -> None:
        decision = route(
            RoutingRequest(
                operation=RoutingOperation.SELECT_BY_TEXT,
                required_capability=RoutingCapability.OBJECT_SELECTION_TEXT,
                preferred_backend="auto",
            ),
            availability_probe=_availability({"grounding_dino"}),
        )
        self.assertEqual(decision.model, "grounding_dino")
        self.assertEqual(decision.backend, "CPU")

    def test_point_selection_routes_to_sam2(self) -> None:
        decision = route(
            RoutingRequest(
                operation=RoutingOperation.SEGMENT_POINT,
                required_capability=RoutingCapability.OBJECT_SELECTION_POINT,
                preferred_backend="auto",
            ),
            availability_probe=_availability({"sam2"}),
        )
        self.assertEqual(decision.model, "sam2")

    def test_global_instruction_edit_routes_to_instruct_pix2pix_when_available(self) -> None:
        decision = route(
            RoutingRequest(
                operation=RoutingOperation.EDIT_BY_INSTRUCTION,
                required_capability=RoutingCapability.GLOBAL_INSTRUCTION_EDIT,
                preferred_backend="auto",
                execution_preference=ExecutionPreference.CLOUD_FIRST,
            ),
            availability_probe=_availability({"instruct_pix2pix"}),
        )
        self.assertEqual(decision.model, "instruct_pix2pix")
        self.assertEqual(decision.backend, "CLOUD_GPU")

    def test_unavailable_backend_raises(self) -> None:
        with self.assertRaises(RoutingError):
            route(
                RoutingRequest(
                    operation=RoutingOperation.INPAINT,
                    required_capability=RoutingCapability.LOCALIZED_INPAINT,
                    preferred_backend="auto",
                ),
                availability_probe=_availability(set()),
            )

    def test_preferred_backend_unavailable_raises(self) -> None:
        with self.assertRaises(RoutingError) as ctx:
            route(
                RoutingRequest(
                    operation=RoutingOperation.INPAINT,
                    required_capability=RoutingCapability.LOCALIZED_INPAINT,
                    preferred_backend="moebius",
                ),
                availability_probe=_availability(set()),
            )
        self.assertIn("moebius", str(ctx.exception).lower())
        self.assertTrue(ctx.exception.fallbacks)

    def test_local_first_prefers_moebius_over_pixelhacker(self) -> None:
        decision = route(
            RoutingRequest(
                operation=RoutingOperation.INPAINT,
                required_capability=RoutingCapability.LOCALIZED_INPAINT,
                preferred_backend="auto",
                execution_preference=ExecutionPreference.LOCAL_FIRST,
            ),
            availability_probe=_availability({"moebius", "pixelhacker"}),
        )
        self.assertEqual(decision.model, "moebius")

    def test_cloud_first_prefers_pixelhacker_when_available(self) -> None:
        decision = route(
            RoutingRequest(
                operation=RoutingOperation.INPAINT,
                required_capability=RoutingCapability.LOCALIZED_INPAINT,
                preferred_backend="auto",
                execution_preference=ExecutionPreference.CLOUD_FIRST,
            ),
            availability_probe=_availability({"moebius", "pixelhacker"}),
        )
        self.assertEqual(decision.model, "pixelhacker")

    def test_fastest_available_prefers_validated_local(self) -> None:
        decision = route(
            RoutingRequest(
                operation=RoutingOperation.INPAINT,
                required_capability=RoutingCapability.LOCALIZED_INPAINT,
                preferred_backend="auto",
                execution_preference=ExecutionPreference.FASTEST_AVAILABLE,
            ),
            availability_probe=_availability({"moebius", "pixelhacker"}),
        )
        self.assertEqual(decision.model, "moebius")

    def test_quality_first_prefers_pixelhacker_when_available(self) -> None:
        decision = route(
            RoutingRequest(
                operation=RoutingOperation.INPAINT,
                required_capability=RoutingCapability.LOCALIZED_INPAINT,
                preferred_backend="auto",
                execution_preference=ExecutionPreference.QUALITY_FIRST,
            ),
            availability_probe=_availability({"moebius", "pixelhacker"}),
        )
        self.assertEqual(decision.model, "pixelhacker")

    def test_no_silent_capability_substitution(self) -> None:
        with self.assertRaises(RoutingError):
            route(
                RoutingRequest(
                    operation=RoutingOperation.EDIT_BY_INSTRUCTION,
                    required_capability=RoutingCapability.GLOBAL_INSTRUCTION_EDIT,
                    preferred_backend="moebius",
                ),
                availability_probe=_availability({"moebius"}),
            )

    def test_explicit_wrong_capability_backend_rejected(self) -> None:
        with self.assertRaises(RoutingError):
            route(
                RoutingRequest(
                    operation=RoutingOperation.INPAINT,
                    required_capability=RoutingCapability.LOCALIZED_INPAINT,
                    preferred_backend="sam2",
                ),
                availability_probe=_availability({"sam2"}),
            )

    def test_deterministic_routing(self) -> None:
        request = RoutingRequest(
            operation=RoutingOperation.INPAINT,
            required_capability=RoutingCapability.LOCALIZED_INPAINT,
            preferred_backend="auto",
            execution_preference=ExecutionPreference.LOCAL_FIRST,
        )
        probe = _availability({"moebius", "pixelhacker"})
        first = route(request, availability_probe=probe)
        second = route(request, availability_probe=probe)
        self.assertEqual(first.model, second.model)
        self.assertEqual(first.reason, second.reason)

    def test_automatic_backend_aliases(self) -> None:
        self.assertTrue(is_automatic_backend("auto"))
        self.assertTrue(is_automatic_backend("automatic"))
        self.assertFalse(is_automatic_backend("moebius"))


if __name__ == "__main__":
    unittest.main()
