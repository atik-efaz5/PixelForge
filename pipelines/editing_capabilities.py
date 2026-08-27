"""Declared localized-editing capabilities for PixelForge backends.

This module records what each validated backend actually supports. It does not
perform inference. Semantic instruction editing is **not** claimed for Moebius.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class EditIntent(str, Enum):
    """High-level edit intent separated from selection and generation."""

    LOCALIZED_INPAINT = "localized_inpaint"
    """Fill or remove the masked region using mask-conditioned diffusion only."""

    SEMANTIC_REPLACE = "semantic_replace"
    """Replace masked content per a natural-language instruction (not on Moebius)."""


@dataclass(frozen=True)
class BackendEditCapability:
    backend_id: str
    supported_intents: frozenset[EditIntent]
    accepts_text_instruction: bool
    accepts_reference_image: bool
    notes: str


MOEBIUS_CAPABILITY = BackendEditCapability(
    backend_id="moebius",
    supported_intents=frozenset({EditIntent.LOCALIZED_INPAINT}),
    accepts_text_instruction=False,
    accepts_reference_image=False,
    notes=(
        "Mask-conditioned object removal / inpainting only. Conditioning uses fixed "
        "learned embedding indices for CFG, not user text."
    ),
)

BACKEND_EDIT_CAPABILITIES: dict[str, BackendEditCapability] = {
    "moebius": MOEBIUS_CAPABILITY,
}


def capability_for(backend: str) -> BackendEditCapability | None:
    return BACKEND_EDIT_CAPABILITIES.get(backend.strip().lower())


def supports_intent(backend: str, intent: EditIntent) -> bool:
    cap = capability_for(backend)
    return cap is not None and intent in cap.supported_intents


def localized_edit_supported(backend: str) -> bool:
    return supports_intent(backend, EditIntent.LOCALIZED_INPAINT)
