"""Declared editing capabilities for PixelForge backends.

This module records what each validated backend actually supports. It does not
perform inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class EditIntent(str, Enum):
    """High-level edit intent separated from selection and generation."""

    LOCALIZED_INPAINT = "localized_inpaint"
    """Fill or remove the masked region using mask-conditioned diffusion only."""

    SEMANTIC_REPLACE = "semantic_replace"
    """Replace masked content per a natural-language instruction (planned composite)."""

    GLOBAL_INSTRUCTION_EDIT = "global_instruction_edit"
    """Edit the full image from a natural-language instruction."""

    MASK_CONDITIONED_EDIT = "mask_conditioned_edit"
    """Edit only a masked region (requires mask input at inference)."""


@dataclass(frozen=True)
class BackendEditCapability:
    backend_id: str
    supported_intents: frozenset[EditIntent]
    accepts_text_instruction: bool
    accepts_reference_image: bool
    accepts_mask: bool
    notes: str


MOEBIUS_CAPABILITY = BackendEditCapability(
    backend_id="moebius",
    supported_intents=frozenset(
        {EditIntent.LOCALIZED_INPAINT, EditIntent.MASK_CONDITIONED_EDIT}
    ),
    accepts_text_instruction=False,
    accepts_reference_image=False,
    accepts_mask=True,
    notes=(
        "Mask-conditioned object removal / inpainting only. Conditioning uses fixed "
        "learned embedding indices for CFG, not user text."
    ),
)

INSTRUCT_PIX2PIX_CAPABILITY = BackendEditCapability(
    backend_id="instruct_pix2pix",
    supported_intents=frozenset({EditIntent.GLOBAL_INSTRUCTION_EDIT}),
    accepts_text_instruction=True,
    accepts_reference_image=False,
    accepts_mask=False,
    notes=(
        "Global full-frame instruction editing via CLIP text + source-image latent "
        "concatenation. No mask-conditioned path in upstream inference."
    ),
)

BACKEND_EDIT_CAPABILITIES: dict[str, BackendEditCapability] = {
    "moebius": MOEBIUS_CAPABILITY,
    "instruct_pix2pix": INSTRUCT_PIX2PIX_CAPABILITY,
}


def capability_for(backend: str) -> BackendEditCapability | None:
    return BACKEND_EDIT_CAPABILITIES.get(backend.strip().lower())


def supports_intent(backend: str, intent: EditIntent) -> bool:
    cap = capability_for(backend)
    return cap is not None and intent in cap.supported_intents


def localized_edit_supported(backend: str) -> bool:
    return supports_intent(backend, EditIntent.LOCALIZED_INPAINT)


def global_instruction_edit_supported(backend: str) -> bool:
    return supports_intent(backend, EditIntent.GLOBAL_INSTRUCTION_EDIT)
