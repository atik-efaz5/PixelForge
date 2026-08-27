"""Deterministic heuristic ranking for inpainting candidates.

The candidate score is a deterministic heuristic and is not a calibrated measure
of human preference. Do not interpret scores as quality probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from evaluation.metrics.preservation import outside_mask_preservation_score
from evaluation.reproducibility import sha256_bytes
from models.types import validate_image, validate_mask

# Inside-mask change: normalized MAE sweet spot (not too static, not extreme).
INSIDE_CHANGE_LOW = 0.02
INSIDE_CHANGE_HIGH = 0.45
INSIDE_CHANGE_IDEAL_LOW = 0.05
INSIDE_CHANGE_IDEAL_HIGH = 0.25

# Artifact heuristics.
SATURATION_FRACTION_WARN = 0.12
LOCAL_VARIANCE_WARN = 0.35

SIGNAL_WEIGHTS: dict[str, float] = {
    "output_valid": 0.20,
    "finite_pixels": 0.10,
    "correct_dimensions": 0.15,
    "preservation_outside_mask": 0.25,
    "mask_change_magnitude": 0.20,
    "artifact_heuristic": 0.10,
}


@dataclass(frozen=True)
class ScoreComponent:
    name: str
    value: float | None
    available: bool
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": None if self.value is None else round(self.value, 6),
            "available": self.available,
            "note": self.note,
        }


@dataclass(frozen=True)
class ScoredInpaintCandidate:
    candidate_id: str
    score: float
    reasons: tuple[str, ...]
    rejected: bool = False
    rejection_reason: str | None = None
    validity_status: str = "valid"
    components: dict[str, ScoreComponent] = field(default_factory=dict)
    output_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "score": round(self.score, 6),
            "reasons": list(self.reasons),
            "rejected": self.rejected,
            "rejection_reason": self.rejection_reason,
            "validity_status": self.validity_status,
            "components": {k: v.to_dict() for k, v in self.components.items()},
            "output_hash": self.output_hash,
        }


@dataclass(frozen=True)
class InpaintCandidateRanking:
    selected: ScoredInpaintCandidate
    ordered: tuple[ScoredInpaintCandidate, ...]
    rejected: tuple[ScoredInpaintCandidate, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected_candidate_id": self.selected.candidate_id,
            "ordered": [c.to_dict() for c in self.ordered],
            "rejected": [c.to_dict() for c in self.rejected],
        }

    def to_eval_record(
        self,
        *,
        candidate_count: int,
        selected_candidate_id: str | None = None,
    ) -> dict[str, Any]:
        """Evaluation-report fragment for candidate generation."""
        return {
            "candidate_count": candidate_count,
            "candidate_hashes": [
                c.output_hash for c in self.ordered if c.output_hash is not None
            ],
            "scores": {c.candidate_id: c.score for c in self.ordered},
            "ranking": [c.candidate_id for c in self.ordered],
            "selected_candidate_id": selected_candidate_id or self.selected.candidate_id,
            "rejected_candidate_ids": [c.candidate_id for c in self.rejected],
        }


def _inside_mask_change_score(
    original: np.ndarray,
    edited: np.ndarray,
    mask: np.ndarray,
) -> ScoreComponent:
    m = validate_mask(mask)
    inside = m
    count = int(inside.sum())
    if count == 0:
        return ScoreComponent(
            name="mask_change_magnitude",
            value=None,
            available=False,
            note="no pixels inside mask",
        )
    diff = np.abs(original.astype(np.float64) - edited.astype(np.float64))
    mae = float(diff[inside].mean() / 255.0)
    if mae < INSIDE_CHANGE_LOW:
        score = max(0.0, mae / INSIDE_CHANGE_LOW) * 0.4
        note = "very low inside-mask change"
    elif mae > INSIDE_CHANGE_HIGH:
        score = max(0.0, 1.0 - (mae - INSIDE_CHANGE_HIGH) / INSIDE_CHANGE_HIGH)
        note = "excessive inside-mask change"
    elif INSIDE_CHANGE_IDEAL_LOW <= mae <= INSIDE_CHANGE_IDEAL_HIGH:
        score = 1.0
        note = "inside-mask change in expected range"
    elif mae < INSIDE_CHANGE_IDEAL_LOW:
        span = INSIDE_CHANGE_IDEAL_LOW - INSIDE_CHANGE_LOW
        score = 0.4 + 0.6 * ((mae - INSIDE_CHANGE_LOW) / span if span else 1.0)
        note = "moderate inside-mask change (below ideal)"
    else:
        span = INSIDE_CHANGE_HIGH - INSIDE_CHANGE_IDEAL_HIGH
        score = max(0.0, 1.0 - (mae - INSIDE_CHANGE_IDEAL_HIGH) / span if span else 0.0)
        note = "moderate inside-mask change (above ideal)"
    return ScoreComponent(
        name="mask_change_magnitude",
        value=float(min(1.0, max(0.0, score))),
        available=True,
        note=note,
    )


def _artifact_heuristic_score(edited: np.ndarray, mask: np.ndarray) -> ScoreComponent:
    m = validate_mask(mask)
    region = edited[m] if m.any() else edited.reshape(-1, 3)
    if region.size == 0:
        return ScoreComponent(
            name="artifact_heuristic",
            value=None,
            available=False,
            note="empty scored region",
        )
    saturated = np.logical_or(
        region <= 2,
        region >= 253,
    ).all(axis=1)
    sat_frac = float(saturated.mean())
    gray = region.astype(np.float64).mean(axis=1)
    local_var = float(gray.var() / (255.0**2))
    penalty = 0.0
    notes: list[str] = []
    if sat_frac > SATURATION_FRACTION_WARN:
        penalty += min(0.5, (sat_frac - SATURATION_FRACTION_WARN) * 2.0)
        notes.append("high saturated-pixel fraction inside mask")
    if local_var > LOCAL_VARIANCE_WARN:
        penalty += min(0.5, (local_var - LOCAL_VARIANCE_WARN) * 1.5)
        notes.append("high luminance variance inside mask")
    score = max(0.0, 1.0 - penalty)
    return ScoreComponent(
        name="artifact_heuristic",
        value=score,
        available=True,
        note="; ".join(notes) if notes else "no strong artifact signals",
    )


def _check_output_valid(image: np.ndarray) -> tuple[bool, str | None]:
    try:
        validate_image(image)
    except ValueError as exc:
        return False, str(exc)
    if image.size == 0:
        return False, "empty output"
    return True, None


def _check_finite_pixels(image: np.ndarray) -> tuple[bool, str | None]:
    if not np.isfinite(image).all():
        return False, "non-finite pixel values"
    return True, None


def score_inpaint_candidate(
    original: np.ndarray,
    mask: np.ndarray,
    candidate_id: str,
    edited: np.ndarray,
) -> ScoredInpaintCandidate:
    """Score one inpainting candidate with transparent deterministic signals."""
    original = validate_image(original)
    mask = validate_mask(mask, image=original)
    reasons: list[str] = []
    components: dict[str, ScoreComponent] = {}

    valid, valid_note = _check_output_valid(edited)
    components["output_valid"] = ScoreComponent(
        name="output_valid",
        value=1.0 if valid else 0.0,
        available=True,
        note=valid_note,
    )
    if not valid:
        return ScoredInpaintCandidate(
            candidate_id=candidate_id,
            score=0.0,
            reasons=("invalid output image",),
            rejected=True,
            rejection_reason=valid_note or "invalid output",
            validity_status="rejected",
            components=components,
            output_hash=None,
        )

    edited = validate_image(edited)
    output_hash = sha256_bytes(edited.tobytes())

    finite_ok, finite_note = _check_finite_pixels(edited)
    components["finite_pixels"] = ScoreComponent(
        name="finite_pixels",
        value=1.0 if finite_ok else 0.0,
        available=True,
        note=finite_note,
    )
    if not finite_ok:
        return ScoredInpaintCandidate(
            candidate_id=candidate_id,
            score=0.0,
            reasons=("non-finite pixels",),
            rejected=True,
            rejection_reason=finite_note,
            validity_status="rejected",
            components=components,
            output_hash=output_hash,
        )

    dims_ok = edited.shape == original.shape
    components["correct_dimensions"] = ScoreComponent(
        name="correct_dimensions",
        value=1.0 if dims_ok else 0.0,
        available=True,
        note=None if dims_ok else f"expected {original.shape}, got {edited.shape}",
    )
    if not dims_ok:
        return ScoredInpaintCandidate(
            candidate_id=candidate_id,
            score=0.0,
            reasons=("dimension mismatch",),
            rejected=True,
            rejection_reason=components["correct_dimensions"].note,
            validity_status="rejected",
            components=components,
            output_hash=output_hash,
        )

    preservation = outside_mask_preservation_score(original, edited, mask)
    if preservation.value is None:
        components["preservation_outside_mask"] = ScoreComponent(
            name="preservation_outside_mask",
            value=None,
            available=False,
            note=preservation.note,
        )
    else:
        components["preservation_outside_mask"] = ScoreComponent(
            name="preservation_outside_mask",
            value=float(preservation.value),
            available=True,
            note=preservation.note,
        )
        if preservation.value < 0.85:
            reasons.append("low outside-mask preservation")

    components["mask_change_magnitude"] = _inside_mask_change_score(original, edited, mask)
    components["artifact_heuristic"] = _artifact_heuristic_score(edited, mask)

    active_weights = {
        key: weight
        for key, weight in SIGNAL_WEIGHTS.items()
        if key in components and components[key].available and components[key].value is not None
    }
    weight_sum = sum(active_weights.values())
    if weight_sum == 0:
        score = 0.0
        reasons.append("no available scoring signals")
    else:
        score = sum(
            components[key].value * active_weights[key]  # type: ignore[operator]
            for key in active_weights
        ) / weight_sum

    unavailable = [
        name
        for name, comp in components.items()
        if not comp.available
    ]
    if unavailable:
        reasons.append(f"unavailable signals omitted: {', '.join(unavailable)}")

    score = float(min(1.0, max(0.0, score)))
    if not reasons:
        reasons.append("heuristic aggregate candidate score")

    return ScoredInpaintCandidate(
        candidate_id=candidate_id,
        score=score,
        reasons=tuple(reasons),
        validity_status="valid",
        components=components,
        output_hash=output_hash,
    )


def rank_inpaint_candidates(
    original: np.ndarray,
    mask: np.ndarray,
    candidates: list[tuple[str, np.ndarray]],
) -> InpaintCandidateRanking:
    """Rank inpainting candidates; deterministic tie-break by candidate id."""
    scored = [
        score_inpaint_candidate(original, mask, candidate_id, edited)
        for candidate_id, edited in candidates
    ]
    rejected = tuple(c for c in scored if c.rejected)
    valid = [c for c in scored if not c.rejected]
    if not valid:
        raise ValueError("no valid inpainting candidates after scoring")
    valid.sort(key=lambda c: (-c.score, c.candidate_id))
    return InpaintCandidateRanking(
        selected=valid[0],
        ordered=tuple(valid),
        rejected=rejected,
    )
