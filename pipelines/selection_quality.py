"""Deterministic heuristic scoring for segmentation mask candidates.

This is NOT a learned quality model. Scores are relative rankings only and
must not be interpreted as calibrated probabilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from models.types import validate_mask

# Area ratio bounds (fraction of image pixels).
MIN_AREA_RATIO = 0.001
MAX_AREA_RATIO = 0.65
TINY_REJECT_RATIO = 0.0005
OVERSIZED_REJECT_RATIO = 0.75
SWEET_SPOT_LOW = 0.008
SWEET_SPOT_HIGH = 0.40

TIER_HIGH_MIN = 0.72
TIER_MEDIUM_MIN = 0.45


class ConfidenceTier(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


@dataclass(frozen=True)
class ScoredCandidate:
    candidate_id: str
    score: float
    tier: ConfidenceTier
    reasons: tuple[str, ...]
    rejected: bool = False
    rejection_reason: str | None = None
    mask_area_ratio: float = 0.0
    signals: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "score": round(self.score, 6),
            "tier": self.tier.value,
            "reasons": list(self.reasons),
            "rejected": self.rejected,
            "rejection_reason": self.rejection_reason,
            "mask_area_ratio": round(self.mask_area_ratio, 6),
            "signals": {k: round(v, 6) for k, v in self.signals.items()},
        }


@dataclass(frozen=True)
class SelectionRanking:
    selected: ScoredCandidate
    candidates: tuple[ScoredCandidate, ...]
    rejected: tuple[ScoredCandidate, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "selected": self.selected.to_dict(),
            "candidates": [c.to_dict() for c in self.candidates],
            "rejected": [c.to_dict() for c in self.rejected],
        }


def _mask_area_ratio(mask: np.ndarray) -> float:
    m = validate_mask(mask)
    return float(m.sum()) / float(m.size)


def _box_mask(h: int, w: int, x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    box = np.zeros((h, w), dtype=bool)
    xi1 = max(0, int(np.floor(x1)))
    yi1 = max(0, int(np.floor(y1)))
    xi2 = min(w, int(np.ceil(x2)))
    yi2 = min(h, int(np.ceil(y2)))
    if xi2 > xi1 and yi2 > yi1:
        box[yi1:yi2, xi1:xi2] = True
    return box


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(inter) / float(union)


def _largest_component_ratio(mask: np.ndarray) -> float:
    """Fraction of True pixels in the largest 4-connected component."""
    m = validate_mask(mask)
    h, w = m.shape
    if not m.any():
        return 0.0
    visited = np.zeros_like(m, dtype=bool)
    largest = 0
    for y in range(h):
        for x in range(w):
            if not m[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            visited[y, x] = True
            size = 0
            while stack:
                cy, cx = stack.pop()
                size += 1
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and m[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        stack.append((ny, nx))
            largest = max(largest, size)
    total = int(m.sum())
    return largest / total if total else 0.0


def _boundary_smoothness(mask: np.ndarray) -> float:
    """Compactness proxy in [0, 1]; higher ≈ smoother boundary."""
    m = validate_mask(mask)
    area = int(m.sum())
    if area == 0:
        return 0.0
    padded = np.pad(m, 1, mode="constant", constant_values=False)
    interior = padded[1:-1, 1:-1]
    neighbors = (
        padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
    )
    boundary = interior & (neighbors < 4)
    perimeter = max(int(boundary.sum()), 1)
    compactness = (4.0 * np.pi * area) / float(perimeter**2)
    return float(min(1.0, max(0.0, compactness)))


def _area_sweet_spot(area_ratio: float) -> float:
    if SWEET_SPOT_LOW <= area_ratio <= SWEET_SPOT_HIGH:
        return 1.0
    if area_ratio < SWEET_SPOT_LOW:
        return max(0.0, area_ratio / SWEET_SPOT_LOW)
    if area_ratio <= MAX_AREA_RATIO:
        span = MAX_AREA_RATIO - SWEET_SPOT_HIGH
        return max(0.0, 1.0 - (area_ratio - SWEET_SPOT_HIGH) / span)
    return 0.0


def _tier_from_score(score: float) -> ConfidenceTier:
    if score >= TIER_HIGH_MIN:
        return ConfidenceTier.HIGH
    if score >= TIER_MEDIUM_MIN:
        return ConfidenceTier.MEDIUM
    return ConfidenceTier.LOW


def score_mask_candidate(
    mask: np.ndarray,
    *,
    candidate_id: str,
    point_xy: tuple[int, int] | None = None,
    box_xyxy: tuple[float, float, float, float] | None = None,
    sam_confidence: float | None = None,
) -> ScoredCandidate:
    """Score one mask candidate with deterministic heuristics."""
    m = validate_mask(mask)
    h, w = m.shape
    area_ratio = _mask_area_ratio(m)
    reasons: list[str] = []

    if area_ratio < TINY_REJECT_RATIO:
        return ScoredCandidate(
            candidate_id=candidate_id,
            score=0.0,
            tier=ConfidenceTier.LOW,
            reasons=("mask area below minimum threshold",),
            rejected=True,
            rejection_reason="mask too small",
            mask_area_ratio=area_ratio,
        )
    if area_ratio > OVERSIZED_REJECT_RATIO:
        return ScoredCandidate(
            candidate_id=candidate_id,
            score=0.0,
            tier=ConfidenceTier.LOW,
            reasons=("mask area above maximum threshold",),
            rejected=True,
            rejection_reason="mask too large",
            mask_area_ratio=area_ratio,
        )
    if not m.any():
        return ScoredCandidate(
            candidate_id=candidate_id,
            score=0.0,
            tier=ConfidenceTier.LOW,
            reasons=("empty mask",),
            rejected=True,
            rejection_reason="empty mask",
            mask_area_ratio=0.0,
        )

    if point_xy is not None:
        px, py = point_xy
        if not (0 <= px < w and 0 <= py < h):
            return ScoredCandidate(
                candidate_id=candidate_id,
                score=0.0,
                tier=ConfidenceTier.LOW,
                reasons=("prompt point outside image",),
                rejected=True,
                rejection_reason="invalid prompt point",
                mask_area_ratio=area_ratio,
            )
        if not m[py, px]:
            return ScoredCandidate(
                candidate_id=candidate_id,
                score=0.0,
                tier=ConfidenceTier.LOW,
                reasons=("mask does not contain prompt point",),
                rejected=True,
                rejection_reason="point outside mask",
                mask_area_ratio=area_ratio,
            )
        reasons.append("contains prompt point")

    signals: dict[str, float] = {}
    signals["area_sweet_spot"] = _area_sweet_spot(area_ratio)
    signals["component_sanity"] = _largest_component_ratio(m)
    signals["boundary_smoothness"] = _boundary_smoothness(m)

    if point_xy is not None:
        signals["point_contains"] = 1.0

    if box_xyxy is not None:
        box = _box_mask(h, w, *box_xyxy)
        signals["box_iou"] = _mask_iou(m, box)
        inside = np.logical_and(m, box).sum()
        signals["box_containment"] = float(inside) / float(max(int(m.sum()), 1))
        reasons.append(f"box IoU {signals['box_iou']:.2f}")

    if sam_confidence is not None:
        signals["sam_confidence"] = float(min(1.0, max(0.0, sam_confidence)))

    weights = {
        "area_sweet_spot": 0.18,
        "component_sanity": 0.18,
        "boundary_smoothness": 0.12,
        "point_contains": 0.22,
        "box_iou": 0.20,
        "box_containment": 0.15,
        "sam_confidence": 0.15,
    }
    active = {k: v for k, v in signals.items() if k in weights}
    weight_sum = sum(weights[k] for k in active)
    score = sum(signals[k] * weights[k] for k in active) / weight_sum if weight_sum else 0.0

    if area_ratio < MIN_AREA_RATIO:
        score *= 0.5
        reasons.append("small mask penalty")
    if area_ratio > MAX_AREA_RATIO:
        score *= 0.4
        reasons.append("large mask penalty")
    if signals.get("component_sanity", 1.0) < 0.85:
        score *= 0.85
        reasons.append("fragmented mask penalty")

    score = float(min(1.0, max(0.0, score)))
    tier = _tier_from_score(score)
    if not reasons:
        reasons.append("heuristic aggregate score")

    return ScoredCandidate(
        candidate_id=candidate_id,
        score=score,
        tier=tier,
        reasons=tuple(reasons),
        mask_area_ratio=area_ratio,
        signals=signals,
    )


def rank_candidates(scored: list[ScoredCandidate]) -> SelectionRanking:
    """Pick the best non-rejected candidate; deterministic tie-break by id."""
    rejected = tuple(c for c in scored if c.rejected)
    valid = [c for c in scored if not c.rejected]
    if not valid:
        raise ValueError("no valid mask candidates after scoring")
    valid.sort(key=lambda c: (-c.score, c.candidate_id))
    selected = valid[0]
    return SelectionRanking(
        selected=selected,
        candidates=tuple(valid),
        rejected=rejected,
    )
