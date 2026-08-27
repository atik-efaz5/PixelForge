"""Deterministic seed derivation for multi-candidate inpainting.

Each candidate receives a distinct derived seed from a base seed. Adapters set
per-call RNG state from the derived seed; global random state is not relied on.
"""

from __future__ import annotations

DEFAULT_BASE_SEED = 0
SEED_STRIDE = 10007

ALLOWED_CANDIDATE_COUNTS = frozenset({1, 2})


def validate_candidate_count(count: int) -> int:
    """Validate ``candidate_count`` is 1 or 2."""
    if not isinstance(count, int):
        raise TypeError(f"candidate_count must be int, got {type(count).__name__}")
    if count not in ALLOWED_CANDIDATE_COUNTS:
        raise ValueError(
            f"candidate_count must be 1 or 2, got {count}. "
            "Large candidate sets are not supported in this phase."
        )
    return count


def derive_candidate_seeds(base_seed: int, count: int) -> list[int]:
    """Derive ``count`` deterministic seeds from ``base_seed``.

  Strategy: ``base_seed + index * SEED_STRIDE`` for index in ``0 .. count-1``.
  ``SEED_STRIDE`` is a large prime to reduce accidental collision with nearby
  base seeds while remaining reproducible.
    """
    validate_candidate_count(count)
    if not isinstance(base_seed, int):
        raise TypeError(f"base_seed must be int, got {type(base_seed).__name__}")
    return [int(base_seed + index * SEED_STRIDE) for index in range(count)]


def resolve_base_seed(params_seed: int | None) -> int:
    """Use explicit request seed when provided, otherwise the default base."""
    return DEFAULT_BASE_SEED if params_seed is None else int(params_seed)
