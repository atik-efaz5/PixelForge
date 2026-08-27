"""Unit tests for candidate seed derivation."""

from __future__ import annotations

import unittest

from pipelines.candidate_seeds import (
    DEFAULT_BASE_SEED,
    SEED_STRIDE,
    derive_candidate_seeds,
    resolve_base_seed,
    validate_candidate_count,
)


class TestCandidateSeeds(unittest.TestCase):
    def test_validate_candidate_count_accepts_one_and_two(self) -> None:
        self.assertEqual(validate_candidate_count(1), 1)
        self.assertEqual(validate_candidate_count(2), 2)

    def test_validate_candidate_count_rejects_other_values(self) -> None:
        with self.assertRaises(ValueError):
            validate_candidate_count(0)
        with self.assertRaises(ValueError):
            validate_candidate_count(3)

    def test_derive_candidate_seeds_is_deterministic(self) -> None:
        first = derive_candidate_seeds(42, 2)
        second = derive_candidate_seeds(42, 2)
        self.assertEqual(first, second)
        self.assertEqual(first, [42, 42 + SEED_STRIDE])

    def test_derive_candidate_seeds_single(self) -> None:
        self.assertEqual(derive_candidate_seeds(7, 1), [7])

    def test_resolve_base_seed_defaults(self) -> None:
        self.assertEqual(resolve_base_seed(None), DEFAULT_BASE_SEED)
        self.assertEqual(resolve_base_seed(99), 99)


if __name__ == "__main__":
    unittest.main()
