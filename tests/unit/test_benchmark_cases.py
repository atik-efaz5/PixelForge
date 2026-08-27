"""Unit tests for deterministic MVP benchmark cases."""

from __future__ import annotations

import unittest

from evaluation.benchmarks.cases import (
    IMG_SIZE,
    SEED,
    all_cases,
    make_case1_isolated_disc,
)
from evaluation.reproducibility import sha256_array


class TestBenchmarkCases(unittest.TestCase):
    def test_all_cases_count(self) -> None:
        self.assertEqual(len(all_cases()), 3)

    def test_resolution(self) -> None:
        for image, _case in all_cases():
            self.assertEqual(image.shape, (IMG_SIZE, IMG_SIZE, 3))

    def test_case1_reference_mask_matches_disc(self) -> None:
        _image, case = make_case1_isolated_disc()
        self.assertIsNotNone(case.reference_mask)
        assert case.reference_mask is not None
        cx, cy = case.point_xy
        self.assertTrue(case.reference_mask[cy, cx])

    def test_deterministic_image_hash(self) -> None:
        img1, _ = make_case1_isolated_disc()
        img2, _ = make_case1_isolated_disc()
        h1 = sha256_array(img1)["image_sha256"]
        h2 = sha256_array(img2)["image_sha256"]
        self.assertEqual(h1, h2)

    def test_seed_constant(self) -> None:
        self.assertEqual(SEED, 16001)


if __name__ == "__main__":
    unittest.main()
