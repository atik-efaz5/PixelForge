"""Unit tests for reproducibility helpers."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from evaluation.reproducibility import (
    build_reproducibility_record,
    sha256_array,
    sha256_bytes,
    sha256_file,
)
from evaluation.types import ExperimentConfig


class TestReproducibility(unittest.TestCase):
    def test_sha256_bytes_known(self) -> None:
        digest = sha256_bytes(b"pixelforge")
        expected = hashlib.sha256(b"pixelforge").hexdigest()
        self.assertEqual(digest, expected)

    def test_sha256_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.bin"
            path.write_bytes(b"abc")
            self.assertEqual(sha256_file(path), sha256_bytes(b"abc"))

    def test_sha256_array_deterministic(self) -> None:
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        mask = np.zeros((4, 4), dtype=bool)
        first = sha256_array(img, mask)
        second = sha256_array(img, mask)
        self.assertEqual(first, second)

    def test_build_reproducibility_record_no_secrets(self) -> None:
        config = ExperimentConfig(
            experiment_id="exp-1",
            model="moebius",
            backend="LOCAL_MPS",
            operation="inpaint",
            environment="pixelforge-moebius",
            model_commit="abc123",
            seed=7,
            parameters={"num_steps": 20},
        )
        record = build_reproducibility_record(
            config=config,
            device="mps",
            image_hash="img",
            mask_hash="mask",
            output_hash=None,
        )
        self.assertEqual(record["model"], "moebius")
        self.assertEqual(record["hashes"]["image_sha256"], "img")
        self.assertNotIn("api_key", record)
        self.assertNotIn("password", record)


if __name__ == "__main__":
    unittest.main()
