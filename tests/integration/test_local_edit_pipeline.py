"""Integration test: real SAM 2 → Moebius object-removal pipeline.

Not collected by ``python -m unittest discover -s tests/unit``. Run explicitly:

    PIXELFORGE_RUN_INTEGRATION=1 python -m unittest tests.integration.test_local_edit_pipeline -v

Requires checkpoints, ``pixelforge-sam2-v2`` / ``pixelforge-moebius`` envs, and MPS.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

import numpy as np

from models.registry import reset_registry
from models.types import validate_image
from pipelines.orchestration.image_edit_pipeline import ImageEditPipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "integration"
TEST_IMAGE = PROJECT_ROOT / "tests" / "fixtures" / "edit_pipeline_input.png"

# Disc center from smoke test synthetic image (if fixture missing).
_DEFAULT_POINT = (160, 256)


def _make_fixture_image(path: Path) -> np.ndarray:
    """Deterministic RGB image matching smoke-test geometry."""
    h, w = 512, 512
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    bg_r = 30 + 60 * (yy / (h - 1))
    bg_g = 45 + 50 * (yy / (h - 1))
    bg_b = 70 + 40 * (xx / (w - 1))
    texture = 6.0 * np.sin(xx / 23.0) * np.cos(yy / 31.0)
    img = np.stack([bg_r + texture, bg_g + texture, bg_b + texture], axis=-1)
    cx, cy = 160, 256
    radius = 72
    disc = ((xx - cx) ** 2 + (yy - cy) ** 2) <= float(radius) ** 2
    shade = 1.0 - 0.25 * (((xx - cx) ** 2 + (yy - cy) ** 2) / float(radius) ** 2)
    img[disc] = np.stack([225 * shade, 140 * shade, 55 * shade], axis=-1)[disc]
    arr = np.clip(img, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    from PIL import Image

    Image.fromarray(arr, mode="RGB").save(path)
    return validate_image(arr)


def _load_test_image() -> np.ndarray:
    if TEST_IMAGE.is_file():
        from PIL import Image

        return validate_image(np.asarray(Image.open(TEST_IMAGE).convert("RGB")))
    return _make_fixture_image(TEST_IMAGE)


@unittest.skipUnless(
    os.environ.get("PIXELFORGE_RUN_INTEGRATION") == "1",
    "set PIXELFORGE_RUN_INTEGRATION=1 to run real SAM2→Moebius integration",
)
class TestLocalEditPipeline(unittest.TestCase):
    def setUp(self) -> None:
        reset_registry()

    def tearDown(self) -> None:
        reset_registry()

    def test_sam2_moebius_remove_object(self) -> None:
        image = _load_test_image()
        h, w = image.shape[:2]
        x, y = _DEFAULT_POINT
        pipeline = ImageEditPipeline()
        result = pipeline.remove_object(image, x, y, backend="moebius")

        self.assertEqual(result.result.shape, (h, w, 3))
        self.assertEqual(result.mask.shape, (h, w))
        self.assertTrue(np.isfinite(result.result).all())
        self.assertTrue(result.mask.any())
        self.assertGreater(result.latency.total_ms, 0)

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        from PIL import Image

        Image.fromarray(result.result, mode="RGB").save(
            OUTPUT_DIR / "edit_pipeline_result.png"
        )
        Image.fromarray((result.mask.astype(np.uint8) * 255), mode="L").save(
            OUTPUT_DIR / "edit_pipeline_mask.png"
        )


if __name__ == "__main__":
    unittest.main()
