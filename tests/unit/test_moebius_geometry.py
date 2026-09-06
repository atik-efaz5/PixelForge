"""Unit tests for Moebius crop / 512 composite geometry."""

from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from models.moebius_geometry import (
    _max_pool_mask,
    prepare_moebius_work,
    restore_moebius_result,
    try_uniform_neighbor_fill,
)


class TestMoebiusGeometry(unittest.TestCase):
    def test_local_mask_uses_crop_not_full_frame(self) -> None:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        image[..., 0] = 10
        image[..., 1] = 20
        image[..., 2] = 30
        image[0, 0] = (1, 2, 3)
        image[200, 400] = (9, 8, 7)
        mask = np.zeros((480, 640), dtype=bool)
        mask[100:180, 200:360] = True

        work_image, work_mask, geom = prepare_moebius_work(image, mask)

        self.assertFalse(geom.skipped)
        self.assertEqual(work_image.shape, (512, 512, 3))
        self.assertEqual(work_mask.shape, (512, 512))
        self.assertEqual(geom.work_size, 512)
        self.assertEqual(geom.orig_h, 480)
        self.assertEqual(geom.orig_w, 640)
        self.assertLess(geom.crop_h, 480)
        self.assertLess(geom.crop_w, 640)
        self.assertLessEqual(geom.crop_y0, 100)
        self.assertGreaterEqual(geom.crop_y0 + geom.crop_h, 180)
        self.assertLessEqual(geom.crop_x0, 200)
        self.assertGreaterEqual(geom.crop_x0 + geom.crop_w, 360)
        self.assertTrue(bool(work_mask.any()))

        generated = np.full((512, 512, 3), 200, dtype=np.uint8)
        out = restore_moebius_result(image, mask, generated, geom)
        self.assertEqual(out.shape, (480, 640, 3))
        np.testing.assert_array_equal(out[0, 0], image[0, 0])
        np.testing.assert_array_equal(out[140, 280], generated[0, 0])
        np.testing.assert_allclose(out[99, 280], generated[0, 0], atol=8)
        self.assertFalse(np.array_equal(out[140, 280], image[140, 280]))

    def test_max_pool_keeps_pixels_nearest_would_drop(self) -> None:
        src = np.zeros((3, 3), dtype=bool)
        src[1, 1] = True
        nearest = (
            np.asarray(
                Image.fromarray((src.astype(np.uint8) * 255), mode="L").resize(
                    (2, 2), resample=Image.Resampling.NEAREST
                )
            )
            > 0
        )
        pooled = _max_pool_mask(src, 2, 2)
        self.assertFalse(bool(nearest.any()))
        self.assertTrue(bool(pooled.any()))

        image = np.zeros((800, 800, 3), dtype=np.uint8)
        image[:] = (10, 20, 30)
        mask = np.zeros((800, 800), dtype=bool)
        mask[50:650, 50:650] = True
        mask[1, 1] = True
        work_image, work_mask, geom = prepare_moebius_work(image, mask)
        self.assertFalse(geom.skipped)
        self.assertEqual(work_image.shape, (512, 512, 3))
        self.assertGreater(geom.crop_h, 512)
        self.assertTrue(bool(work_mask.any()))
        mapped_y = min((1 * geom.scaled_h) // geom.crop_h, geom.scaled_h - 1)
        mapped_x = min((1 * geom.scaled_w) // geom.crop_w, geom.scaled_w - 1)
        self.assertTrue(
            bool(work_mask[geom.pad_top + mapped_y, geom.pad_left + mapped_x])
        )

    def test_512_square_preserves_far_pixels(self) -> None:
        rng = np.random.default_rng(0)
        image = rng.integers(0, 256, size=(512, 512, 3), dtype=np.uint8)
        mask = np.zeros((512, 512), dtype=bool)
        mask[40:80, 40:80] = True

        work_image, work_mask, geom = prepare_moebius_work(image, mask)

        self.assertTrue(geom.skipped)
        self.assertEqual(work_image.shape, (512, 512, 3))
        np.testing.assert_array_equal(work_image, image)
        self.assertGreater(int(work_mask.sum()), int(mask.sum()))
        np.testing.assert_array_equal(work_mask[40:80, 40:80], mask[40:80, 40:80])

        generated = np.full((512, 512, 3), 255, dtype=np.uint8)
        out = restore_moebius_result(image, mask, generated, geom)
        self.assertEqual(out.shape, (512, 512, 3))
        np.testing.assert_array_equal(out[400, 400], image[400, 400])
        np.testing.assert_array_equal(out[60, 60], generated[60, 60])
        np.testing.assert_allclose(out[39, 60], generated[39, 60], atol=8)
        self.assertFalse(np.array_equal(out[60, 60], image[60, 60]))

    def test_uniform_page_fills_hole_with_neighbor_color(self) -> None:
        page = np.full((96, 96, 3), 248, dtype=np.uint8)
        page[24:72, 24:72] = (30, 60, 90)
        mask = np.zeros((96, 96), dtype=bool)
        mask[24:72, 24:72] = True

        filled = try_uniform_neighbor_fill(page, mask)
        self.assertIsNotNone(filled)
        assert filled is not None
        np.testing.assert_allclose(filled[48, 48], (248, 248, 248), atol=2)
        np.testing.assert_array_equal(filled[8, 8], page[8, 8])
        self.assertFalse(np.array_equal(filled[48, 48], page[48, 48]))

    def test_noisy_neighbors_skip_solid_fill(self) -> None:
        rng = np.random.default_rng(1)
        image = rng.integers(0, 256, size=(96, 96, 3), dtype=np.uint8)
        mask = np.zeros((96, 96), dtype=bool)
        mask[24:72, 24:72] = True
        self.assertIsNone(try_uniform_neighbor_fill(image, mask))

    def test_striped_neighbors_skip_solid_fill(self) -> None:
        image = np.zeros((96, 96, 3), dtype=np.uint8)
        image[:, ::2] = (210, 200, 190)
        image[:, 1::2] = (40, 80, 30)
        mask = np.zeros((96, 96), dtype=bool)
        mask[24:72, 24:72] = True
        self.assertIsNone(try_uniform_neighbor_fill(image, mask))


if __name__ == "__main__":
    unittest.main()
