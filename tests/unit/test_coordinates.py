"""Unit tests mirroring frontend coordinate mapping (letterbox + DPR-safe layout)."""

from __future__ import annotations

import math
import unittest
from dataclasses import dataclass


@dataclass
class DisplayLayout:
    scale: float
    offset_x: float
    offset_y: float
    rendered_width: float
    rendered_height: float


def compute_display_layout(
    container_w: float,
    container_h: float,
    image_w: float,
    image_h: float,
) -> DisplayLayout:
    scale = min(container_w / image_w, container_h / image_h)
    rendered_width = image_w * scale
    rendered_height = image_h * scale
    return DisplayLayout(
        scale=scale,
        offset_x=(container_w - rendered_width) / 2,
        offset_y=(container_h - rendered_height) / 2,
        rendered_width=rendered_width,
        rendered_height=rendered_height,
    )


def pointer_to_image_coords(
    pointer_x: float,
    pointer_y: float,
    layout: DisplayLayout,
    image_w: int,
    image_h: int,
) -> tuple[int, int] | None:
    rel_x = pointer_x - layout.offset_x
    rel_y = pointer_y - layout.offset_y
    if (
        rel_x < 0
        or rel_y < 0
        or rel_x > layout.rendered_width
        or rel_y > layout.rendered_height
    ):
        return None
    x = min(max(0, math.floor(rel_x / layout.scale)), image_w - 1)
    y = min(max(0, math.floor(rel_y / layout.scale)), image_h - 1)
    return x, y


class TestCoordinates(unittest.TestCase):
    def test_letterbox_center_maps_to_image_center(self) -> None:
        layout = compute_display_layout(800, 600, 400, 200)
        coords = pointer_to_image_coords(
            layout.offset_x + layout.rendered_width / 2,
            layout.offset_y + layout.rendered_height / 2,
            layout,
            400,
            200,
        )
        self.assertIsNotNone(coords)
        assert coords is not None
        self.assertEqual(coords, (200, 100))

    def test_click_outside_image_returns_none(self) -> None:
        layout = compute_display_layout(800, 600, 400, 200)
        self.assertIsNone(pointer_to_image_coords(0, 0, layout, 400, 200))
        self.assertIsNone(
            pointer_to_image_coords(
                layout.offset_x + layout.rendered_width + 1,
                layout.offset_y,
                layout,
                400,
                200,
            )
        )

    def test_top_left_corner(self) -> None:
        layout = compute_display_layout(640, 480, 320, 240)
        coords = pointer_to_image_coords(
            layout.offset_x,
            layout.offset_y,
            layout,
            320,
            240,
        )
        self.assertEqual(coords, (0, 0))

    def test_portrait_in_landscape_container(self) -> None:
        layout = compute_display_layout(1000, 500, 200, 800)
        self.assertGreater(layout.offset_x, 0)
        coords = pointer_to_image_coords(
            layout.offset_x,
            layout.offset_y,
            layout,
            200,
            800,
        )
        self.assertEqual(coords, (0, 0))

    def test_resize_changes_scale_but_corner_still_maps(self) -> None:
        small = compute_display_layout(400, 300, 100, 100)
        large = compute_display_layout(800, 600, 100, 100)
        self.assertNotEqual(small.scale, large.scale)
        for layout in (small, large):
            coords = pointer_to_image_coords(
                layout.offset_x,
                layout.offset_y,
                layout,
                100,
                100,
            )
            self.assertEqual(coords, (0, 0))


if __name__ == "__main__":
    unittest.main()
