"""Deterministic MVP benchmark cases (synthetic 512×512 scenes)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from evaluation.reproducibility import sha256_array
from models.types import validate_image, validate_mask

IMG_SIZE = 512
SEED = 16001  # fixed; image generation uses closed-form math only (no RNG).


@dataclass(frozen=True)
class BenchmarkCase:
    """One reproducible benchmark scene."""

    case_id: str
    description: str
    point_xy: tuple[int, int]
    text_prompt: str
    expected_region: str
    reference_mask: np.ndarray | None = None

    def to_dict(self, *, image: np.ndarray) -> dict[str, Any]:
        ref_hash = None
        if self.reference_mask is not None:
            ref_hash = sha256_array(self.reference_mask)["image_sha256"]
        return {
            "case_id": self.case_id,
            "description": self.description,
            "resolution": f"{IMG_SIZE}x{IMG_SIZE}",
            "point_xy": list(self.point_xy),
            "text_prompt": self.text_prompt,
            "expected_region": self.expected_region,
            "seed": SEED,
            "image_sha256": sha256_array(image)["image_sha256"],
            "reference_mask_sha256": ref_hash,
            "has_reference_mask": self.reference_mask is not None,
        }


def _grid() -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE].astype(np.float64)
    return yy, xx


def _disc_mask(cx: int, cy: int, radius: int) -> np.ndarray:
    yy, xx = _grid()
    return validate_mask(((xx - cx) ** 2 + (yy - cy) ** 2) <= float(radius) ** 2)


def make_case1_isolated_disc() -> tuple[np.ndarray, BenchmarkCase]:
    """Single orange disc on a smooth gradient background."""
    yy, xx = _grid()
    bg_r = 28 + 55 * (yy / (IMG_SIZE - 1))
    bg_g = 40 + 48 * (yy / (IMG_SIZE - 1))
    bg_b = 62 + 38 * (xx / (IMG_SIZE - 1))
    img = np.stack([bg_r, bg_g, bg_b], axis=-1)

    cx, cy, radius = 320, 256, 70
    disc = _disc_mask(cx, cy, radius)
    shade = 1.0 - 0.22 * (
        ((xx - cx) ** 2 + (yy - cy) ** 2) / float(radius) ** 2
    )
    img[disc] = np.stack([228 * shade, 132 * shade, 48 * shade], axis=-1)[disc]

    image = validate_image(np.clip(img, 0, 255).astype(np.uint8))
    case = BenchmarkCase(
        case_id="case1_isolated_disc",
        description="Simple isolated orange disc on gradient background",
        point_xy=(cx, cy),
        text_prompt="orange circle",
        expected_region="circular disc, ~7% of frame",
        reference_mask=_disc_mask(cx, cy, radius),
    )
    return image, case


def make_case2_adjacent_objects() -> tuple[np.ndarray, BenchmarkCase]:
    """Red and blue squares touching along a vertical edge."""
    yy, xx = _grid()
    img = np.stack(
        [
            24 + 40 * (yy / (IMG_SIZE - 1)),
            30 + 35 * (xx / (IMG_SIZE - 1)),
            36 + 30 * (yy / (IMG_SIZE - 1)),
        ],
        axis=-1,
    )

    red = (slice(200, 280), slice(120, 200))
    blue = (slice(200, 280), slice(200, 280))
    img[red] = np.array([210, 55, 55], dtype=np.float64)
    img[blue] = np.array([55, 95, 220], dtype=np.float64)

    image = validate_image(np.clip(img, 0, 255).astype(np.uint8))
    case = BenchmarkCase(
        case_id="case2_adjacent_objects",
        description="Red square touching blue square — tests boundary ambiguity",
        point_xy=(160, 240),
        text_prompt="red square",
        expected_region="left red square, not the blue neighbor",
    )
    return image, case


def make_case3_textured_background() -> tuple[np.ndarray, BenchmarkCase]:
    """Green ellipse on a high-frequency sinusoidal texture."""
    yy, xx = _grid()
    texture = (
        18.0 * np.sin(xx / 11.0) * np.cos(yy / 13.0)
        + 12.0 * np.sin(xx / 37.0 + yy / 29.0)
    )
    img = np.stack(
        [
            48 + texture + 10 * (yy / (IMG_SIZE - 1)),
            42 + texture + 8 * (xx / (IMG_SIZE - 1)),
            38 + texture,
        ],
        axis=-1,
    )

    cx, cy, rx, ry = 256, 256, 90, 55
    ellipse = ((xx - cx) / float(rx)) ** 2 + ((yy - cy) / float(ry)) ** 2 <= 1.0
    img[ellipse] = np.stack(
        [
            70 + 12 * (yy[ellipse] / (IMG_SIZE - 1)),
            185 + 8 * np.sin(xx[ellipse] / 19.0),
            78 + 6 * (xx[ellipse] / (IMG_SIZE - 1)),
        ],
        axis=-1,
    )

    image = validate_image(np.clip(img, 0, 255).astype(np.uint8))
    case = BenchmarkCase(
        case_id="case3_textured_background",
        description="Green ellipse on busy sinusoidal background",
        point_xy=(cx, cy),
        text_prompt="green ellipse",
        expected_region="central green ellipse over textured field",
    )
    return image, case


def all_cases() -> list[tuple[np.ndarray, BenchmarkCase]]:
    return [
        make_case1_isolated_disc(),
        make_case2_adjacent_objects(),
        make_case3_textured_background(),
    ]
