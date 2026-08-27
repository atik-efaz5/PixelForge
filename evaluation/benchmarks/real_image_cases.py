"""Deterministic ~512×512 scene generators for Phase 20 real-image evaluation.

These are locally generated scenes (no external dataset downloads). They are
designed to resemble common editing categories while providing exact reference
masks where the target region is constructed analytically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from evaluation.reproducibility import sha256_array
from models.types import validate_image, validate_mask

IMG_SIZE = 512
SEED = 20001  # metadata only; scenes use closed-form math (no RNG).


@dataclass(frozen=True)
class RealImageCase:
    case_id: str
    category: str
    description: str
    point_xy: tuple[int, int]
    text_prompt: str
    target_object: str
    expected_region: str
    run_inpaint: bool = False
    reference_mask: np.ndarray | None = None

    def to_dict(self, *, image: np.ndarray) -> dict[str, Any]:
        ref_hash = None
        if self.reference_mask is not None:
            ref_hash = sha256_array(self.reference_mask)["image_sha256"]
        return {
            "case_id": self.case_id,
            "category": self.category,
            "description": self.description,
            "resolution": f"{IMG_SIZE}x{IMG_SIZE}",
            "point_xy": list(self.point_xy),
            "text_prompt": self.text_prompt,
            "target_object": self.target_object,
            "expected_region": self.expected_region,
            "run_inpaint": self.run_inpaint,
            "seed": SEED,
            "image_sha256": sha256_array(image)["image_sha256"],
            "reference_mask_sha256": ref_hash,
            "has_reference_mask": self.reference_mask is not None,
            "ground_truth_justification": (
                "Analytic mask for constructed target region in deterministic scene"
                if self.reference_mask is not None
                else None
            ),
        }


def _grid() -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:IMG_SIZE, 0:IMG_SIZE].astype(np.float64)
    return yy, xx


def _sky_ground(img: np.ndarray, horizon: int = 300) -> np.ndarray:
    yy, xx = _grid()
    sky = np.stack(
        [
            120 + 40 * (yy / max(horizon, 1)),
            170 + 35 * (yy / max(horizon, 1)),
            220 + 20 * (xx / (IMG_SIZE - 1)),
        ],
        axis=-1,
    )
    ground = np.stack(
        [
            55 + 25 * (yy / (IMG_SIZE - 1)),
            95 + 30 * (xx / (IMG_SIZE - 1)),
            48 + 15 * (yy / (IMG_SIZE - 1)),
        ],
        axis=-1,
    )
    out = img.astype(np.float64)
    out[yy < horizon] = sky[yy < horizon]
    out[yy >= horizon] = ground[yy >= horizon]
    return out


def make_real01_person() -> tuple[np.ndarray, RealImageCase]:
    """Person-like silhouette in a park scene."""
    img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float64)
    img = _sky_ground(img, horizon=310)

    yy, xx = _grid()
    # Simple path / shadow
    path = (yy > 360) & (np.abs(xx - 256) < 90)
    img[path] = np.array([92, 88, 72], dtype=np.float64)

    head = ((xx - 256) ** 2 + (yy - 170) ** 2) <= 38**2
    torso = (np.abs(xx - 256) <= 42) & (yy >= 205) & (yy <= 300)
    legs = (
        ((np.abs(xx - 232) <= 18) | (np.abs(xx - 280) <= 18))
        & (yy >= 300)
        & (yy <= 390)
    )
    person = head | torso | legs
    skin = np.array([224, 186, 152], dtype=np.float64)
    shirt = np.array([58, 108, 198], dtype=np.float64)
    pants = np.array([48, 52, 68], dtype=np.float64)
    img[head] = skin
    img[torso] = shirt
    img[legs] = pants

    image = validate_image(np.clip(img, 0, 255).astype(np.uint8))
    case = RealImageCase(
        case_id="real01_person",
        category="person",
        description="Stylized person silhouette on outdoor path",
        point_xy=(256, 240),
        text_prompt="blue shirt person",
        target_object="person",
        expected_region="head, torso, and legs of central figure",
        run_inpaint=True,
        reference_mask=validate_mask(person),
    )
    return image, case


def make_real02_vehicle() -> tuple[np.ndarray, RealImageCase]:
    """Red car on a road segment."""
    img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float64)
    img = _sky_ground(img, horizon=280)
    yy, xx = _grid()
    road = yy >= 280
    img[road] = np.array([58, 58, 62], dtype=np.float64)
    lane = (yy >= 350) & (yy <= 358)
    img[lane] = np.array([210, 180, 40], dtype=np.float64)

    body = (
        (xx >= 150)
        & (xx <= 360)
        & (yy >= 300)
        & (yy <= 370)
        & (((xx - 150) / 210) ** 2 + ((yy - 335) / 40) ** 2 <= 1.0)
    )
    cabin = (xx >= 220) & (xx <= 320) & (yy >= 305) & (yy <= 345)
    wheels = (
        ((xx - 190) ** 2 + (yy - 372) ** 2 <= 22**2)
        | ((xx - 320) ** 2 + (yy - 372) ** 2 <= 22**2)
    )
    car = body | cabin | wheels
    img[body | wheels] = np.array([198, 48, 42], dtype=np.float64)
    img[cabin] = np.array([170, 210, 225], dtype=np.float64)

    image = validate_image(np.clip(img, 0, 255).astype(np.uint8))
    case = RealImageCase(
        case_id="real02_vehicle",
        category="vehicle",
        description="Red car on gray road under sky",
        point_xy=(255, 335),
        text_prompt="red car",
        target_object="car",
        expected_region="red vehicle body, cabin, and wheels",
        run_inpaint=True,
        reference_mask=validate_mask(car),
    )
    return image, case


def make_real03_furniture() -> tuple[np.ndarray, RealImageCase]:
    """Chair in a simple indoor corner."""
    img = np.stack(
        [
            38 + 12 * (_grid()[0] / (IMG_SIZE - 1)),
            36 + 10 * (_grid()[1] / (IMG_SIZE - 1)),
            42 + 8 * (_grid()[0] / (IMG_SIZE - 1)),
        ],
        axis=-1,
    )
    yy, xx = _grid()
    floor = yy >= 380
    img[floor] = np.array([118, 92, 68], dtype=np.float64)

    seat = (xx >= 200) & (xx <= 310) & (yy >= 260) & (yy <= 290)
    back = (xx >= 200) & (xx <= 230) & (yy >= 150) & (yy <= 290)
    leg1 = (xx >= 205) & (xx <= 220) & (yy >= 290) & (yy <= 380)
    leg2 = (xx >= 290) & (xx <= 305) & (yy >= 290) & (yy <= 380)
    chair = seat | back | leg1 | leg2
    wood = np.array([142, 98, 52], dtype=np.float64)
    img[chair] = wood

    image = validate_image(np.clip(img, 0, 255).astype(np.uint8))
    case = RealImageCase(
        case_id="real03_furniture",
        category="furniture",
        description="Wooden chair in indoor corner",
        point_xy=(245, 270),
        text_prompt="wooden chair",
        target_object="chair",
        expected_region="chair seat, back, and legs",
        reference_mask=validate_mask(chair),
    )
    return image, case


def make_real04_outdoor() -> tuple[np.ndarray, RealImageCase]:
    """Tree on an outdoor field."""
    img = _sky_ground(np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float64), horizon=250)
    yy, xx = _grid()
    trunk = (np.abs(xx - 330) <= 22) & (yy >= 220) & (yy <= 390)
    foliage = ((xx - 330) ** 2 + (yy - 190) ** 2) <= 85**2
    tree = trunk | foliage
    img[trunk] = np.array([98, 62, 38], dtype=np.float64)
    img[foliage] = np.array([48, 138, 58], dtype=np.float64)

    image = validate_image(np.clip(img, 0, 255).astype(np.uint8))
    case = RealImageCase(
        case_id="real04_outdoor",
        category="outdoor_scene",
        description="Single tree on grassy field",
        point_xy=(330, 200),
        text_prompt="green tree",
        target_object="tree",
        expected_region="tree trunk and foliage",
        reference_mask=validate_mask(tree),
    )
    return image, case


def make_real05_textured() -> tuple[np.ndarray, RealImageCase]:
    """Mug on a brick-textured wall."""
    yy, xx = _grid()
    brick_r = 140 + 18 * np.sin(yy / 14.0) * np.cos(xx / 22.0)
    brick_g = 72 + 12 * np.sin(xx / 17.0 + yy / 19.0)
    brick_b = 58 + 10 * np.cos(yy / 23.0)
    img = np.stack([brick_r, brick_g, brick_b], axis=-1)

    table = yy >= 360
    img[table] = np.array([96, 78, 62], dtype=np.float64)

    mug_body = ((xx - 256) ** 2 + (yy - 310) ** 2) <= 48**2
    mug_handle = ((xx - 300) ** 2 + (yy - 310) ** 2) <= 20**2
    mug = mug_body | mug_handle
    img[mug] = np.array([238, 238, 242], dtype=np.float64)

    image = validate_image(np.clip(img, 0, 255).astype(np.uint8))
    case = RealImageCase(
        case_id="real05_textured",
        category="textured_background",
        description="White mug on table against brick-like texture",
        point_xy=(256, 310),
        text_prompt="white mug",
        target_object="mug",
        expected_region="white mug body and handle",
        reference_mask=validate_mask(mug),
    )
    return image, case


def make_real06_adjacent() -> tuple[np.ndarray, RealImageCase]:
    """Red ball touching a green ball."""
    yy, xx = _grid()
    img = np.stack(
        [
            32 + 20 * (yy / (IMG_SIZE - 1)),
            34 + 18 * (xx / (IMG_SIZE - 1)),
            40 + 16 * (yy / (IMG_SIZE - 1)),
        ],
        axis=-1,
    )
    red_ball = ((xx - 190) ** 2 + (yy - 256) ** 2) <= 62**2
    green_ball = ((xx - 290) ** 2 + (yy - 256) ** 2) <= 62**2
    shade = 1.0 - 0.15 * (
        ((xx - 190) ** 2 + (yy - 256) ** 2) / float(62**2)
    )
    img[red_ball] = np.stack(
        [210 * shade[red_ball], 48 * shade[red_ball], 44 * shade[red_ball]],
        axis=-1,
    )
    gshade = 1.0 - 0.15 * (
        ((xx - 290) ** 2 + (yy - 256) ** 2) / float(62**2)
    )
    img[green_ball] = np.stack(
        [48 * gshade[green_ball], 175 * gshade[green_ball], 72 * gshade[green_ball]],
        axis=-1,
    )

    image = validate_image(np.clip(img, 0, 255).astype(np.uint8))
    case = RealImageCase(
        case_id="real06_adjacent",
        category="adjacent_objects",
        description="Red ball touching green ball — selection ambiguity",
        point_xy=(190, 256),
        text_prompt="red ball",
        target_object="red ball",
        expected_region="left red ball only, not green neighbor",
        run_inpaint=True,
        reference_mask=validate_mask(red_ball),
    )
    return image, case


def make_real07_cluttered() -> tuple[np.ndarray, RealImageCase]:
    """Book near a pen on a desk."""
    yy, xx = _grid()
    img = np.stack(
        [
            48 + 14 * (yy / (IMG_SIZE - 1)),
            42 + 12 * (xx / (IMG_SIZE - 1)),
            38 + 10 * (yy / (IMG_SIZE - 1)),
        ],
        axis=-1,
    )
    desk = yy >= 300
    img[desk] = np.array([108, 82, 58], dtype=np.float64)

    book = (xx >= 170) & (xx <= 260) & (yy >= 240) & (yy <= 310)
    pen = (np.abs(xx - 310) < 8) & (yy >= 220) & (yy <= 320)
    img[book] = np.array([178, 42, 38], dtype=np.float64)
    img[pen] = np.array([42, 92, 198], dtype=np.float64)

    image = validate_image(np.clip(img, 0, 255).astype(np.uint8))
    case = RealImageCase(
        case_id="real07_cluttered",
        category="cluttered_scene",
        description="Red book beside blue pen on desk",
        point_xy=(215, 275),
        text_prompt="red book",
        target_object="book",
        expected_region="red book only, not the nearby pen",
        reference_mask=validate_mask(book),
    )
    return image, case


def all_cases() -> list[tuple[np.ndarray, RealImageCase]]:
    return [
        make_real01_person(),
        make_real02_vehicle(),
        make_real03_furniture(),
        make_real04_outdoor(),
        make_real05_textured(),
        make_real06_adjacent(),
        make_real07_cluttered(),
    ]


def inpaint_case_ids() -> frozenset[str]:
    return frozenset(case.case_id for _img, case in all_cases() if case.run_inpaint)
