#!/usr/bin/env python3
"""Run Moebius inpainting in the pixelforge-moebius environment.

Invoked by the FastAPI service when the backend process cannot import Moebius
dependencies directly. Reads image/mask paths, writes result PNG, prints JSON
metadata on the last stdout line.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from models.registry import get_adapter, reset_registry
from models.types import pil_rgb_to_array, validate_mask


def main() -> int:
    if len(sys.argv) != 4:
        print("usage: isolated_inpaint_worker.py <image.png> <mask.png> <out.png>", file=sys.stderr)
        return 2

    image_path = Path(sys.argv[1])
    mask_path = Path(sys.argv[2])
    out_path = Path(sys.argv[3])

    image = pil_rgb_to_array(Image.open(image_path).convert("RGB"))
    mask_arr = np.asarray(Image.open(mask_path).convert("L"))
    mask = validate_mask(mask_arr >= 128, image=image)

    reset_registry()
    adapter = get_adapter("moebius")
    adapter.load()
    try:
        result = adapter.infer(image, mask)
    finally:
        adapter.unload()

    Image.fromarray(result.result).save(out_path)
    meta = {
        "model": result.model,
        "backend": result.backend.value,
        "latency_ms": result.latency_ms,
        "memory_mb": result.memory_mb,
        "metadata": result.metadata,
    }
    print(json.dumps(meta))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
