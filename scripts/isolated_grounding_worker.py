#!/usr/bin/env python3
"""Run Grounding DINO in an isolated environment and print JSON detections."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from models.registry import get_adapter, reset_registry
from models.types import pil_rgb_to_array


def main() -> int:
    if len(sys.argv) != 3:
        print(
            "usage: isolated_grounding_worker.py <image.png> <prompt>",
            file=sys.stderr,
        )
        return 2

    image_path = Path(sys.argv[1])
    prompt = sys.argv[2]
    from PIL import Image

    image = pil_rgb_to_array(Image.open(image_path).convert("RGB"))
    reset_registry()
    adapter = get_adapter("grounding_dino")
    adapter.load()
    try:
        result = adapter.infer(image, prompt)
    finally:
        adapter.unload()

    payload = {
        "model": result.model,
        "prompt": result.prompt,
        "backend": result.backend.value,
        "metadata": result.metadata,
        "detections": [
            {
                "x1": d.x1,
                "y1": d.y1,
                "x2": d.x2,
                "y2": d.y2,
                "confidence": d.confidence,
                "label": d.label,
            }
            for d in result.detections
        ],
    }
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
