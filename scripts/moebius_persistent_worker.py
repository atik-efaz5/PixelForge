#!/usr/bin/env python3
"""Long-lived Moebius inpainting worker for the pixelforge-moebius environment.

Reads newline-delimited JSON commands on stdin and writes one JSON response per
line on stdout. Loads Moebius once and keeps weights resident until shutdown.

Protocol
--------
Startup (stdout)::

    {"event": "ready", "model": "...", "load_ms": 5692.1}

Request::

    {"id": "1", "cmd": "inpaint", "image": "/path/img.png", "mask": "/path/mask.png", "out": "/path/out.png"}

Response::

    {"id": "1", "ok": true, "model": "...", "latency_ms": 26836, "memory_mb": 4504, "metadata": {}}

Ping::

    {"id": "2", "cmd": "ping"}

Shutdown::

    {"id": "3", "cmd": "shutdown"}
"""

from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image

from models.registry import get_adapter, reset_registry
from models.types import pil_rgb_to_array, validate_mask


def _emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stdout.flush()


@contextmanager
def _upstream_stdout_to_stderr():
    """Keep worker protocol lines on stdout; send upstream prints to stderr."""
    previous = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = previous


def _load_adapter():
    reset_registry()
    adapter = get_adapter("moebius")
    t0 = time.perf_counter()
    adapter.load()
    load_ms = (time.perf_counter() - t0) * 1000.0
    return adapter, load_ms


def _inpaint(adapter, image_path: Path, mask_path: Path, out_path: Path) -> dict:
    image = pil_rgb_to_array(Image.open(image_path).convert("RGB"))
    mask_arr = np.asarray(Image.open(mask_path).convert("L"))
    mask = validate_mask(mask_arr >= 128, image=image)

    t0 = time.perf_counter()
    result = adapter.infer(image, mask)
    infer_ms = (time.perf_counter() - t0) * 1000.0

    Image.fromarray(result.result).save(out_path)
    return {
        "model": result.model,
        "backend": result.backend.value,
        "latency_ms": result.latency_ms or round(infer_ms, 3),
        "infer_ms": round(infer_ms, 3),
        "memory_mb": result.memory_mb,
        "metadata": result.metadata,
        "output_path": str(out_path),
    }


def main() -> int:
    with _upstream_stdout_to_stderr():
        adapter, load_ms = _load_adapter()
    _emit(
        {
            "event": "ready",
            "model": adapter.model_name,
            "load_ms": round(load_ms, 3),
        }
    )

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as exc:
            _emit({"ok": False, "error": f"invalid json: {exc}"})
            continue

        request_id = msg.get("id")
        cmd = str(msg.get("cmd", "")).lower()

        if cmd == "shutdown":
            try:
                adapter.unload()
            finally:
                _emit({"id": request_id, "ok": True, "event": "shutdown"})
            return 0

        if cmd == "ping":
            _emit(
                {
                    "id": request_id,
                    "ok": True,
                    "loaded": adapter.is_loaded,
                    "model": adapter.model_name,
                }
            )
            continue

        if cmd == "inpaint":
            try:
                image_path = Path(str(msg["image"]))
                mask_path = Path(str(msg["mask"]))
                out_path = Path(str(msg["out"]))
                payload = _inpaint(adapter, image_path, mask_path, out_path)
                _emit({"id": request_id, "ok": True, **payload})
            except Exception as exc:
                _emit({"id": request_id, "ok": False, "error": str(exc)})
            continue

        _emit({"id": request_id, "ok": False, "error": f"unknown cmd: {cmd}"})

    try:
        adapter.unload()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
