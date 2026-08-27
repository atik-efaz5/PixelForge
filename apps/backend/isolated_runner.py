"""Subprocess bridge for model inference in isolated conda environments."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image

from models.config import model_entry, project_root
from models.errors import ModelInferenceError, ModelLoadError
from models.types import (
    BackendType,
    BoundingBox,
    GroundingResult,
    InpaintParams,
    InpaintingResult,
    validate_image,
)

logger = logging.getLogger(__name__)

_INPAINT_WORKER = project_root() / "scripts" / "isolated_inpaint_worker.py"
_GROUNDING_WORKER = project_root() / "scripts" / "isolated_grounding_worker.py"
_SUBPROCESS_TIMEOUT_SEC = float(os.environ.get("PIXELFORGE_SUBPROCESS_TIMEOUT", "600"))


def _run_isolated(cmd: list[str], *, label: str) -> subprocess.CompletedProcess[str]:
    """Run an isolated worker subprocess with timeout and captured output."""
    try:
        return subprocess.run(
            cmd,
            cwd=str(project_root()),
            capture_output=True,
            text=True,
            check=False,
            timeout=_SUBPROCESS_TIMEOUT_SEC,
        )
    except subprocess.TimeoutExpired as exc:
        raise ModelInferenceError(
            f"{label} timed out after {_SUBPROCESS_TIMEOUT_SEC:.0f}s"
        ) from exc


def _env_python(env_name: str, *, override_var: str) -> Path:
    override = os.environ.get(override_var, "").strip()
    if override:
        return Path(override).expanduser()
    return Path(f"/opt/anaconda3/envs/{env_name}/bin/python")


def moebius_python() -> Path:
    env_name = str(model_entry("moebius").get("environment") or "pixelforge-moebius")
    return _env_python(env_name, override_var="PIXELFORGE_MOEBIUS_PYTHON")


def grounding_python() -> Path:
    env_name = str(
        model_entry("grounding_dino").get("environment") or "pixelforge-grounding-dino"
    )
    return _env_python(env_name, override_var="PIXELFORGE_GROUNDING_DINO_PYTHON")


def ground_via_isolated_env(image: np.ndarray, prompt: str) -> GroundingResult:
    """Run Grounding DINO in its isolated environment."""
    if not _GROUNDING_WORKER.is_file():
        raise ModelLoadError(f"Isolated grounding worker not found: {_GROUNDING_WORKER}")

    python = grounding_python()
    if not python.is_file():
        raise ModelLoadError(f"Grounding DINO python not found: {python}")

    image = validate_image(image)
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="pixelforge_ground_") as tmp:
        tmp_path = Path(tmp)
        image_path = tmp_path / "image.png"
        Image.fromarray(image).save(image_path)
        proc = _run_isolated(
            [str(python), str(_GROUNDING_WORKER), str(image_path), prompt],
            label="Grounding DINO isolated inference",
        )
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()[-500:]
            raise ModelInferenceError(
                f"Grounding DINO isolated inference failed (exit {proc.returncode}). {detail}"
            )
        lines = [line for line in (proc.stdout or "").splitlines() if line.strip()]
        payload = json.loads(lines[-1]) if lines else {}

    wall_ms = (time.perf_counter() - t0) * 1000.0
    detections = [
        BoundingBox(
            x1=float(item["x1"]),
            y1=float(item["y1"]),
            x2=float(item["x2"]),
            y2=float(item["y2"]),
            confidence=float(item["confidence"]),
            label=str(item["label"]),
        )
        for item in payload.get("detections", [])
    ]
    metadata = dict(payload.get("metadata") or {})
    metadata["isolated_env"] = str(python)
    metadata["isolated_wall_ms"] = round(wall_ms, 3)
    return GroundingResult(
        detections=detections,
        model=str(payload.get("model") or "Grounding DINO"),
        prompt=str(payload.get("prompt") or prompt),
        backend=BackendType.CPU,
        metadata=metadata,
    )


def inpaint_via_isolated_env(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    params: InpaintParams | None = None,
) -> InpaintingResult:
    """Run Moebius inpainting in ``pixelforge-moebius`` when in-process load fails."""
    if params is not None:
        logger.warning("isolated_inpaint ignores custom params in MVP bridge")
    if not _INPAINT_WORKER.is_file():
        raise ModelLoadError(f"Isolated worker not found: {_INPAINT_WORKER}")

    python = moebius_python()
    if not python.is_file():
        raise ModelLoadError(f"Moebius python not found: {python}")

    image = validate_image(image)
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="pixelforge_inpaint_") as tmp:
        tmp_path = Path(tmp)
        image_path = tmp_path / "image.png"
        mask_path = tmp_path / "mask.png"
        out_path = tmp_path / "result.png"
        Image.fromarray(image).save(image_path)
        Image.fromarray((mask.astype(np.uint8) * 255), mode="L").save(mask_path)

        proc = _run_isolated(
            [str(python), str(_INPAINT_WORKER), str(image_path), str(mask_path), str(out_path)],
            label="Moebius isolated inference",
        )
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()[-500:]
            raise ModelInferenceError(
                f"Moebius isolated inference failed (exit {proc.returncode}). {detail}"
            )
        if not out_path.is_file():
            raise ModelInferenceError("Moebius isolated inference produced no output.")

        result_arr = validate_image(np.asarray(Image.open(out_path).convert("RGB")))
        lines = [line for line in (proc.stdout or "").splitlines() if line.strip()]
        meta = json.loads(lines[-1]) if lines else {}

    wall_ms = (time.perf_counter() - t0) * 1000.0
    return InpaintingResult(
        result=result_arr,
        latency_ms=float(meta.get("latency_ms") or wall_ms),
        memory_mb=meta.get("memory_mb"),
        model=str(meta.get("model") or "Moebius"),
        backend=BackendType.LOCAL_MPS,
        metadata={
            **(meta.get("metadata") or {}),
            "isolated_env": str(python),
            "isolated_wall_ms": round(wall_ms, 3),
        },
    )
