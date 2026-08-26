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
from models.types import BackendType, InpaintParams, InpaintingResult, validate_image

logger = logging.getLogger(__name__)

_WORKER = project_root() / "scripts" / "isolated_inpaint_worker.py"


def _env_python(env_name: str, *, override_var: str) -> Path:
    override = os.environ.get(override_var, "").strip()
    if override:
        return Path(override).expanduser()
    return Path(f"/opt/anaconda3/envs/{env_name}/bin/python")


def moebius_python() -> Path:
    env_name = str(model_entry("moebius").get("environment") or "pixelforge-moebius")
    return _env_python(env_name, override_var="PIXELFORGE_MOEBIUS_PYTHON")


def inpaint_via_isolated_env(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    params: InpaintParams | None = None,
) -> InpaintingResult:
    """Run Moebius inpainting in ``pixelforge-moebius`` when in-process load fails."""
    if params is not None:
        logger.warning("isolated_inpaint ignores custom params in MVP bridge")
    if not _WORKER.is_file():
        raise ModelLoadError(f"Isolated worker not found: {_WORKER}")

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

        proc = subprocess.run(
            [str(python), str(_WORKER), str(image_path), str(mask_path), str(out_path)],
            cwd=str(project_root()),
            capture_output=True,
            text=True,
            check=False,
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
