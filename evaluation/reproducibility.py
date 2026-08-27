"""Reproducibility metadata and content hashing."""

from __future__ import annotations

import hashlib
import platform
import sys
from pathlib import Path
from typing import Any

from evaluation.types import ExperimentConfig


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    p = Path(path)
    digest = hashlib.sha256()
    with p.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(image: Any, mask: Any | None = None, output: Any | None = None) -> dict[str, str | None]:
    """Hash numpy arrays as raw byte content (layout-dependent)."""
    import numpy as np

    def _hash_arr(arr: Any | None) -> str | None:
        if arr is None:
            return None
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"expected ndarray, got {type(arr).__name__}")
        return sha256_bytes(arr.tobytes())

    return {
        "image_sha256": _hash_arr(image),
        "mask_sha256": _hash_arr(mask),
        "output_sha256": _hash_arr(output),
    }


def build_reproducibility_record(
    *,
    config: ExperimentConfig,
    device: str | None = None,
    image_hash: str | None = None,
    mask_hash: str | None = None,
    output_hash: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Capture environment and content fingerprints for one run."""
    record: dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "architecture": platform.machine(),
        "model": config.model,
        "model_commit": config.model_commit,
        "backend": config.backend,
        "device": device,
        "environment": config.environment,
        "seed": config.seed,
        "parameters": dict(config.parameters),
        "hashes": {
            "image_sha256": image_hash,
            "mask_sha256": mask_hash,
            "output_sha256": output_hash,
        },
    }
    if extra:
        record["extra"] = extra
    return record
