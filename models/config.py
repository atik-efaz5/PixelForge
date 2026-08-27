"""Load ``configs/models.yaml`` and resolve paths relative to the project root.

Checkpoint locations are project-relative and overridable by environment
variables. User-specific absolute paths are not stored in the YAML file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def project_root() -> Path:
    override = os.environ.get("PIXELFORGE_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[1]


def _expand(value: str) -> str:
    return os.path.expandvars(os.path.expanduser(value))


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        return {}
    if not path.is_file():
        return {}
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        return {}
    return data


def load_models_config() -> dict[str, Any]:
    """Return the ``models`` mapping from ``configs/models.yaml``, or defaults."""
    path = project_root() / "configs" / "models.yaml"
    data = _load_yaml(path)
    models = data.get("models") if isinstance(data.get("models"), dict) else None
    return models if models else dict(_DEFAULTS)


def model_entry(name: str) -> dict[str, Any]:
    models = load_models_config()
    entry = models.get(name) or _DEFAULTS.get(name)
    if not entry:
        return {}
    return dict(entry)


def resolve_path(value: str | Path | None) -> Path | None:
    if value is None or value == "":
        return None
    path = Path(_expand(str(value)))
    if not path.is_absolute():
        path = project_root() / path
    return path


_DEFAULTS: dict[str, dict[str, Any]] = {
    "sam2": {
        "model_id": "sam2.1_hiera_tiny",
        "display_name": "SAM 2.1 Hiera-Tiny",
        "backend": "LOCAL_MPS",
        "enabled": True,
        "environment": "pixelforge-sam2-v2",
        "checkpoint": "checkpoints/sam2/sam2.1_hiera_tiny.pt",
        "config_file": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "upstream_dir": "research/upstream/sam2",
    },
    "moebius": {
        "model_id": "moebius_ft_places2",
        "display_name": "Moebius",
        "backend": "LOCAL_MPS",
        "enabled": True,
        "environment": "pixelforge-moebius",
        "checkpoint": "checkpoints/moebius/ft_places2/diffusion_pytorch_model.bin",
        "vae_dir": "checkpoints/moebius/vae",
        "upstream_dir": "research/upstream/Moebius",
        "config_file": "config/model_cfg/moebius.yaml",
    },
    "pixelhacker": {
        "model_id": "pixelhacker_ft_places2",
        "display_name": "PixelHacker",
        "backend": "CLOUD_GPU",
        "enabled": True,
        "environment": "pixelforge-pixelhacker",
        "checkpoint": "checkpoints/pixelhacker/ft_places2/diffusion_pytorch_model.bin",
        "vae_dir": "checkpoints/pixelhacker/vae",
        "upstream_dir": "research/upstream/PixelHacker",
        "endpoint": "",
    },
    "grounding_dino": {
        "model_id": "groundingdino_swint_ogc",
        "display_name": "Grounding DINO SwinT OGC",
        "backend": "CPU",
        "enabled": True,
        "environment": "pixelforge-grounding-dino",
        "checkpoint": "checkpoints/grounding_dino/groundingdino_swint_ogc.pth",
        "config_file": "research/upstream/Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        "upstream_dir": "research/upstream/Grounded-Segment-Anything",
        "device": "cpu",
        "box_threshold": 0.3,
        "text_threshold": 0.25,
    },
}
