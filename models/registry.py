"""Lazy adapter registry. Construction does not load weights.

Intelligent routing is Phase 10. Callers ask for a named adapter explicitly.
"""

from __future__ import annotations

from collections.abc import Callable

from models.adapters.base import ModelAdapter
from models.errors import ModelUnavailableError

_FACTORIES: dict[str, Callable[[], ModelAdapter]] | None = None
_INSTANCES: dict[str, ModelAdapter] = {}


def _factories() -> dict[str, Callable[[], ModelAdapter]]:
    global _FACTORIES
    if _FACTORIES is None:
        from models.adapters.grounding_dino_adapter import GroundingDINOAdapter
        from models.adapters.moebius_adapter import MoebiusAdapter
        from models.adapters.pixelhacker_adapter import PixelHackerAdapter
        from models.adapters.sam2_adapter import SAM2Adapter

        _FACTORIES = {
            "sam2": SAM2Adapter,
            "moebius": MoebiusAdapter,
            "pixelhacker": PixelHackerAdapter,
            "grounding_dino": GroundingDINOAdapter,
        }
    return _FACTORIES


def known_models() -> tuple[str, ...]:
    return tuple(_factories().keys())


def get_adapter(name: str) -> ModelAdapter:
    """Return a cached adapter instance. Does not call ``load()``."""
    key = name.strip().lower()
    factories = _factories()
    if key not in factories:
        raise ModelUnavailableError(f"Unknown model '{name}'.")
    if key not in _INSTANCES:
        _INSTANCES[key] = factories[key]()
    return _INSTANCES[key]


def reset_registry() -> None:
    """Drop cached instances. Used by tests. Unloads any resident local model."""
    for adapter in list(_INSTANCES.values()):
        try:
            adapter.unload()
        except Exception:
            pass
    _INSTANCES.clear()
