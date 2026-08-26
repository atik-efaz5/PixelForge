"""Shared device and import helpers. No research-model imports."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def mps_is_available() -> bool:
    """True when this process can place tensors on Apple MPS.

    Metal reachability is per-process (Phase 3). A missing ``torch`` is False.
    """
    try:
        import torch
    except ImportError:
        return False
    try:
        return bool(torch.backends.mps.is_built() and torch.backends.mps.is_available())
    except Exception:
        return False


def prepare_upstream_import(repo: Path) -> None:
    """Put a READ-ONLY checkout on ``sys.path`` without writing bytecode into it."""
    sys.dont_write_bytecode = True
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    root = str(repo)
    if root not in sys.path:
        sys.path.insert(0, root)


def mps_memory_mb() -> float | None:
    try:
        import torch

        if not hasattr(torch, "mps"):
            return None
        bytes_ = torch.mps.driver_allocated_memory()
        return round(bytes_ / 1024**2, 2)
    except Exception:
        return None


def mps_synchronize() -> None:
    try:
        import torch

        torch.mps.synchronize()
    except Exception:
        pass
