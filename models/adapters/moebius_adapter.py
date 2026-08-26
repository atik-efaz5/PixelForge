"""Moebius student adapter — local MPS inpainting.

Teacher-free import isolation lives here, not in the upstream tree. GLA / ``fla``
are never loaded. See ``docs/experiments/MOEBIUS_MPS_VALIDATION.md``.
"""

from __future__ import annotations

import importlib
import os
import sys
import time
import types
from pathlib import Path
from typing import Any

import numpy as np

from models.adapters._runtime import (
    mps_is_available,
    mps_memory_mb,
    mps_synchronize,
    prepare_upstream_import,
)
from models.adapters.base import InpaintingAdapter
from models.config import model_entry, resolve_path
from models.errors import ModelInferenceError, ModelLoadError, ModelUnavailableError
from models.types import (
    BackendType,
    InpaintParams,
    InpaintingResult,
    image_to_pil_rgb,
    mask_to_pil_l,
    pil_rgb_to_array,
    validate_image,
    validate_mask,
)

STUDENT_CLASS = "UNet2DLambdaDWConvMixFFNConditionModel_prune_down_mid_up_block_8x8"
NUM_EMBEDDINGS = 20
SCHEDULER_KWARGS = dict(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    num_train_timesteps=1000,
    clip_sample=False,
)
_DEFAULT_STEPS = 20
_DEFAULT_GUIDANCE = 2.5
_DEFAULT_NOISE_OFFSET = 0.0357
_DEFAULT_PASTE = True
_DEFAULT_STRENGTH = 0.99
_DEFAULT_SIZE = 512


def install_student_import_surrogate(repo: Path) -> None:
    """Seed ``sys.modules['model_lib']`` so the CUDA-only teacher is never imported.

    Upstream ``model_lib/__init__.py`` line 6 imports PixelHacker GLA (``fla``).
    This surrogate keeps the real ``__path__`` and re-exports student symbols
    from lines 2–3 only. Nothing is stubbed; student source runs verbatim.
    Idempotent. Does not modify files on disk.
    """
    existing = sys.modules.get("model_lib")
    if existing is not None and getattr(existing, "_pixelforge_student_surrogate", False):
        return

    prepare_upstream_import(repo)
    pkg = types.ModuleType("model_lib")
    pkg.__path__ = [str(repo / "model_lib")]
    pkg.__package__ = "model_lib"
    pkg._pixelforge_student_surrogate = True  # type: ignore[attr-defined]
    sys.modules["model_lib"] = pkg

    student_mod = importlib.import_module("model_lib.nets.unet_lambda_prune_lite")
    blocks_mod = importlib.import_module("model_lib.nets.unet_lambda_dwconv_blocks")
    setattr(pkg, STUDENT_CLASS, getattr(student_mod, STUDENT_CLASS))
    for name in dir(blocks_mod):
        if not name.startswith("_"):
            setattr(pkg, name, getattr(blocks_mod, name))

    leaked = (
        "fla" in sys.modules
        or "triton" in sys.modules
        or any("unet_gla" in m or "layers.gla" in m for m in sys.modules)
    )
    if leaked:
        raise ModelLoadError(
            "Moebius student import pulled in the CUDA teacher path."
        )


class MoebiusAdapter(InpaintingAdapter):
    """Validated Phase 4 backend. ``CONDITIONAL`` because of import isolation."""

    def __init__(
        self,
        *,
        checkpoint_path: Path | None = None,
        vae_dir: Path | None = None,
        upstream_dir: Path | None = None,
        enabled: bool | None = None,
    ) -> None:
        super().__init__()
        cfg = model_entry("moebius")
        env_ckpt = os.environ.get("PIXELFORGE_MOEBIUS_CHECKPOINT")
        env_vae = os.environ.get("PIXELFORGE_MOEBIUS_VAE")
        self._checkpoint = (
            Path(env_ckpt).expanduser()
            if env_ckpt
            else checkpoint_path
            or resolve_path(cfg.get("checkpoint"))
        )
        self._vae_dir = (
            Path(env_vae).expanduser()
            if env_vae
            else vae_dir
            or resolve_path(cfg.get("vae_dir"))
        )
        self._upstream = upstream_dir or resolve_path(cfg.get("upstream_dir"))
        cfg_rel = str(cfg.get("config_file") or "config/model_cfg/moebius.yaml")
        self._config_file = (self._upstream / cfg_rel) if self._upstream else None
        self._enabled = cfg.get("enabled", True) if enabled is None else enabled
        self._display = str(cfg.get("display_name") or "Moebius")
        self._pipe: Any = None

    @property
    def model_name(self) -> str:
        return self._display

    @property
    def backend_type(self) -> BackendType:
        return BackendType.LOCAL_MPS

    def is_available(self) -> bool:
        if not self._enabled:
            return False
        if self._checkpoint is None or not self._checkpoint.is_file():
            return False
        if self._vae_dir is None:
            return False
        if not (self._vae_dir / "config.json").is_file():
            return False
        if not (self._vae_dir / "diffusion_pytorch_model.bin").is_file():
            return False
        if self._upstream is None or not self._upstream.is_dir():
            return False
        return self._mps_is_available()

    def _mps_is_available(self) -> bool:
        return mps_is_available()

    def load(self) -> None:
        if self._loaded:
            return
        if not self._enabled:
            self._error = "disabled"
            raise ModelUnavailableError("Moebius is disabled.")
        if self._checkpoint is None or not self._checkpoint.is_file():
            self._error = "checkpoint"
            raise ModelUnavailableError("Moebius student checkpoint is not available.")
        if self._vae_dir is None or not (self._vae_dir / "config.json").is_file():
            self._error = "vae"
            raise ModelUnavailableError("Moebius VAE checkpoint is not available.")
        if not self._mps_is_available():
            self._error = "mps"
            raise ModelUnavailableError(
                "Moebius requires Apple MPS, which is not available in this process."
            )
        if self._upstream is None or not self._upstream.is_dir():
            self._error = "upstream"
            raise ModelUnavailableError("Moebius upstream repository is not available.")

        self._begin_local_load()
        self._loading = True
        self._error = None
        try:
            install_student_import_surrogate(self._upstream)
            import torch
            from diffusers import DDIMScheduler
            from diffusers.models import AutoencoderKL
            from removal.v1_2 import build_removal_model, load_cfg, load_removal_model
            from removal.v1_2.pipeline import RemovalSDXLPipeline_BatchMode

            cfg_path = self._config_file or (
                self._upstream / "config" / "model_cfg" / "moebius.yaml"
            )
            model_cfg = load_cfg(str(cfg_path))
            model_cfg["vae"]["model_dir"] = str(self._vae_dir)

            dtype = torch.float32
            device = "mps"
            removal_model = build_removal_model(model_cfg, NUM_EMBEDDINGS)
            load_removal_model(removal_model, str(self._checkpoint), device, dtype)
            vae = AutoencoderKL.from_pretrained(model_cfg["vae"]["model_dir"])
            vae.to(device=device, dtype=dtype)
            scheduler = DDIMScheduler(**SCHEDULER_KWARGS)
            self._pipe = RemovalSDXLPipeline_BatchMode(
                removal_model=removal_model,
                vae=vae,
                scheduler=scheduler,
                device=device,
                dtype=dtype,
            )
            self._loaded = True
        except (ModelLoadError, ModelUnavailableError):
            self._end_local_unload()
            raise
        except Exception as exc:
            self._end_local_unload()
            self._error = type(exc).__name__
            raise ModelLoadError("Moebius failed to load.") from exc
        finally:
            self._loading = False

    def unload(self) -> None:
        self._pipe = None
        self._loaded = False
        self._loading = False
        self._end_local_unload()
        try:
            import gc

            import torch

            gc.collect()
            if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                torch.mps.empty_cache()
        except Exception:
            pass

    def infer(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        params: InpaintParams | None = None,
        /,
    ) -> InpaintingResult:
        image = validate_image(image)
        mask = validate_mask(mask, image=image)
        if not self._loaded:
            self.load()
        if self._pipe is None:
            raise ModelLoadError("Moebius is not loaded.")

        p = params or InpaintParams()
        num_steps = p.num_steps if p.num_steps is not None else _DEFAULT_STEPS
        guidance = (
            p.guidance_scale if p.guidance_scale is not None else _DEFAULT_GUIDANCE
        )
        strength = p.strength if p.strength is not None else _DEFAULT_STRENGTH
        paste = p.paste if p.paste is not None else _DEFAULT_PASTE
        noise_offset = (
            p.noise_offset if p.noise_offset is not None else _DEFAULT_NOISE_OFFSET
        )
        image_size = p.image_size if p.image_size is not None else _DEFAULT_SIZE

        pil_img = image_to_pil_rgb(image)
        pil_mask = mask_to_pil_l(mask)
        t0 = time.perf_counter()
        try:
            out_list = self._pipe(
                [pil_img],
                [pil_mask],
                image_size=image_size,
                num_steps=num_steps,
                guidance_scale=guidance,
                noise_offset=noise_offset,
                paste=paste,
                compensate=False,
                strength=strength,
                mute=True,
                visualize=False,
            )
            mps_synchronize()
        except Exception as exc:
            self._error = type(exc).__name__
            raise ModelInferenceError("Moebius inference failed.") from exc

        latency_ms = (time.perf_counter() - t0) * 1000.0
        result = pil_rgb_to_array(out_list[0])
        return InpaintingResult(
            result=result,
            latency_ms=round(latency_ms, 3),
            memory_mb=mps_memory_mb(),
            model=self.model_name,
            backend=self.backend_type,
            metadata={
                "num_steps": num_steps,
                "guidance_scale": guidance,
                "strength": strength,
                "paste": paste,
                "noise_offset": noise_offset,
                "image_size": image_size,
                "device": "mps",
                "import_isolation": True,
            },
        )
