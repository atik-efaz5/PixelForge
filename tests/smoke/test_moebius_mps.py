"""Moebius student inference — Apple MPS reproducibility gate (PixelForge Phase 4).

Answers exactly one question: can Moebius *student* inference produce a valid
inpainting result on M3 Pro MPS **without changing the research algorithm**?

Design constraints this file is built around
--------------------------------------------
1.  The upstream repository is READ-ONLY and is never modified. Moebius is
    imported by putting the pinned checkout on ``sys.path``; it is deliberately
    NOT pip-installed, and bytecode writing is disabled before any upstream
    import so the checkout stays byte-identical.

2.  **The student path cannot be imported normally.** ``model_lib/__init__.py``
    line 6 imports the *teacher*::

        from .nets.unet_gla import UNet2DGLAConditionModel   # PixelHacker teacher

    which reaches ``model_lib/nets/layers/gla/gla.py:16``::

        from fla.ops.gla import chunk_gla, fused_chunk_gla, fused_recurrent_gla

    ``fla`` is ``flash-linear-attention[cuda]==0.3.2``, which requires Triton.
    Triton publishes no macOS / Apple-Silicon wheels, so ``fla`` is not
    installable on this host at all. Because Python executes a package's
    ``__init__.py`` even when you import only a submodule, ``import
    model_lib.nets.unet_lambda_prune_lite`` — the *student* — dies on the
    *teacher's* CUDA-only dependency.

    The directive forbids editing the upstream file. This harness therefore
    seeds ``sys.modules['model_lib']`` with a surrogate package object that
    carries the **real** ``__path__`` but an empty body, so ``__init__.py``
    never runs while submodule imports still resolve normally under their true
    dotted names. Nothing is stubbed, mocked, or reimplemented: the student
    source executes verbatim through the standard import machinery. See
    ``install_student_import_surrogate`` for the full rationale and the runtime
    assertions that prove no teacher/GLA/Triton code was loaded.

3.  No silent CPU fallback. MPS is required by default. CPU requires an explicit
    ``--allow-cpu-fallback`` and then forces the verdict to PARTIAL, never PASS.

4.  Nothing is fabricated. Every number in ``result.json`` is measured. Fields
    that could not be measured are ``null``, never guessed.

5.  The test image and mask are generated from closed-form arithmetic with no
    RNG, so they are byte-identical on every run and every machine.

6.  **Only device placement and file paths are configured.** Steps (20),
    resolution (512), guidance scale (2.5), noise offset (0.0357), paste (True),
    strength (0.99) and dtype (float32) are all read from upstream's own
    ``infer/utils.py`` argparse defaults and ``pipeline.py`` signature defaults.
    The research algorithm is untouched.

Run directly::

    python tests/smoke/test_moebius_mps.py

or under pytest::

    pytest tests/smoke/test_moebius_mps.py -v -s
"""

from __future__ import annotations

# Disable bytecode writing BEFORE importing anything from the upstream tree, so
# that importing Moebius cannot create __pycache__/ directories inside it.
import sys

sys.dont_write_bytecode = True

import argparse
import hashlib
import importlib
import json
import os
import platform
import re
import resource
import subprocess
import time
import traceback
import types
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# --------------------------------------------------------------------------- #
# Pinned facts. Every one is read from the repository, never assumed.
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MOEBIUS_REPO = PROJECT_ROOT / "research" / "upstream" / "Moebius"

# research/upstream/LOCKFILE.md -- Phase 2 pin, verified twice, never changed.
MOEBIUS_PINNED_COMMIT = "b88d462bacb9af6e7128a3b4cc4a07418bedfd61"

# The student class named by config/model_cfg/moebius.yaml -> model.model_type.
# removal_model.py:71 resolves it with eval() against that module's globals,
# which is why the surrogate must expose it as a model_lib attribute.
STUDENT_CLASS = "UNet2DLambdaDWConvMixFFNConditionModel_prune_down_mid_up_block_8x8"

# Upstream config, used verbatim except for the VAE directory (see below).
MODEL_CONFIG = MOEBIUS_REPO / "config" / "model_cfg" / "moebius.yaml"

# Weights live outside the upstream tree and outside git.
#   README.md:159-172 -> https://huggingface.co/hustvl/Moebius/tree/main/ft_places2
#   README.md:123-146 -> https://huggingface.co/hustvl/PixelHacker/tree/main/vae
CKPT_ROOT = PROJECT_ROOT / "checkpoints" / "moebius"
STUDENT_WEIGHT = CKPT_ROOT / "ft_places2" / "diffusion_pytorch_model.bin"
VAE_DIR = CKPT_ROOT / "vae"

STUDENT_WEIGHT_URL = (
    "https://huggingface.co/hustvl/Moebius/resolve/main/ft_places2/"
    "diffusion_pytorch_model.bin"
)
VAE_REPO_URL = "https://huggingface.co/hustvl/PixelHacker/tree/main/vae"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "moebius_mps_validation"

# --------------------------------------------------------------------------- #
# Inference configuration -- every value below is an UPSTREAM default, quoted
# with its source. Nothing here is a PixelForge tuning choice.
# --------------------------------------------------------------------------- #

NUM_EMBEDDINGS = 20  # infer/utils.py:67  build_removal_model(model_cfg, 20)
IMAGE_SIZE = 512  # infer/utils.py --resolution default; yaml data.image_size
NUM_STEPS = 20  # infer/utils.py --num-step default
GUIDANCE_SCALE = 2.5  # infer/utils.py --cfg default
NOISE_OFFSET = 0.0357  # infer/utils.py --noise-offset default
PASTE = True  # infer/utils.py --pst default
COMPENSATE = False  # infer/utils.py --cps default
STRENGTH = 0.99  # pipeline.py __call__ signature default
BATCH_SIZE = 1  # directive: batch size = 1

# infer/utils.py:71-73 -- DDIMScheduler arguments, byte-identical.
SCHEDULER_KWARGS = dict(
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    num_train_timesteps=1000,
    clip_sample=False,
)

# infer/utils.py:79 passes dtype=torch.float -> float32 is the UPSTREAM
# inference default. (pipeline.py's own signature default is float16, but the
# upstream inference entry point overrides it to float32.)
DTYPE_NAME = "float32"

# Synthetic scene geometry -- fixed, so image and mask are reproducible.
IMG_W = IMG_H = 512
DISC_CENTER = (352, 200)  # (x, y) -- the object to be removed
DISC_RADIUS = 74

WARM_ITERS = 3

# Documented student size, README.md describes Moebius as a ~0.2B student.
DOCUMENTED_PARAMS_MILLIONS = 226.04
PARAM_COUNT_TOLERANCE = 0.02

# The eleven checks the Phase 4 directive requires, in order.
MANDATED_CHECKS = [
    "1_mps_available",
    "2_student_model_import",
    "3_vae_import",
    "4_checkpoint_loads",
    "5_model_moves_to_mps",
    "6_vae_moves_to_mps",
    "7_pipeline_initializes",
    "8_inference_executes",
    "9_output_is_finite",
    "10_output_valid_dimensions",
    "11_output_writes_png",
]


# --------------------------------------------------------------------------- #
# Result record
# --------------------------------------------------------------------------- #


@dataclass
class Record:
    """Everything measured during a run. Serialized verbatim to result.json."""

    verdict: str = "NOT_RUN"
    failure_stage: str | None = None
    failure_type: str | None = None
    failure_message: str | None = None
    failure_traceback: str | None = None
    failing_operation: str | None = None
    failure_classification: str | None = None
    checks: dict[str, Any] = field(
        default_factory=lambda: {k: None for k in MANDATED_CHECKS}
    )
    environment: dict[str, Any] = field(default_factory=dict)
    torch_info: dict[str, Any] = field(default_factory=dict)
    hardware: dict[str, Any] = field(default_factory=dict)
    repository: dict[str, Any] = field(default_factory=dict)
    import_strategy: dict[str, Any] = field(default_factory=dict)
    checkpoints: dict[str, Any] = field(default_factory=dict)
    device: dict[str, Any] = field(default_factory=dict)
    inference_config: dict[str, Any] = field(default_factory=dict)
    model_load: dict[str, Any] = field(default_factory=dict)
    vae_load: dict[str, Any] = field(default_factory=dict)
    test_input: dict[str, Any] = field(default_factory=dict)
    inference: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)
    memory: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, sort_keys=False, default=str)


class GateFailure(RuntimeError):
    """A validation gate was not met.

    ``blocking=True`` marks an *environmental precondition* that was absent --
    no Metal device visible, no checkpoint on disk, a package that cannot be
    installed on this platform. Nothing about Moebius was disproven, so the run
    is inconclusive (PARTIAL) rather than negative (FAIL).

    ``blocking=False`` means the pipeline actually ran and produced a wrong or
    unusable result, which is a genuine FAIL about this commit on this hardware.
    """

    def __init__(
        self,
        stage: str,
        message: str,
        blocking: bool = False,
        classification: str | None = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.blocking = blocking
        self.classification = classification


# The directive's failure taxonomy. Recorded rather than guessed at.
FAILURE_CLASSES = {
    "A": "dependency incompatibility",
    "B": "import contamination",
    "C": "unsupported MPS operator",
    "D": "dtype issue",
    "E": "checkpoint/config mismatch",
    "F": "memory exhaustion",
    "G": "model implementation incompatibility",
}


# --------------------------------------------------------------------------- #
# Stage 1 -- pinned repository verification
# --------------------------------------------------------------------------- #


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(MOEBIUS_REPO), *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _count_pycache() -> int:
    return sum(1 for _ in MOEBIUS_REPO.rglob("__pycache__")) + sum(
        1 for _ in MOEBIUS_REPO.rglob("*.egg-info")
    )


def verify_pinned_repository(rec: Record) -> None:
    if not MOEBIUS_REPO.is_dir():
        raise GateFailure(
            "repository",
            f"upstream Moebius clone missing at {MOEBIUS_REPO}",
            blocking=True,
        )

    head = _git("rev-parse", "HEAD")
    head_again = _git("rev-parse", "HEAD")  # verified twice, per project policy
    dirty = [ln for ln in _git("status", "--porcelain").splitlines() if ln.strip()]

    rec.repository = {
        "path": str(MOEBIUS_REPO),
        "expected_commit": MOEBIUS_PINNED_COMMIT,
        "observed_commit": head,
        "observed_commit_second_pass": head_again,
        "commit_matches_pin": head == MOEBIUS_PINNED_COMMIT == head_again,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "changed_files_before": len(dirty),
        "working_tree_clean_before": not dirty,
        "pycache_or_egginfo_before": _count_pycache(),
        "installed_into_environment": False,
        "import_mechanism": "sys.path insertion (no pip install, upstream untouched)",
        "license": "Apache-2.0 (README.md:186 -- covers code AND pretrained weights)",
    }

    if head != MOEBIUS_PINNED_COMMIT or head_again != MOEBIUS_PINNED_COMMIT:
        raise GateFailure(
            "repository",
            f"commit mismatch: expected {MOEBIUS_PINNED_COMMIT}, "
            f"observed {head} / {head_again}",
        )
    if dirty:
        raise GateFailure(
            "repository",
            f"upstream working tree is dirty ({len(dirty)} files) -- READ-ONLY "
            f"rule violated: {dirty[:5]}",
        )


def confirm_upstream_untouched(rec: Record) -> None:
    """Re-verify after every import and inference. The READ-ONLY rule is a
    postcondition, not just a precondition."""
    dirty = [ln for ln in _git("status", "--porcelain").splitlines() if ln.strip()]
    head = _git("rev-parse", "HEAD")
    rec.repository["observed_commit_after_run"] = head
    rec.repository["changed_files_after"] = len(dirty)
    rec.repository["working_tree_clean_after"] = not dirty
    rec.repository["pycache_or_egginfo_after"] = _count_pycache()
    rec.repository["upstream_unmodified"] = (
        not dirty and head == MOEBIUS_PINNED_COMMIT and _count_pycache() == 0
    )


# --------------------------------------------------------------------------- #
# Stage 2 -- device verification
# --------------------------------------------------------------------------- #


def _metal_device_available() -> bool | None:
    """Probe Metal directly, bypassing PyTorch.

    ``torch.backends.mps.is_available()`` returning False is ambiguous: it emits
    a misleading "macOS 14.0+" message even on macOS 15 when the process simply
    has no GPU device (e.g. inside a sandbox). This distinguishes "OS too old"
    from "no GPU visible to this process" -- the difference between a real
    incompatibility and an environment problem. Phase 3 hit exactly this.
    """
    try:
        import ctypes

        lib = ctypes.CDLL("/System/Library/Frameworks/Metal.framework/Metal")
        lib.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
        return bool(lib.MTLCreateSystemDefaultDevice())
    except Exception:
        return None


def _sw_vers(key: str) -> str | None:
    try:
        return subprocess.run(
            ["sw_vers", key], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return None


def _sysctl(key: str) -> str | None:
    try:
        return subprocess.run(
            ["sysctl", "-n", key], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return None


def verify_device(rec: Record, allow_cpu_fallback: bool) -> str:
    import torch

    rec.environment = {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
    }

    versions: dict[str, Any] = {"torch_version": torch.__version__}
    for mod in ("torchvision", "numpy", "diffusers", "transformers", "accelerate",
                "timm", "einops", "omegaconf", "safetensors", "PIL", "cv2"):
        try:
            versions[f"{mod}_version"] = getattr(
                importlib.import_module(mod), "__version__", "unknown"
            )
        except Exception as exc:
            versions[f"{mod}_version"] = f"ABSENT ({type(exc).__name__})"

    metal = _metal_device_available()
    rec.torch_info = {
        **versions,
        "mps_is_built": bool(torch.backends.mps.is_built()),
        "mps_is_available": bool(torch.backends.mps.is_available()),
        "cuda_is_available": bool(torch.cuda.is_available()),
        "fla_installed": importlib.util.find_spec("fla") is not None,
        "triton_installed": importlib.util.find_spec("triton") is not None,
    }
    for major in (13, 14, 15):
        try:
            rec.torch_info[f"mps_is_on_macos_{major}_0_or_newer"] = bool(
                torch._C._mps_is_on_macos_or_newer(major, 0)
            )
        except Exception:
            rec.torch_info[f"mps_is_on_macos_{major}_0_or_newer"] = None

    mem = _sysctl("hw.memsize")
    rec.hardware = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": _sysctl("machdep.cpu.brand_string") or platform.processor(),
        "physical_cores": _sysctl("hw.physicalcpu"),
        "logical_cores": _sysctl("hw.logicalcpu"),
        "unified_memory_bytes": mem,
        "macos_product_version": _sw_vers("-productVersion"),
        "macos_build": _sw_vers("-buildVersion"),
        "metal_default_device_creatable": metal,
    }

    # fla/Triton availability is a platform fact worth recording either way:
    # it is the sole reason the import surrogate exists.
    if rec.torch_info["fla_installed"]:
        rec.notes.append(
            "flash-linear-attention IS installed in this environment. The import "
            "surrogate is still used, because loading the teacher is unnecessary "
            "for student inference and would change what is being measured."
        )

    if torch.backends.mps.is_available():
        try:
            t = torch.ones((2, 2), device="mps")
            got = (t @ t + 1.5).cpu().tolist()
            expected = [[3.5, 3.5], [3.5, 3.5]]
            rec.device = {
                "mps_tensor_op": {
                    "attempted": True,
                    "succeeded": True,
                    "expression": "torch.ones((2,2), device='mps') @ itself + 1.5",
                    "value": got,
                    "expected": expected,
                    "matches_expected": got == expected,
                },
                "selected": "mps",
            }
            if got != expected:
                # Completing without raising is not the same as being correct.
                raise GateFailure(
                    "device",
                    f"MPS arithmetic is wrong: got {got}, expected {expected}",
                    classification="C",
                )
            rec.checks["1_mps_available"] = True
            return "mps"
        except GateFailure:
            raise
        except Exception as exc:
            rec.device = {
                "mps_tensor_op": {
                    "attempted": True,
                    "succeeded": False,
                    "error": f"{type(exc).__name__}: {exc}",
                },
                "selected": None,
            }
            rec.checks["1_mps_available"] = False
            raise GateFailure(
                "device",
                f"MPS reported available but a real tensor op failed: {exc}",
                blocking=True,
                classification="C",
            )

    # MPS not available -- diagnose properly rather than quoting torch's text.
    rec.checks["1_mps_available"] = False
    diagnosis = (
        "no Metal device is visible to this process "
        "(MTLCreateSystemDefaultDevice() returned NULL). This is an execution-"
        "context limitation, NOT a PyTorch defect, NOT an OS version problem, "
        "and NOT a Moebius incompatibility. Re-run from an interactive terminal "
        "with GPU access."
        if metal is False
        else "torch.backends.mps.is_available() is False; direct Metal probe was "
        "inconclusive."
    )
    rec.device = {"selected": None, "diagnosis": diagnosis}

    if allow_cpu_fallback:
        rec.device["selected"] = "cpu"
        rec.notes.append(
            "CPU FALLBACK WAS USED EXPLICITLY. The verdict is capped at PARTIAL "
            "and can never be PASS: this run says nothing about MPS."
        )
        return "cpu"

    raise GateFailure("device", f"MPS unusable -- {diagnosis}", blocking=True)


def _sync(torch_mod, device: str) -> None:
    if device == "mps":
        torch_mod.mps.synchronize()


# --------------------------------------------------------------------------- #
# Stage 3 -- the import strategy
# --------------------------------------------------------------------------- #


def install_student_import_surrogate(rec: Record) -> types.ModuleType:
    """Load ONLY the student implementation from a READ-ONLY upstream tree.

    Why this is necessary
    ---------------------
    ``model_lib/__init__.py`` is six lines and imports both halves of the
    distillation pair::

        1  # Moebius student
        2  from .nets.unet_lambda_prune_lite import UNet2DLambdaDWConv...8x8
        3  from .nets.unet_lambda_dwconv_blocks import *
        4
        5  # PixelHacker teacher
        6  from .nets.unet_gla import UNet2DGLAConditionModel

    Line 6 reaches ``model_lib/nets/layers/gla/gla.py``, which imports ``fla``
    (``flash-linear-attention[cuda]``) at module level. ``fla`` requires Triton,
    which has no macOS/Apple-Silicon wheels. Python runs a package's
    ``__init__.py`` before any of its submodules, so the *teacher's* CUDA-only
    dependency blocks the *student's* import.

    Why this is not a modification of the research code
    ---------------------------------------------------
    * The upstream file is not edited, moved, patched, or monkeypatched.
    * Nothing is stubbed or mocked. A stub was considered and REJECTED: because
      ``gla.py:27`` subclasses ``fla``'s ``GatedLinearAttention``, a mock would
      have to fake real behaviour, and it would still execute teacher code.
    * The student modules are imported under their true dotted names through the
      ordinary import machinery, so their source executes verbatim.
    * Only line 6 -- the teacher -- is omitted. Lines 2-3, the student exports,
      are reproduced exactly, which is what ``removal_model.py:47``'s
      ``from model_lib import *`` needs. (``removal_model.py:71`` resolves the
      architecture with ``eval(model_type)`` against that module's globals, so
      the class must genuinely be an attribute of ``model_lib``.)

    This works because ``model_lib/nets/`` and everything below it are PEP 420
    implicit namespace packages -- they have no ``__init__.py`` of their own --
    so nothing else needs to be intercepted.

    The return value is the surrogate; the assertions afterwards prove that no
    teacher, GLA, ``fla`` or Triton module was loaded.
    """
    if str(MOEBIUS_REPO) not in sys.path:
        sys.path.insert(0, str(MOEBIUS_REPO))

    init_src = (MOEBIUS_REPO / "model_lib" / "__init__.py").read_text()

    pkg = types.ModuleType("model_lib")
    pkg.__path__ = [str(MOEBIUS_REPO / "model_lib")]  # the REAL path
    pkg.__package__ = "model_lib"
    pkg.__doc__ = (
        "PixelForge surrogate for model_lib. Carries the real __path__ so "
        "submodules resolve normally, with an empty body so the upstream "
        "__init__.py -- which imports the CUDA-only teacher -- never executes. "
        "Upstream is not modified."
    )
    sys.modules["model_lib"] = pkg

    try:
        student_mod = importlib.import_module("model_lib.nets.unet_lambda_prune_lite")
        blocks_mod = importlib.import_module("model_lib.nets.unet_lambda_dwconv_blocks")
    except Exception as exc:
        raise GateFailure(
            "import_strategy",
            f"student module import failed: {type(exc).__name__}: {exc}",
            classification="B",
        ) from exc

    # Reproduce model_lib/__init__.py lines 2-3 exactly, and only those.
    setattr(pkg, STUDENT_CLASS, getattr(student_mod, STUDENT_CLASS))
    exported = [STUDENT_CLASS]
    for name in dir(blocks_mod):
        if not name.startswith("_"):
            setattr(pkg, name, getattr(blocks_mod, name))
            exported.append(name)

    leaked = {
        "fla": "fla" in sys.modules,
        "triton": "triton" in sys.modules,
        "unet_gla": any("unet_gla" in m for m in sys.modules),
        "layers.gla": any("layers.gla" in m for m in sys.modules),
    }

    rec.import_strategy = {
        "reason": (
            "model_lib/__init__.py:6 imports the PixelHacker teacher "
            "(unet_gla -> layers/gla/gla.py:16 -> fla), and fla "
            "(flash-linear-attention[cuda]) requires Triton, which has no "
            "Apple-Silicon wheels. Python executes __init__.py before any "
            "submodule, so the teacher's CUDA-only dependency blocks the "
            "student's import."
        ),
        "mechanism": (
            "sys.modules['model_lib'] is seeded with a surrogate ModuleType "
            "carrying the real __path__ and an empty body, then the student "
            "submodules are imported under their true dotted names and "
            "re-exported exactly as __init__.py lines 2-3 do."
        ),
        "upstream_file_modified": False,
        "anything_stubbed_or_mocked": False,
        "upstream_init_source": init_src,
        "upstream_init_lines_reproduced": [2, 3],
        "upstream_init_lines_omitted": [6],
        "omitted_symbol": "UNet2DGLAConditionModel (teacher, not used by student inference)",
        "student_module": student_mod.__name__,
        "student_module_file": getattr(student_mod, "__file__", None),
        "blocks_module_file": getattr(blocks_mod, "__file__", None),
        "exported_symbol_count": len(exported),
        "exported_symbols": sorted(exported),
        "namespace_package_note": (
            "model_lib/nets/ and below have no __init__.py (PEP 420 implicit "
            "namespace packages), so no further interception is required and "
            "the student source executes verbatim."
        ),
        "teacher_contamination_check": leaked,
        "teacher_free": not any(leaked.values()),
        "rejected_alternative": (
            "Stubbing 'fla' in sys.modules was rejected: gla.py:27 subclasses "
            "fla's GatedLinearAttention, so a stub would have to fake real "
            "behaviour, and it would still execute teacher code."
        ),
    }

    if any(leaked.values()):
        raise GateFailure(
            "import_strategy",
            f"teacher/CUDA contamination leaked into sys.modules: {leaked}",
            classification="B",
        )

    rec.checks["2_student_model_import"] = True
    confirm_upstream_untouched(rec)
    return pkg


# --------------------------------------------------------------------------- #
# Stage 4 -- checkpoint audit
# --------------------------------------------------------------------------- #


def _sha256(path: Path, limit_bytes: int | None = None) -> str:
    h = hashlib.sha256()
    read = 0
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
            read += len(chunk)
            if limit_bytes and read >= limit_bytes:
                break
    return h.hexdigest()


def verify_checkpoints(rec: Record) -> None:
    student = {
        "role": "Moebius student, ft_places2",
        "filename": STUDENT_WEIGHT.name,
        "path": str(STUDENT_WEIGHT),
        "source_url": STUDENT_WEIGHT_URL,
        "source_documented_at": "research/upstream/Moebius/README.md:159-172",
        "upstream_published_checksum": None,
        "present": STUDENT_WEIGHT.is_file(),
    }
    vae = {
        "role": "SD VAE (sdvae_f8d4), required by config/model_cfg/moebius.yaml",
        "directory": str(VAE_DIR),
        "source_url": VAE_REPO_URL,
        "source_documented_at": "research/upstream/Moebius/README.md:123-146",
        "expected_files": ["config.json", "diffusion_pytorch_model.bin"],
        "upstream_published_checksum": None,
        "present": (VAE_DIR / "config.json").is_file(),
    }

    if student["present"]:
        student["size_bytes"] = STUDENT_WEIGHT.stat().st_size
        student["size_mib"] = round(student["size_bytes"] / 1024**2, 2)
        student["observed_sha256"] = _sha256(STUDENT_WEIGHT)
        student["observed_sha256_note"] = (
            "SHA-256 of the file as downloaded on this host. Upstream publishes "
            "no checksum, so this is a local re-acquisition anchor, not "
            "verification against an upstream-published value."
        )
    if vae["present"]:
        vae["files"] = sorted(p.name for p in VAE_DIR.iterdir() if p.is_file())
        vae["size_bytes"] = sum(
            p.stat().st_size for p in VAE_DIR.iterdir() if p.is_file()
        )
        vae["size_mib"] = round(vae["size_bytes"] / 1024**2, 2)

    rec.checkpoints = {
        "student": student,
        "vae": vae,
        "downloaded_but_not_required": [],
        "deliberately_not_downloaded": [
            "hustvl/Moebius pretrained (not needed for this gate)",
            "hustvl/Moebius ft_celebahq (not needed)",
            "hustvl/Moebius ft_ffhq (not needed)",
            "PixelHacker teacher weights (teacher is not on the student path)",
            "training datasets (Places2, CelebA-HQ, FFHQ)",
        ],
        "storage_policy": "outside git; checkpoints/ is git-ignored",
    }

    missing = [
        n for n, d in (("student", student), ("vae", vae)) if not d["present"]
    ]
    if missing:
        raise GateFailure(
            "checkpoint",
            f"required checkpoint(s) absent: {missing}. Student weight expected "
            f"at {STUDENT_WEIGHT}; VAE directory expected at {VAE_DIR}.",
            blocking=True,  # nothing about Moebius was tested
            # Deliberately unclassified: a file that was never downloaded is an
            # environmental absence, not a checkpoint/config MISMATCH (class E).
            # Labelling it E would misreport an un-run gate as a real defect.
            classification=None,
        )

    # Pre-flight the pickle format. torch 2.6+ resolves weights_only=None to
    # True; a non-tensor payload would raise. Distinguishing that here means a
    # checkpoint-format problem (class E) can never be misreported as an MPS
    # operator problem (class C) later.
    import torch

    try:
        sd = torch.load(STUDENT_WEIGHT, map_location="cpu", weights_only=True)
        student["loads_with_weights_only_true"] = True
        student["state_dict_entries"] = len(sd)
        student["state_dict_prefixes"] = sorted(
            {k.split(".")[0] for k in sd.keys()}
        )
        student["state_dict_dtypes"] = sorted(
            {str(v.dtype) for v in sd.values() if hasattr(v, "dtype")}
        )
        del sd
    except Exception as exc:
        student["loads_with_weights_only_true"] = False
        student["weights_only_error"] = f"{type(exc).__name__}: {exc}"
        student["remediation"] = (
            "Upstream calls torch.load without weights_only "
            "(removal_model.py:95). If this checkpoint needs the legacy "
            "unpickler, set TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 -- a "
            "PyTorch-supported environment switch that changes neither the "
            "upstream file nor the research algorithm."
        )
        raise GateFailure(
            "checkpoint",
            f"student checkpoint failed to unpickle: {exc}",
            classification="E",
        ) from exc

    rec.checks["4_checkpoint_loads"] = True


# --------------------------------------------------------------------------- #
# Stage 5 -- deterministic test input
# --------------------------------------------------------------------------- #


def make_synthetic_scene() -> tuple[np.ndarray, np.ndarray]:
    """A 512x512 scene with one obvious object, and a boolean mask over it.

    No RNG anywhere: closed-form arithmetic over np.mgrid only, so both arrays
    are byte-identical on every run and every machine.

    The mask is produced as a **boolean H x W array** -- PixelForge's internal
    mask contract, and exactly what SAM 2 produces after the cast recorded in
    Phase 3 -- so this test exercises the real SAM 2 -> Moebius handoff.
    """
    yy, xx = np.mgrid[0:IMG_H, 0:IMG_W].astype(np.float64)

    # Background: a vertical gradient plus a sinusoidal texture, so the region
    # under the mask is genuinely non-trivial to reconstruct (a flat background
    # would be recoverable by any blur and would prove nothing).
    base = 60.0 + 110.0 * (yy / IMG_H)
    texture = 26.0 * np.sin(xx / 17.0) * np.cos(yy / 23.0)
    bg = base + texture

    img = np.empty((IMG_H, IMG_W, 3), np.float64)
    img[..., 0] = bg * 0.85
    img[..., 1] = bg * 0.95
    img[..., 2] = bg * 1.10

    # Two horizontal bars crossing the frame, so a correct inpaint has to
    # continue visible structure through the removed region.
    for y0, y1 in ((150, 168), (300, 316)):
        img[y0:y1, :, :] = np.array([210.0, 196.0, 150.0])

    # The object to be removed: a shaded disc, clearly foreign to the scene.
    cx, cy = DISC_CENTER
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    disc = dist <= DISC_RADIUS
    shade = 1.0 - 0.45 * (dist / max(DISC_RADIUS, 1)).clip(0, 1)
    img[disc, 0] = (225.0 * shade)[disc]
    img[disc, 1] = (70.0 * shade)[disc]
    img[disc, 2] = (55.0 * shade)[disc]

    image = img.clip(0, 255).astype(np.uint8)

    # Boolean mask, dilated slightly beyond the disc as a real selection would
    # be, still fully deterministic.
    mask_bool = dist <= (DISC_RADIUS + 6)
    return image, mask_bool


def write_test_input(rec: Record) -> tuple[Any, Any, np.ndarray]:
    from PIL import Image

    image, mask_bool = make_synthetic_scene()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # PixelForge's internal contract is a boolean H x W array. Moebius wants a
    # uint8 0/255 grayscale 'L' image: infer/utils_dataset.py:169 opens masks
    # with .convert("L"), and pipeline.py:188 binarizes at 255/2. This is the
    # exact conversion the SAM 2 -> Moebius handoff will perform.
    mask_u8 = (mask_bool.astype(np.uint8)) * 255

    img_path = OUTPUT_DIR / "input_synthetic.png"
    mask_path = OUTPUT_DIR / "input_mask.png"
    Image.fromarray(image).save(img_path)
    Image.fromarray(mask_u8, mode="L").save(mask_path)

    # Round-trip, so what the model sees is what is on disk.
    pil_img = Image.open(img_path).convert("RGB")
    pil_mask = Image.open(mask_path).convert("L")
    if not np.array_equal(np.array(pil_img), image):
        raise GateFailure("test_input", "PNG round-trip altered the image")
    if not np.array_equal(np.array(pil_mask), mask_u8):
        raise GateFailure("test_input", "PNG round-trip altered the mask")

    area = int(mask_bool.sum())
    rec.test_input = {
        "image_path": str(img_path),
        "mask_path": str(mask_path),
        "resolution": f"{IMG_W}x{IMG_H}",
        "image_dtype": "uint8",
        "image_sha256": _sha256(img_path),
        "mask_sha256": _sha256(mask_path),
        "image_array_sha256": hashlib.sha256(image.tobytes()).hexdigest(),
        "mask_array_sha256": hashlib.sha256(mask_u8.tobytes()).hexdigest(),
        "determinism": "no RNG; closed-form arithmetic over np.mgrid only",
        "object": f"shaded disc r={DISC_RADIUS} at {DISC_CENTER}",
        "internal_mask_dtype": "bool (PixelForge H x W contract)",
        "delivered_mask_dtype": "uint8 grayscale 'L', values {0, 255}",
        "mask_polarity": "WHITE(255) = region to INPAINT; BLACK(0) = region to KEEP",
        "mask_polarity_evidence": [
            "pipeline.py:232 mask = np.asarray(input_mask)/255. -> {0.,1.}",
            "pipeline.py:175-176 mask=where(mask>=0.5,1,0); masked_image=image*(1-mask) "
            "-> content is ZEROED where the mask is white, i.e. white is generated",
            "pipeline.py:252 ours_np = ours_np*m_img + (1-m_img)*img_np -> white takes "
            "GENERATED pixels, black takes ORIGINAL pixels",
        ],
        "mask_area_pixels": area,
        "mask_area_percent": round(100.0 * area / (IMG_W * IMG_H), 4),
        "ground_truth_available": False,
        "metric_note": (
            "No ground truth exists for a synthetic scene, so no PSNR/SSIM/LPIPS/"
            "FID or any other reference-dependent metric is reported. This gate "
            "tests that valid inpainting EXECUTES, not that it is accurate."
        ),
    }
    rec.artifacts["input_image"] = str(img_path)
    rec.artifacts["input_mask"] = str(mask_path)
    return pil_img, pil_mask, mask_bool


# --------------------------------------------------------------------------- #
# Stage 6 -- build the pipeline
# --------------------------------------------------------------------------- #


def verify_upstream_imports(rec: Record) -> None:
    """Import every upstream symbol the inference path needs, before touching
    weights.

    Kept as its own stage on purpose: whether the code *imports* is independent
    of whether checkpoints happen to be on disk. Folding it into pipeline
    construction would let a missing checkpoint mask an import problem, and
    would make an import failure (class A/B) look like a checkpoint failure
    (class E).
    """
    try:
        from diffusers import DDIMScheduler  # noqa: F401
        from diffusers.models import AutoencoderKL

        from removal.v1_2 import (  # noqa: F401
            build_removal_model,
            load_cfg,
            load_removal_model,
        )
        from removal.v1_2.pipeline import (  # noqa: F401
            RemovalSDXLPipeline_BatchMode,
        )
    except Exception as exc:
        rec.checks["3_vae_import"] = False
        raise GateFailure(
            "import_strategy",
            f"upstream inference imports failed: {type(exc).__name__}: {exc}",
            classification="B",
        ) from exc

    rec.import_strategy["removal_v1_2_imported_normally"] = True
    rec.import_strategy["removal_v1_2_note"] = (
        "removal.v1_2 is imported NORMALLY so upstream's own package __init__ "
        "runs verbatim (it also pulls .dataset -> transformers). Exactly ONE "
        "surrogate exists, for exactly one unavoidable reason."
    )
    rec.import_strategy["vae_class"] = AutoencoderKL.__name__
    rec.import_strategy["vae_class_module"] = AutoencoderKL.__module__
    rec.import_strategy["imports_verified_before_checkpoints"] = True

    # Re-check contamination: removal.v1_2's own __init__ has now run, which is
    # the import most likely to pull the teacher in behind our back.
    leaked = {
        "fla": "fla" in sys.modules,
        "triton": "triton" in sys.modules,
        "unet_gla": any("unet_gla" in m for m in sys.modules),
        "layers.gla": any("layers.gla" in m for m in sys.modules),
    }
    rec.import_strategy["teacher_contamination_check_after_removal_import"] = leaked
    rec.import_strategy["teacher_free_after_removal_import"] = not any(leaked.values())
    if any(leaked.values()):
        raise GateFailure(
            "import_strategy",
            f"teacher/CUDA contamination appeared while importing removal.v1_2: {leaked}",
            classification="B",
        )

    rec.checks["3_vae_import"] = True
    confirm_upstream_untouched(rec)


def build_pipeline(rec: Record, device: str):
    """Replicate upstream ``infer/utils.py:build_pipeline`` statement for
    statement, with two deviations, both recorded:

      1. ``device`` is ``mps`` instead of ``cuda``. ``--device`` is a plain CLI
         argument upstream, so this is device placement, not an algorithm change.
      2. ``build_vae`` is inlined instead of imported. Importing it pulls
         ``utils_train`` -> ``library/{train_util,chinese_sdxl_train_util}.py``,
         i.e. the training stack (tensorboard, toml, orjson, CLIP/Bert/T5
         tokenizers), purely to reach a ONE-LINE function. ``build_vae``'s
         complete body is ``AutoencoderKL.from_pretrained(cfg['vae']['model_dir'])``
         (utils_train.py:27-29), reproduced verbatim below.

    Everything else -- the config, the model builder, the weight loader, the
    scheduler arguments, the pipeline class, the dtype -- is upstream's own.
    """
    import torch
    from diffusers import DDIMScheduler
    from diffusers.models import AutoencoderKL

    dtype = torch.float32  # infer/utils.py:79 passes dtype=torch.float

    try:
        from removal.v1_2.pipeline import (
            RemovalSDXLPipeline_BatchMode as Removal_Pipeline,
        )
        from removal.v1_2 import build_removal_model, load_cfg, load_removal_model
    except Exception as exc:
        raise GateFailure(
            "import_strategy",
            f"upstream removal.v1_2 import failed: {type(exc).__name__}: {exc}",
            classification="B",
        ) from exc

    # ---- config -------------------------------------------------------- #
    # infer/utils.py:65 -- load_cfg(path) -> dict, then build_removal_model
    # receives that dict, which routes load_cfg through OmegaConf. Upstream's
    # exact sequence is preserved.
    model_cfg = load_cfg(str(MODEL_CONFIG))

    # The yaml's vae.model_dir is './weight/vae' -- CWD-relative, so it only
    # resolves if you happen to run from the repo root. Overriding it to an
    # absolute path outside the READ-ONLY tree is a path fix, not a model change.
    original_vae_dir = model_cfg["vae"]["model_dir"]
    model_cfg["vae"]["model_dir"] = str(VAE_DIR)

    rec.inference_config = {
        "config_file": str(MODEL_CONFIG),
        "device": device,
        "dtype": DTYPE_NAME,
        "batch_size": BATCH_SIZE,
        "image_size": IMAGE_SIZE,
        "num_steps": NUM_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "noise_offset": NOISE_OFFSET,
        "paste": PASTE,
        "compensate": COMPENSATE,
        "strength": STRENGTH,
        "num_embeddings": NUM_EMBEDDINGS,
        "scheduler": {"class": "DDIMScheduler", **SCHEDULER_KWARGS},
        "vae_model_dir_original": original_vae_dir,
        "vae_model_dir_used": str(VAE_DIR),
        "seed": 0,
        "seed_source": (
            "pipeline.py:347-350 seeds random, np.random and torch.manual_seed "
            "with seed=0 when retry=0, so upstream inference is deterministic "
            "by default."
        ),
        "provenance": (
            "Every value above is an upstream default read from "
            "infer/utils.py argparse, pipeline.py signature defaults, or "
            "config/model_cfg/moebius.yaml. Only 'device' and the two paths "
            "differ from an upstream CUDA run."
        ),
        "determinism_caveat": (
            "MPS and CPU use different RNG streams, so torch.manual_seed(0) "
            "does not make MPS output bit-identical to CUDA or CPU output. "
            "Runs are reproducible per-device, not across devices."
        ),
    }

    # ---- student model ------------------------------------------------- #
    t0 = time.perf_counter()
    removal_model = build_removal_model(model_cfg, NUM_EMBEDDINGS)
    build_seconds = time.perf_counter() - t0

    n_params = sum(p.numel() for p in removal_model.parameters())
    rec.model_load = {
        "student_class": STUDENT_CLASS,
        "build_seconds": round(build_seconds, 4),
        "parameter_count": n_params,
        "parameter_count_millions": round(n_params / 1e6, 2),
        "documented_parameter_count_millions": DOCUMENTED_PARAMS_MILLIONS,
        "num_embeddings": removal_model.num_embeddings,
        "embedding_shape": list(removal_model.embedding_layer.weight.shape),
        "unet_in_channels": removal_model.diff_model.config.in_channels,
        "unet_out_channels": removal_model.diff_model.config.out_channels,
        "unet_sample_size": removal_model.diff_model.config.sample_size,
        "encoder_hid_dim": removal_model.diff_model.config.encoder_hid_dim,
        "text_encoder_required": False,
        "text_encoder_note": (
            "input_ids are indices into a learned nn.Embedding(20, 3072) "
            "(removal_model.py:14-18), not tokens. No tokenizer, no CLIP and no "
            "text encoder is on the inference path."
        ),
    }

    # ---- weights ------------------------------------------------------- #
    t0 = time.perf_counter()
    msg = load_removal_model(removal_model, str(STUDENT_WEIGHT), device, dtype)
    load_seconds = time.perf_counter() - t0
    rec.model_load["weight_load_seconds"] = round(load_seconds, 4)
    rec.model_load["model_load_seconds"] = round(build_seconds + load_seconds, 4)
    rec.model_load["load_state_dict_message"] = str(msg)
    rec.model_load["missing_keys"] = list(getattr(msg, "missing_keys", []) or [])
    rec.model_load["unexpected_keys"] = list(getattr(msg, "unexpected_keys", []) or [])

    if rec.model_load["missing_keys"] or rec.model_load["unexpected_keys"]:
        raise GateFailure(
            "model_load",
            f"state_dict mismatch: {len(rec.model_load['missing_keys'])} missing, "
            f"{len(rec.model_load['unexpected_keys'])} unexpected",
            classification="E",
        )

    # Prove the weights are actually trained values, not random init. Upstream
    # loads with strict=True so a mismatch would raise, but a silent no-op
    # would not -- compare against the file directly.
    ref = torch.load(STUDENT_WEIGHT, map_location="cpu", weights_only=True)
    live = dict(removal_model.state_dict())
    matched = mismatched = 0
    for k, v in ref.items():
        if k in live and hasattr(v, "shape"):
            if torch.allclose(live[k].detach().to("cpu", torch.float32),
                              v.to(torch.float32), atol=0, rtol=0):
                matched += 1
            else:
                mismatched += 1
    del ref
    rec.model_load["checkpoint_tensors_matched"] = matched
    rec.model_load["checkpoint_tensors_mismatched"] = mismatched
    rec.model_load["weights_verified_loaded"] = matched > 0 and mismatched == 0
    if not rec.model_load["weights_verified_loaded"]:
        raise GateFailure(
            "model_load",
            f"weights not verifiably loaded ({matched} matched, {mismatched} "
            f"mismatched) -- any output would be attributable to random init",
            classification="E",
        )

    devs = {str(p.device) for p in removal_model.parameters()}
    dts = {str(p.dtype) for p in removal_model.parameters()}
    rec.model_load["parameter_devices"] = sorted(devs)
    rec.model_load["parameter_dtypes"] = sorted(dts)
    rec.model_load["on_requested_device"] = all(d.startswith(device) for d in devs)
    if not rec.model_load["on_requested_device"]:
        raise GateFailure(
            "model_load", f"model is on {sorted(devs)}, expected {device}"
        )
    rec.checks["5_model_moves_to_mps"] = device == "mps"

    # ---- VAE ----------------------------------------------------------- #
    # utils_train.py:27-29 build_vae, complete body, inlined.
    t0 = time.perf_counter()
    vae = AutoencoderKL.from_pretrained(model_cfg["vae"]["model_dir"])
    vae_load_seconds = time.perf_counter() - t0
    vae.to(device=device, dtype=dtype)

    vae_devs = {str(p.device) for p in vae.parameters()}
    rec.vae_load = {
        "source_directory": str(VAE_DIR),
        "class": type(vae).__name__,
        "vae_load_seconds": round(vae_load_seconds, 4),
        "parameter_count": sum(p.numel() for p in vae.parameters()),
        "parameter_count_millions": round(
            sum(p.numel() for p in vae.parameters()) / 1e6, 2
        ),
        "block_out_channels": list(vae.config.block_out_channels),
        "scaling_factor": float(vae.config.scaling_factor),
        "latent_channels": int(vae.config.latent_channels),
        "downsample_ratio": 2 ** (len(vae.config.block_out_channels) - 1),
        "parameter_devices": sorted(vae_devs),
        "parameter_dtypes": sorted({str(p.dtype) for p in vae.parameters()}),
        "on_requested_device": all(d.startswith(device) for d in vae_devs),
        "build_vae_inlined": True,
        "build_vae_inline_reason": (
            "Importing utils_train.build_vae pulls library/train_util.py and "
            "chinese_sdxl_train_util.py -- the training stack -- to reach a "
            "one-line function. Its complete body "
            "(AutoencoderKL.from_pretrained(cfg['vae']['model_dir'])) is "
            "reproduced verbatim instead."
        ),
    }
    if not rec.vae_load["on_requested_device"]:
        raise GateFailure(
            "vae_load", f"VAE is on {sorted(vae_devs)}, expected {device}"
        )
    rec.checks["6_vae_moves_to_mps"] = device == "mps"

    # ---- scheduler + pipeline ------------------------------------------ #
    scheduler = DDIMScheduler(**SCHEDULER_KWARGS)
    try:
        pipe = Removal_Pipeline(
            removal_model=removal_model,
            vae=vae,
            scheduler=scheduler,
            device=device,
            dtype=dtype,
        )
    except Exception as exc:
        raise GateFailure(
            "pipeline_init",
            f"pipeline init failed: {type(exc).__name__}: {exc}",
            classification="G",
        ) from exc

    rec.model_load["pipeline_class"] = type(pipe).__name__
    rec.model_load["pipeline_vae_ds_ratio"] = pipe.vae_ds_ratio
    rec.model_load["pipeline_input_ids_shape"] = list(pipe.input_ids.shape)
    rec.model_load["pipeline_input_ids_device"] = str(pipe.input_ids.device)
    rec.checks["7_pipeline_initializes"] = True

    confirm_upstream_untouched(rec)
    return pipe


# --------------------------------------------------------------------------- #
# Stage 7 -- inference
# --------------------------------------------------------------------------- #


def _identify_failing_op(exc: BaseException, tb_text: str) -> str | None:
    """Extract the operator name from an MPS/aten error, for classification."""
    for pat in (
        r"aten::([a-zA-Z0-9_.]+)",
        r"The operator '([^']+)'",
        r"MPS.*?operator[: ]+([a-zA-Z0-9_.:]+)",
        r"not (?:currently )?implemented for '?([A-Za-z0-9_]+)'?",
        r"NotImplementedError: ([^\n]{0,120})",
    ):
        m = re.search(pat, f"{exc}\n{tb_text}")
        if m:
            return m.group(1)
    return None


def _classify_runtime_failure(exc: BaseException, tb_text: str) -> str:
    """Map a runtime failure onto the directive's taxonomy. Recorded, not guessed."""
    blob = f"{type(exc).__name__}: {exc}\n{tb_text}".lower()
    if "not implemented for" in blob or "notimplementederror" in blob or (
        "mps" in blob and "operator" in blob
    ):
        return "C"
    if "dtype" in blob or "float64" in blob or "double" in blob or "half" in blob:
        return "D"
    if "out of memory" in blob or "cannot allocate" in blob or "mps_malloc" in blob:
        return "F"
    if "size mismatch" in blob or "state_dict" in blob or "shape" in blob:
        return "E"
    if "modulenotfounderror" in blob or "importerror" in blob:
        return "A"
    return "G"


def run_inference(rec: Record, pipe, pil_img, pil_mask) -> Any:
    import torch

    device = rec.device["selected"]
    call_kwargs = dict(
        image_size=IMAGE_SIZE,
        num_steps=NUM_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        noise_offset=NOISE_OFFSET,
        paste=PASTE,
        compensate=COMPENSATE,
        strength=STRENGTH,
        mute=True,
        visualize=False,
    )

    # Cold run -- includes Metal shader compilation, which Phase 3 measured at
    # roughly a 23x one-off cost for SAM 2. Reported separately for that reason.
    _sync(torch, device)
    t0 = time.perf_counter()
    try:
        with torch.no_grad():
            out_list = pipe([pil_img], [pil_mask], **call_kwargs)
        _sync(torch, device)
        cold = time.perf_counter() - t0
    except Exception as exc:
        tb = traceback.format_exc()
        rec.failing_operation = _identify_failing_op(exc, tb)
        rec.checks["8_inference_executes"] = False
        raise GateFailure(
            "inference",
            f"{type(exc).__name__}: {exc}",
            classification=_classify_runtime_failure(exc, tb),
        ) from exc

    rec.checks["8_inference_executes"] = True

    warm: list[float] = []
    for _ in range(WARM_ITERS):
        _sync(torch, device)
        t = time.perf_counter()
        with torch.no_grad():
            pipe([pil_img], [pil_mask], **call_kwargs)
        _sync(torch, device)
        warm.append(time.perf_counter() - t)

    rec.inference = {
        "batch_size": BATCH_SIZE,
        "returned_count": len(out_list),
        "first_inference_seconds": round(cold, 4),
        "warm_iterations": WARM_ITERS,
        "warm_inference_seconds_best": round(min(warm), 4),
        "warm_inference_seconds_mean": round(sum(warm) / len(warm), 4),
        "warm_inference_seconds_all": [round(w, 4) for w in warm],
        "seconds_per_step_warm": round(min(warm) / NUM_STEPS, 4),
        "cold_warm_ratio": round(cold / min(warm), 2) if min(warm) > 0 else None,
        "no_cpu_fallback": device == "mps",
    }
    if len(out_list) != BATCH_SIZE:
        raise GateFailure(
            "inference",
            f"expected {BATCH_SIZE} output(s), got {len(out_list)}",
            classification="G",
        )
    return out_list[0]


def verify_output(rec: Record, result, mask_bool: np.ndarray) -> None:
    from PIL import Image

    arr = np.asarray(result)

    finite = bool(np.isfinite(arr.astype(np.float64)).all())
    rec.checks["9_output_is_finite"] = finite

    valid_dims = (
        arr.ndim == 3
        and arr.shape[2] == 3
        and arr.shape[0] == IMG_H
        and arr.shape[1] == IMG_W
    )
    rec.checks["10_output_valid_dimensions"] = valid_dims

    inside = arr[mask_bool]
    outside = arr[~mask_bool]
    rec.output = {
        "type": type(result).__name__,
        "mode": getattr(result, "mode", None),
        "size": list(getattr(result, "size", [])),
        "array_shape": list(arr.shape),
        "array_dtype": str(arr.dtype),
        "is_finite": finite,
        "has_valid_dimensions": valid_dims,
        "expected_dimensions": f"{IMG_W}x{IMG_H}x3",
        "value_min": int(arr.min()),
        "value_max": int(arr.max()),
        "value_mean": round(float(arr.mean()), 4),
        "is_constant": bool(arr.min() == arr.max()),
        "unique_value_count": int(np.unique(arr).size),
        "masked_region_mean": round(float(inside.mean()), 4),
        "unmasked_region_mean": round(float(outside.mean()), 4),
        "masked_region_std": round(float(inside.std()), 4),
    }

    if not finite:
        raise GateFailure(
            "output", "output contains NaN or Inf", classification="D"
        )
    if not valid_dims:
        raise GateFailure(
            "output",
            f"output shape {arr.shape} != expected ({IMG_H}, {IMG_W}, 3)",
            classification="G",
        )
    if rec.output["is_constant"]:
        raise GateFailure(
            "output",
            "output is a constant image -- inference produced no content",
            classification="G",
        )

    # PNG round-trip.
    out_path = OUTPUT_DIR / "output_inpainted.png"
    result.save(out_path)
    reread = np.array(Image.open(out_path).convert("RGB"))
    png_ok = out_path.is_file() and np.array_equal(reread, arr)
    rec.checks["11_output_writes_png"] = bool(png_ok)
    rec.output["png_path"] = str(out_path)
    rec.output["png_size_bytes"] = out_path.stat().st_size if out_path.is_file() else None
    rec.output["png_roundtrip_lossless"] = bool(png_ok)
    rec.artifacts["output_image"] = str(out_path)
    if not png_ok:
        raise GateFailure("output", "output PNG round-trip failed")

    # A side-by-side, purely for human inspection. Not a metric.
    try:
        img_in = np.array(Image.open(rec.test_input["image_path"]).convert("RGB"))
        m3 = np.repeat((mask_bool[..., None] * 255).astype(np.uint8), 3, axis=2)
        strip = np.concatenate([img_in, m3, arr], axis=1)
        strip_path = OUTPUT_DIR / "comparison_before_mask_after.png"
        Image.fromarray(strip).save(strip_path)
        rec.artifacts["comparison_image"] = str(strip_path)
    except Exception as exc:  # cosmetic only
        rec.notes.append(f"comparison strip not written: {type(exc).__name__}: {exc}")

    rec.output["accuracy_claim"] = (
        "NONE. No ground truth exists, so no reference-dependent metric is "
        "reported and no claim of real-world inpainting quality is made. This "
        "gate establishes that valid inpainting executes on MPS, nothing more."
    )


def record_memory(rec: Record, device: str) -> None:
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # bytes on macOS
    rec.memory = {
        "process_peak_rss_bytes": peak,
        "process_peak_rss_mib": round(peak / 1024**2, 2),
        "unified_memory_bytes": rec.hardware.get("unified_memory_bytes"),
        "note": (
            "Apple Silicon shares one memory pool between CPU and GPU, so "
            "process RSS and MPS allocator figures overlap and must NOT be summed."
        ),
    }
    if device == "mps":
        import torch

        for name, fn in (
            ("mps_current_allocated_memory", "current_allocated_memory"),
            ("mps_driver_allocated_memory", "driver_allocated_memory"),
            ("mps_recommended_max_memory", "recommended_max_memory"),
        ):
            try:
                val = getattr(torch.mps, fn)()
                rec.memory[f"{name}_bytes"] = int(val)
                rec.memory[f"{name}_mib"] = round(val / 1024**2, 2)
            except Exception:
                rec.memory[f"{name}_bytes"] = None


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def run_validation(allow_cpu_fallback: bool = False) -> Record:
    rec = Record()
    device = "unknown"
    try:
        verify_pinned_repository(rec)
        device = verify_device(rec, allow_cpu_fallback=allow_cpu_fallback)
        install_student_import_surrogate(rec)
        verify_upstream_imports(rec)
        pil_img, pil_mask, mask_bool = write_test_input(rec)
        verify_checkpoints(rec)
        pipe = build_pipeline(rec, device)
        result = run_inference(rec, pipe, pil_img, pil_mask)
        verify_output(rec, result, mask_bool)
        record_memory(rec, device)
        confirm_upstream_untouched(rec)

        all_checks = all(rec.checks[k] for k in MANDATED_CHECKS)
        if device != "mps":
            rec.verdict = "PARTIAL"
        elif not all_checks:
            rec.verdict = "FAIL"
        elif not rec.repository.get("upstream_unmodified"):
            rec.verdict = "FAIL"
            rec.notes.append(
                "Upstream tree was modified during the run -- READ-ONLY rule "
                "violated, so the result is not valid."
            )
        else:
            # The import surrogate is a documented compatibility workaround that
            # does not alter the research method, which is precisely the
            # directive's CONDITIONAL definition. PASS is reserved for a run that
            # needs no workaround at all -- only reachable if fla/Triton become
            # installable on Apple Silicon and the teacher imports cleanly.
            rec.verdict = "CONDITIONAL" if not rec.torch_info.get(
                "fla_installed"
            ) else "PASS"

    except GateFailure as exc:
        rec.failure_stage = exc.stage
        rec.failure_type = type(exc).__name__
        rec.failure_message = str(exc)
        rec.failure_traceback = traceback.format_exc()
        rec.failure_classification = (
            f"{exc.classification} -- {FAILURE_CLASSES[exc.classification]}"
            if exc.classification
            else None
        )
        rec.verdict = "PARTIAL" if exc.blocking else "FAIL"
        if exc.blocking:
            rec.notes.append(
                "BLOCKED, NOT REFUTED: a precondition was missing, so Moebius "
                "MPS capability was neither confirmed nor refuted. This is not "
                "evidence that Moebius works on MPS, and it is not evidence "
                "that it does not."
            )
        else:
            rec.notes.append(
                "The upstream repository was NOT patched in response to this "
                "failure, and no CPU fallback was silently substituted."
            )
    except BaseException as exc:  # noqa: BLE001 -- must capture the exact traceback
        tb = traceback.format_exc()
        rec.failure_stage = "inference" if rec.model_load else "setup"
        rec.failure_type = type(exc).__name__
        rec.failure_message = str(exc)
        rec.failure_traceback = tb
        rec.failing_operation = _identify_failing_op(exc, tb)
        cls = _classify_runtime_failure(exc, tb)
        rec.failure_classification = f"{cls} -- {FAILURE_CLASSES[cls]}"
        rec.verdict = "FAIL"
        rec.notes.append(
            "The upstream repository was NOT patched in response to this "
            "failure, and no CPU fallback was silently substituted."
        )

    try:
        confirm_upstream_untouched(rec)
    except Exception:
        pass

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rec.notes.insert(
        0,
        {
            "PASS": "True MPS inference produced a valid inpainting result with "
            "no compatibility workaround required.",
            "CONDITIONAL": "True MPS inference produced a valid inpainting "
            "result, but only with a documented compatibility workaround (the "
            "student-only import surrogate) that does NOT alter the research "
            "method.",
            "PARTIAL": "Inconclusive. A precondition was missing (no Metal "
            "device / no checkpoint) or CPU was used. MPS capability is NOT "
            "established either way.",
            "FAIL": "The pipeline ran and did not work, or would require "
            "modifying the research algorithm -- a real negative result about "
            "this commit on this hardware.",
            "NOT_RUN": "Validation did not start.",
        }[rec.verdict],
    )
    result_path = OUTPUT_DIR / "result.json"
    rec.artifacts["result_json"] = str(result_path)
    result_path.write_text(rec.to_json())
    return rec


def _print_summary(rec: Record) -> None:
    print("\n" + "=" * 72)
    print(f"  Moebius student MPS validation -- VERDICT: {rec.verdict}")
    print("=" * 72)
    e, t = rec.environment, rec.torch_info
    print(f"  python        {e.get('python_version')}  ({e.get('conda_env')})")
    print(f"  torch         {t.get('torch_version')}   diffusers {t.get('diffusers_version')}")
    print(f"  mps built     {t.get('mps_is_built')}    available {t.get('mps_is_available')}")
    print(f"  metal device  {rec.hardware.get('metal_default_device_creatable')}")
    print(f"  commit        {rec.repository.get('observed_commit')}")
    print(f"  device        {rec.device.get('selected')}")
    if rec.import_strategy:
        print(f"  teacher-free  {rec.import_strategy.get('teacher_free')}  "
              f"(contamination: {rec.import_strategy.get('teacher_contamination_check')})")
    if rec.model_load:
        print(f"  student       {rec.model_load.get('parameter_count_millions')} M params  "
              f"load {rec.model_load.get('model_load_seconds')} s")
        print(f"  weights ok    {rec.model_load.get('weights_verified_loaded')}  "
              f"({rec.model_load.get('checkpoint_tensors_matched')} tensors matched)")
    if rec.vae_load:
        print(f"  vae           {rec.vae_load.get('parameter_count_millions')} M params  "
              f"load {rec.vae_load.get('vae_load_seconds')} s")
    if rec.inference:
        print(f"  first infer   {rec.inference.get('first_inference_seconds')} s")
        print(f"  warm infer    {rec.inference.get('warm_inference_seconds_best')} s  "
              f"({rec.inference.get('seconds_per_step_warm')} s/step)")
    if rec.output:
        print(f"  output        {rec.output.get('size')} {rec.output.get('mode')}  "
              f"finite={rec.output.get('is_finite')}  png={rec.output.get('png_roundtrip_lossless')}")
    if rec.memory:
        print(f"  peak RSS      {rec.memory.get('process_peak_rss_mib')} MiB   "
              f"mps driver {rec.memory.get('mps_driver_allocated_memory_mib')} MiB")
    print("  -- mandated checks --")
    for k in MANDATED_CHECKS:
        v = rec.checks[k]
        print(f"       {'PASS' if v else ('FAIL' if v is False else 'NOT REACHED'):11s} {k}")
    if rec.verdict not in ("PASS", "CONDITIONAL"):
        print(f"\n  failure stage {rec.failure_stage}")
        print(f"  failure       {rec.failure_type}: {rec.failure_message}")
        print(f"  class         {rec.failure_classification}")
        if rec.failing_operation:
            print(f"  failing op    {rec.failing_operation}")
    print(f"\n  result.json   {OUTPUT_DIR / 'result.json'}")
    print("=" * 72 + "\n")


def test_moebius_mps_inference() -> None:
    """pytest entry point. Fails unless real MPS inference succeeded."""
    rec = run_validation(allow_cpu_fallback=False)
    _print_summary(rec)
    assert rec.verdict in ("PASS", "CONDITIONAL"), (
        f"verdict={rec.verdict} stage={rec.failure_stage} "
        f"class={rec.failure_classification} op={rec.failing_operation} "
        f"msg={rec.failure_message}"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--allow-cpu-fallback",
        action="store_true",
        help="Permit CPU when MPS is unusable. Caps the verdict at PARTIAL; never PASS.",
    )
    args = ap.parse_args()
    rec = run_validation(allow_cpu_fallback=args.allow_cpu_fallback)
    _print_summary(rec)
    return 0 if rec.verdict in ("PASS", "CONDITIONAL") else 1


if __name__ == "__main__":
    raise SystemExit(main())
