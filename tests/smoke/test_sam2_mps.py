"""SAM 2.1 Hiera-Tiny — Apple MPS reproducibility gate (PixelForge Phase 3).

Answers exactly one question: can the *pinned* SAM 2 commit run real image
segmentation on this machine using Apple MPS?

Design constraints this file is built around
--------------------------------------------
1.  The upstream repository is READ-ONLY. SAM 2 is imported by putting the
    pinned checkout on ``sys.path`` -- it is deliberately NOT pip-installed,
    because ``pip install -e`` would write ``*.egg-info`` into the upstream
    tree. Bytecode writing is disabled for the same reason, so the upstream
    checkout stays byte-identical (verified: 0 changed files before and after).
2.  No silent CPU fallback. MPS is required by default. CPU is only used when
    ``--allow-cpu-fallback`` is passed explicitly, and in that case the verdict
    is forced to PARTIAL and can never be PASS.
3.  Nothing is fabricated. Every number in ``result.json`` is measured. Fields
    that could not be measured are ``null``, never guessed.
4.  The test image is generated from pure closed-form arithmetic with no RNG,
    so it is byte-identical on every run and on every machine. Its SHA-256 is
    recorded to prove that.

Run directly::

    python tests/smoke/test_sam2_mps.py

or under pytest::

    pytest tests/smoke/test_sam2_mps.py -v -s
"""

from __future__ import annotations

# Disable bytecode writing BEFORE importing anything from the upstream tree, so
# that importing SAM 2 cannot create __pycache__/ directories inside it.
import sys

sys.dont_write_bytecode = True

import argparse
import hashlib
import json
import os
import platform
import re
import resource
import subprocess
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# --------------------------------------------------------------------------- #
# Pinned facts. Every one of these is read from the repository, never assumed.
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SAM2_REPO = PROJECT_ROOT / "research" / "upstream" / "sam2"

# research/upstream/LOCKFILE.md -- Phase 2 pin, verified twice.
SAM2_PINNED_COMMIT = "2b90b9f5ceec907a1c18123530e92e794ad901a4"

# research/upstream/sam2/README.md:167 pairs this config with this checkpoint.
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_t.yaml"
CHECKPOINT_NAME = "sam2.1_hiera_tiny.pt"

# README.md:167 column header is "Size (M)" -- 38.9 is the PARAMETER COUNT IN
# MILLIONS, not an on-disk byte size. The repository documents no file size and
# no checksum for any checkpoint, so neither is asserted here. The parameter
# count, however, is a genuine cross-check against the loaded model.
DOCUMENTED_PARAMS_MILLIONS = 38.9
PARAM_COUNT_TOLERANCE = 0.05  # 5% -- absorbs any counting convention difference

# research/upstream/sam2/checkpoints/download_ckpts.sh:40-41
CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
)

# Weights live outside the upstream tree, in a git-ignored project directory.
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "sam2" / CHECKPOINT_NAME
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "sam2_mps_validation"

# Synthetic test image geometry -- fixed, so the image is reproducible.
IMG_W, IMG_H = 640, 480
DISC_CENTER = (400, 240)  # (x, y) -- also the point prompt
DISC_RADIUS = 90
# A single click on a solid disc should recover roughly pi*r^2 / (W*H) == 8.3%.
# These bounds are deliberately loose: they test "a plausible object-sized mask
# was produced", not a reference-dependent accuracy metric (no ground truth
# exists here, so no IoU is reported).
MASK_AREA_MIN_PCT = 1.0
MASK_AREA_MAX_PCT = 60.0

WARM_ITERS = 3


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
    environment: dict[str, Any] = field(default_factory=dict)
    torch_info: dict[str, Any] = field(default_factory=dict)
    hardware: dict[str, Any] = field(default_factory=dict)
    repository: dict[str, Any] = field(default_factory=dict)
    checkpoint: dict[str, Any] = field(default_factory=dict)
    device: dict[str, Any] = field(default_factory=dict)
    model_load: dict[str, Any] = field(default_factory=dict)
    cases: list[dict[str, Any]] = field(default_factory=list)
    memory: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, sort_keys=False, default=str)


class GateFailure(RuntimeError):
    """A validation gate was not met. Carries the stage that failed.

    ``blocking=True`` marks an *environmental precondition* that was absent --
    no Metal device visible, no checkpoint on disk. Nothing about SAM 2 was
    disproven, so the run is inconclusive (PARTIAL) rather than negative (FAIL).
    ``blocking=False`` means the pipeline actually ran and produced a wrong or
    unusable result, which is a genuine FAIL.
    """

    def __init__(self, stage: str, message: str, blocking: bool = False) -> None:
        super().__init__(message)
        self.stage = stage
        self.blocking = blocking


# --------------------------------------------------------------------------- #
# Stage 1 -- pinned repository verification
# --------------------------------------------------------------------------- #


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(SAM2_REPO), *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def verify_pinned_repository(rec: Record) -> None:
    if not SAM2_REPO.is_dir():
        raise GateFailure("repository", f"upstream SAM 2 clone missing at {SAM2_REPO}")

    head = _git("rev-parse", "HEAD")
    head_again = _git("rev-parse", "HEAD")  # verified twice, per project policy
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    dirty = [ln for ln in _git("status", "--porcelain").splitlines() if ln.strip()]

    rec.repository = {
        "path": str(SAM2_REPO),
        "expected_commit": SAM2_PINNED_COMMIT,
        "observed_commit": head,
        "observed_commit_second_pass": head_again,
        "commit_matches_pin": head == SAM2_PINNED_COMMIT == head_again,
        "branch": branch,
        "changed_files": len(dirty),
        "working_tree_clean": not dirty,
        "installed_into_environment": False,
        "import_mechanism": "sys.path insertion (no pip install, upstream untouched)",
    }

    if head != SAM2_PINNED_COMMIT or head_again != SAM2_PINNED_COMMIT:
        raise GateFailure(
            "repository",
            f"commit mismatch: expected {SAM2_PINNED_COMMIT}, observed {head} / {head_again}",
        )
    if dirty:
        raise GateFailure(
            "repository",
            f"upstream working tree is dirty ({len(dirty)} files) -- READ-ONLY rule violated: {dirty[:5]}",
        )


# --------------------------------------------------------------------------- #
# Stage 2 -- device verification
# --------------------------------------------------------------------------- #


def _metal_device_available() -> bool | None:
    """Probe Metal directly.

    ``torch.backends.mps.is_available()`` returning False is ambiguous: it also
    reports a misleading "macOS 14.0+" message when the process simply has no
    access to a GPU device (e.g. inside a sandbox). Probing
    ``MTLCreateSystemDefaultDevice`` distinguishes "OS too old" from "no GPU
    visible to this process", which is the difference between a real
    incompatibility and an environment problem.
    """
    try:
        import ctypes

        lib = ctypes.CDLL("/System/Library/Frameworks/Metal.framework/Metal")
        lib.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
        return bool(lib.MTLCreateSystemDefaultDevice())
    except Exception:
        return None


def verify_device(rec: Record, require_mps: bool, allow_cpu_fallback: bool) -> str:
    import torch
    import torchvision

    rec.environment = {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
    }
    rec.torch_info = {
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "numpy_version": np.__version__,
        "mps_is_built": bool(torch.backends.mps.is_built()),
        "mps_is_available": bool(torch.backends.mps.is_available()),
        "cuda_is_available": bool(torch.cuda.is_available()),
    }
    rec.hardware = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": _sysctl("machdep.cpu.brand_string"),
        "physical_cores": _sysctl("hw.physicalcpu"),
        "logical_cores": _sysctl("hw.logicalcpu"),
        "unified_memory_bytes": _sysctl("hw.memsize"),
        "macos_product_version": _sw_vers("productVersion"),
        "macos_build": _sw_vers("buildVersion"),
        "metal_default_device_creatable": _metal_device_available(),
    }

    for maj, minr in ((13, 0), (14, 0), (15, 0)):
        try:
            from torch._C import _mps_is_on_macos_or_newer as probe

            rec.torch_info[f"mps_is_on_macos_{maj}_{minr}_or_newer"] = bool(probe(maj, minr))
        except Exception:
            break

    mps_ok = bool(torch.backends.mps.is_built() and torch.backends.mps.is_available())

    # A real tensor operation on MPS -- is_available() alone is not evidence.
    tensor_op: dict[str, Any] = {"attempted": mps_ok, "succeeded": False}
    if mps_ok:
        try:
            x = torch.ones((2, 2), device="mps")
            y = x @ x + 1.5
            _sync(torch)
            tensor_op.update(
                succeeded=True,
                expression="torch.ones((2,2), device='mps') @ itself + 1.5",
                value=y.detach().cpu().tolist(),
                expected=[[3.5, 3.5], [3.5, 3.5]],
                matches_expected=np.allclose(y.detach().cpu().numpy(), 3.5),
            )
        except Exception as exc:
            tensor_op.update(
                succeeded=False,
                error_type=type(exc).__name__,
                error=str(exc),
                traceback=traceback.format_exc(),
            )
            mps_ok = False
    rec.device["mps_tensor_op"] = tensor_op

    if mps_ok:
        rec.device["selected"] = "mps"
        return "mps"

    diagnosis = _diagnose_mps_unavailable(rec)
    rec.device["mps_unavailable_diagnosis"] = diagnosis

    if require_mps and not allow_cpu_fallback:
        raise GateFailure(
            "device",
            f"MPS not usable and CPU fallback not permitted. {diagnosis}",
            # No Metal device visible is an environment problem: it tells us
            # nothing about whether SAM 2 works on MPS. A PyTorch build with no
            # MPS support at all is likewise not a SAM 2 result.
            blocking=True,
        )

    rec.device["selected"] = "cpu"
    rec.notes.append(
        "CPU fallback was used. Per the Phase 3 directive this caps the verdict at "
        "PARTIAL -- MPS inference is NOT validated by this run."
    )
    return "cpu"


def _diagnose_mps_unavailable(rec: Record) -> str:
    t = rec.torch_info
    hw = rec.hardware
    if not t.get("mps_is_built"):
        return "This PyTorch build has no MPS support (mps.is_built() is False). Wrong wheel for Apple Silicon."
    if hw.get("metal_default_device_creatable") is False:
        return (
            "MPS is built and the OS is new enough, but MTLCreateSystemDefaultDevice() "
            "returned NULL: no Metal GPU device is visible to this process. This is an "
            "environment/sandbox restriction, NOT a SAM 2 or PyTorch incompatibility. "
            "Re-run from a normal terminal session with GPU access."
        )
    if t.get("mps_is_on_macos_14_0_or_newer") is False:
        return f"macOS {hw.get('macos_product_version')} is older than the 14.0 required by this PyTorch MPS build."
    return "MPS reported unavailable; no more specific cause could be determined."


def _sysctl(key: str) -> str | None:
    try:
        return subprocess.run(
            ["sysctl", "-n", key], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return None


def _sw_vers(key: str) -> str | None:
    try:
        return subprocess.run(
            ["sw_vers", f"-{key}"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return None


def _sync(torch_mod) -> None:
    if hasattr(torch_mod, "mps") and hasattr(torch_mod.mps, "synchronize"):
        try:
            torch_mod.mps.synchronize()
        except Exception:
            pass


# --------------------------------------------------------------------------- #
# Stage 3 -- checkpoint verification (never downloaded by this test)
# --------------------------------------------------------------------------- #


def verify_checkpoint(rec: Record) -> Path:
    rec.checkpoint = {
        "name": CHECKPOINT_NAME,
        "expected_path": str(CHECKPOINT_PATH),
        "documented_source_url": CHECKPOINT_URL,
        "documented_source_file": "research/upstream/sam2/checkpoints/download_ckpts.sh:40-41",
        "paired_config": SAM2_CONFIG,
        "upstream_published_checksum": None,  # none is published upstream; do not invent one
        "present": CHECKPOINT_PATH.is_file(),
    }
    if not CHECKPOINT_PATH.is_file():
        raise GateFailure(
            "checkpoint",
            f"checkpoint not found at {CHECKPOINT_PATH}. This test never downloads weights. "
            f"Fetch it first:\n  mkdir -p {CHECKPOINT_PATH.parent}\n"
            f"  curl -L -o {CHECKPOINT_PATH} {CHECKPOINT_URL}",
            blocking=True,  # weights absent -- nothing about SAM 2 was tested
        )

    size = CHECKPOINT_PATH.stat().st_size
    h = hashlib.sha256()
    with CHECKPOINT_PATH.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    rec.checkpoint.update(
        size_bytes=size,
        size_mib=round(size / (1024 * 1024), 2),
        observed_sha256=h.hexdigest(),
        observed_sha256_note=(
            "SHA-256 of the file as downloaded on this host. Upstream publishes no "
            "checksum, so this is a local integrity anchor for future re-acquisition, "
            "not verification against an upstream-published value."
        ),
    )
    if size < 10 * 1024 * 1024:
        raise GateFailure(
            "checkpoint",
            f"checkpoint at {CHECKPOINT_PATH} is only {size} bytes -- truncated or an error page, not weights.",
        )
    return CHECKPOINT_PATH


# --------------------------------------------------------------------------- #
# Stage 4 -- deterministic test image
# --------------------------------------------------------------------------- #


def make_synthetic_image() -> np.ndarray:
    """Build a deterministic RGB test image with no RNG whatsoever.

    A solid disc on a gradient background with a mild sinusoidal texture and one
    rectangular distractor. Closed-form arithmetic only, so the result is
    byte-identical on every run and every machine.
    """
    yy, xx = np.mgrid[0:IMG_H, 0:IMG_W].astype(np.float64)

    # Background: vertical gradient + gentle deterministic texture.
    bg_r = 30 + 60 * (yy / (IMG_H - 1))
    bg_g = 45 + 50 * (yy / (IMG_H - 1))
    bg_b = 70 + 40 * (xx / (IMG_W - 1))
    texture = 6.0 * np.sin(xx / 23.0) * np.cos(yy / 31.0)
    img = np.stack([bg_r + texture, bg_g + texture, bg_b + texture], axis=-1)

    # Foreground disc -- the segmentation target.
    cx, cy = DISC_CENTER
    disc = ((xx - cx) ** 2 + (yy - cy) ** 2) <= float(DISC_RADIUS) ** 2
    shade = 1.0 - 0.25 * (((xx - cx) ** 2 + (yy - cy) ** 2) / float(DISC_RADIUS) ** 2)
    img[disc] = np.stack(
        [225 * shade, 140 * shade, 55 * shade], axis=-1
    )[disc]

    # Distractor rectangle, well away from the prompt.
    img[330:430, 60:200] = np.array([70, 190, 120], dtype=np.float64)

    return np.clip(img, 0, 255).astype(np.uint8)


def write_test_image(rec: Record) -> tuple[Path, np.ndarray]:
    from PIL import Image

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "input_synthetic.png"
    arr = make_synthetic_image()
    Image.fromarray(arr).save(path, format="PNG", optimize=False)

    # Reload from disk -- the test must segment an image it actually loaded.
    loaded = np.array(Image.open(path).convert("RGB"))
    if not np.array_equal(arr, loaded):
        raise GateFailure("test_image", "PNG round-trip was not lossless; image is not reproducible")

    rec.artifacts["input_image"] = str(path)
    rec.artifacts["input_image_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    rec.artifacts["input_image_array_sha256"] = hashlib.sha256(arr.tobytes()).hexdigest()
    return path, loaded


# --------------------------------------------------------------------------- #
# Stage 5 -- inference
# --------------------------------------------------------------------------- #


def load_model(rec: Record, device: str, ckpt: Path):
    import torch

    sys.path.insert(0, str(SAM2_REPO))
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    try:  # informational only -- see note below
        from sam2 import _C  # noqa: F401

        c_ext = True
    except Exception:
        c_ext = False

    # build_sam2 accepts **kwargs and never reads them, so a misspelled keyword
    # is silently ignored -- yielding a randomly-initialized model on the default
    # device ("cuda") with no error raised. Every keyword below is therefore
    # checked against the verbatim signature, and the outcome is asserted after
    # the call rather than trusted.
    import inspect

    accepted = set(inspect.signature(build_sam2).parameters)
    kwargs = {
        "config_file": SAM2_CONFIG,
        "ckpt_path": str(ckpt),
        "device": device,
        "mode": "eval",  # any other value silently leaves the model in train mode
        # Left at the upstream default. It only injects the dynamic-multimask
        # stability hydra overrides (build_sam.py:81-88); it does not touch _C.
        "apply_postprocessing": True,
    }
    unknown = sorted(set(kwargs) - accepted)
    if unknown:
        raise GateFailure(
            "model_load",
            f"build_sam2() does not accept {unknown}; it would silently ignore them "
            f"and return a random-weight CUDA model. Accepted: {sorted(accepted)}",
        )

    t0 = time.perf_counter()
    model = build_sam2(**kwargs)
    _sync(torch)
    load_s = time.perf_counter() - t0

    predictor = SAM2ImagePredictor(model)  # max_hole_area/max_sprinkle_area default to 0.0

    params = sum(p.numel() for p in model.parameters())
    dtypes = {str(p.dtype) for p in model.parameters()}

    # Assertion 1: the model actually landed on the requested device.
    actual_device = str(next(model.parameters()).device)
    if not actual_device.startswith(device):
        raise GateFailure(
            "model_load",
            f"requested device '{device}' but model parameters are on '{actual_device}'",
        )

    # Assertion 2: real weights were loaded, not random initialization. If
    # ckpt_path were ever dropped, build_sam2 would return an untrained model and
    # still produce a plausible-looking mask -- so compare a real tensor.
    raw = torch.load(str(ckpt), map_location="cpu", weights_only=True)
    sd = raw.get("model", raw)
    named = dict(model.named_parameters())
    matched = mismatched = 0
    for key, ref in sd.items():
        p = named.get(key)
        if p is not None and tuple(p.shape) == tuple(ref.shape):
            if torch.allclose(p.detach().float().cpu(), ref.float(), atol=1e-6):
                matched += 1
            else:
                mismatched += 1
    del raw, sd
    if matched == 0:
        raise GateFailure(
            "model_load",
            "no model parameter matches the checkpoint file -- weights were NOT "
            "loaded, so any mask produced would come from random initialization",
        )

    # Assertion 3: parameter count agrees with the documented 38.9 M.
    documented = DOCUMENTED_PARAMS_MILLIONS * 1e6
    if abs(params - documented) / documented > PARAM_COUNT_TOLERANCE:
        raise GateFailure(
            "model_load",
            f"parameter count {params:,} disagrees with the documented "
            f"{DOCUMENTED_PARAMS_MILLIONS} M for sam2.1_hiera_tiny "
            f"(README.md:167) -- wrong config/checkpoint pairing?",
        )
    rec.model_load = {
        "config": SAM2_CONFIG,
        "model_load_seconds": round(load_s, 4),
        "parameter_count": params,
        "parameter_count_millions": round(params / 1e6, 2),
        "documented_parameter_count_millions": DOCUMENTED_PARAMS_MILLIONS,
        "parameter_count_matches_documented": True,
        "parameter_dtypes": sorted(dtypes),
        "model_device": actual_device,
        "device_requested": device,
        "device_matches_request": True,
        "checkpoint_tensors_matched": matched,
        "checkpoint_tensors_mismatched": mismatched,
        "weights_verified_loaded": matched > 0,
        "weights_verification_note": (
            "Tensors from the checkpoint file were compared element-wise against the "
            "built model's parameters. This proves the mask came from trained weights "
            "rather than random initialization -- build_sam2 silently ignores unknown "
            "keyword arguments, so a dropped ckpt_path would otherwise go unnoticed."
        ),
        "apply_postprocessing": True,
        "c_extension_available": c_ext,
        "c_extension_note": (
            "The compiled _C extension is not built (it is a CUDA-only "
            "torch.CUDAExtension over sam2/csrc/connected_components.cu, and "
            "setup.py's SAM2_BUILD_ALLOW_ERRORS defaults to '1', so a no-CUDA "
            "install succeeds with an empty ext_modules list). It is only reached "
            "from SAM2Transforms.postprocess_masks when max_hole_area>0 or "
            "max_sprinkle_area>0; both default to 0.0 for SAM2ImagePredictor, and "
            "the call site is additionally wrapped in try/except. INSTALL.md states "
            "that skipping it loses only hole/sprinkle removal and that both image "
            "and video applications still work. It therefore cannot affect this test."
        ),
        "dtype_policy": (
            "float32, no autocast. Neither set_image nor predict uses torch.autocast "
            "and no dtype coercion occurs in the image path, so float32 is the "
            "upstream default rather than a choice imposed here. Every autocast "
            "example in the repository is hardcoded to device_type='cuda'. Running "
            "float32 on MPS is therefore the faithful configuration, not a "
            "modification of the research algorithm."
        ),
    }
    return predictor


def _identify_failing_op(exc: BaseException, tb_text: str) -> str | None:
    """Best-effort extraction of the specific operator that failed on MPS."""
    msg = f"{exc}\n{tb_text}"
    patterns = [
        r"The operator '([^']+)' is not currently implemented for the MPS backend",
        r"(aten::[A-Za-z0-9_.]+)",
        r"not implemented for '([^']+)'",
        r"MPS(?:\s+backend)?[^.\n]*?does not support ([A-Za-z0-9_.:]+)",
    ]
    for pat in patterns:
        m = re.search(pat, msg)
        if m:
            return m.group(1)
    frames = [ln.strip() for ln in tb_text.splitlines() if "/sam2/" in ln and ", line " in ln]
    return f"innermost sam2 frame: {frames[-1]}" if frames else None


def run_case(
    rec: Record,
    predictor,
    name: str,
    image: np.ndarray,
    point_xy: tuple[int, int],
    gating: bool,
) -> dict[str, Any]:
    """Run one point-prompted segmentation and record measurements."""
    import torch
    from PIL import Image

    # set_image runs torchvision ToTensor(), which applies the /255 rescale ONLY
    # for uint8 input. A float array here would be normalized against the wrong
    # scale and silently degrade the mask, so the dtype is a hard precondition.
    if image.dtype != np.uint8:
        raise GateFailure(
            "inference",
            f"[{name}] image dtype is {image.dtype}; SAM 2's ToTensor() rescale "
            f"requires uint8 or normalization is silently wrong",
        )

    # (X, Y) in pixels; label 1 == foreground (predict() docstring).
    coords = np.array([[point_xy[0], point_xy[1]]], dtype=np.float32)
    labels = np.array([1], dtype=np.int32)

    case: dict[str, Any] = {
        "name": name,
        "gating": gating,
        "image_dtype": str(image.dtype),
        "image_shape_hwc": list(image.shape),
        "image_resolution": f"{image.shape[1]}x{image.shape[0]}",
        "prompt_type": "single positive point",
        "prompt_point_xy": list(point_xy),
        "prompt_label": 1,
        "multimask_output": False,
        "internal_inference_resolution": "1024x1024 (config image_size; square resize, aspect ratio not preserved)",
        "succeeded": False,
    }

    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass

    t0 = time.perf_counter()
    predictor.set_image(image)
    _sync(torch)
    case["set_image_seconds_cold"] = round(time.perf_counter() - t0, 4)

    t0 = time.perf_counter()
    masks, scores, low_res = predictor.predict(
        point_coords=coords,
        point_labels=labels,
        multimask_output=False,
    )
    _sync(torch)
    case["predict_seconds_cold"] = round(time.perf_counter() - t0, 4)
    case["total_inference_seconds_cold"] = round(
        case["set_image_seconds_cold"] + case["predict_seconds_cold"], 4
    )

    # Steady-state timing, so a cold-start cost is never reported as latency.
    warm_set, warm_pred = [], []
    for _ in range(WARM_ITERS):
        t0 = time.perf_counter()
        predictor.set_image(image)
        _sync(torch)
        warm_set.append(time.perf_counter() - t0)
        t0 = time.perf_counter()
        predictor.predict(point_coords=coords, point_labels=labels, multimask_output=False)
        _sync(torch)
        warm_pred.append(time.perf_counter() - t0)
    case.update(
        warm_iterations=WARM_ITERS,
        set_image_seconds_warm_best=round(min(warm_set), 4),
        set_image_seconds_warm_mean=round(float(np.mean(warm_set)), 4),
        predict_seconds_warm_best=round(min(warm_pred), 4),
        predict_seconds_warm_mean=round(float(np.mean(warm_pred)), 4),
        total_inference_seconds_warm_best=round(min(warm_set) + min(warm_pred), 4),
    )

    # predict() returns float32 0.0/1.0 in CxHxW (sam2_image_predictor.py:300),
    # so cast explicitly to reach PixelForge's boolean H x W mask contract.
    case["returned_masks_dtype"] = str(masks.dtype)
    case["returned_masks_shape"] = list(masks.shape)
    case["returned_scores"] = [round(float(s), 6) for s in np.atleast_1d(scores)]
    case["returned_low_res_shape"] = list(low_res.shape)

    mask = np.asarray(masks[0]).astype(bool)
    if mask.shape != image.shape[:2]:
        raise GateFailure(
            "inference", f"mask shape {mask.shape} does not match image {image.shape[:2]}"
        )

    area = int(mask.sum())
    total = int(mask.size)
    pct = 100.0 * area / total
    ys, xs = np.nonzero(mask)
    case.update(
        mask_dtype_after_cast="bool",
        mask_shape_hw=list(mask.shape),
        mask_area_pixels=area,
        mask_total_pixels=total,
        mask_area_percent=round(pct, 4),
        mask_nonzero=area > 0,
        mask_bbox_xyxy=(
            [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if area else None
        ),
        mask_contains_prompt_point=bool(mask[point_xy[1], point_xy[0]]) if area else False,
    )

    if area == 0:
        raise GateFailure("inference", f"[{name}] mask is empty -- zero area, no segmentation produced")
    if gating and not (MASK_AREA_MIN_PCT <= pct <= MASK_AREA_MAX_PCT):
        raise GateFailure(
            "inference",
            f"[{name}] mask area {pct:.2f}% outside plausible range "
            f"[{MASK_AREA_MIN_PCT}, {MASK_AREA_MAX_PCT}]%",
        )

    # Artifacts.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mask_path = OUTPUT_DIR / f"{name}_mask.png"
    overlay_path = OUTPUT_DIR / f"{name}_overlay.png"
    Image.fromarray((mask.astype(np.uint8) * 255)).save(mask_path)
    Image.fromarray(_make_overlay(image, mask, point_xy)).save(overlay_path)
    case["mask_path"] = str(mask_path)
    case["overlay_path"] = str(overlay_path)
    case["succeeded"] = True
    return case


def _make_overlay(image: np.ndarray, mask: np.ndarray, point_xy: tuple[int, int]) -> np.ndarray:
    """Red 45% tint inside the mask, yellow boundary, cyan crosshair at the prompt."""
    out = image.astype(np.float64).copy()
    tint = np.array([255.0, 40.0, 40.0])
    out[mask] = 0.55 * out[mask] + 0.45 * tint

    m = mask.astype(np.int16)
    edge = np.zeros_like(mask)
    edge[1:, :] |= (m[1:, :] != m[:-1, :])
    edge[:-1, :] |= (m[1:, :] != m[:-1, :])
    edge[:, 1:] |= (m[:, 1:] != m[:, :-1])
    edge[:, :-1] |= (m[:, 1:] != m[:, :-1])
    out[edge] = np.array([255.0, 230.0, 0.0])

    px, py = point_xy
    h, w = mask.shape
    for d in range(-7, 8):
        if 0 <= py + d < h:
            out[py + d, max(0, px - 1) : min(w, px + 2)] = np.array([0.0, 255.0, 255.0])
        if 0 <= px + d < w:
            out[max(0, py - 1) : min(h, py + 2), px + d] = np.array([0.0, 255.0, 255.0])
    return np.clip(out, 0, 255).astype(np.uint8)


def record_memory(rec: Record, device: str) -> None:
    import torch

    mem: dict[str, Any] = {
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "unified_memory_bytes": rec.hardware.get("unified_memory_bytes"),
        "note": (
            "Apple Silicon shares one memory pool between CPU and GPU, so process "
            "RSS and MPS allocator figures overlap and must not be summed."
        ),
    }
    mem["process_peak_rss_mib"] = round(mem["process_peak_rss_bytes"] / (1024 * 1024), 2)
    if device == "mps" and hasattr(torch, "mps"):
        for attr in (
            "current_allocated_memory",
            "driver_allocated_memory",
            "recommended_max_memory",
        ):
            fn = getattr(torch.mps, attr, None)
            if callable(fn):
                try:
                    val = int(fn())
                    mem[f"mps_{attr}_bytes"] = val
                    mem[f"mps_{attr}_mib"] = round(val / (1024 * 1024), 2)
                except Exception:
                    mem[f"mps_{attr}_bytes"] = None
    rec.memory = mem


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #


def run_validation(allow_cpu_fallback: bool = False) -> Record:
    rec = Record()
    device = "unknown"
    try:
        verify_pinned_repository(rec)
        device = verify_device(rec, require_mps=True, allow_cpu_fallback=allow_cpu_fallback)
        ckpt = verify_checkpoint(rec)
        _, synthetic = write_test_image(rec)
        predictor = load_model(rec, device, ckpt)

        rec.cases.append(
            run_case(rec, predictor, "synthetic_disc", synthetic, DISC_CENTER, gating=True)
        )

        # Secondary, non-gating: a natural photograph already present in the
        # pinned checkout (read-only, nothing downloaded). A solid synthetic disc
        # alone is weak evidence that real segmentation happened.
        natural = SAM2_REPO / "notebooks" / "images" / "truck.jpg"
        if natural.is_file():
            from PIL import Image

            arr = np.array(Image.open(natural).convert("RGB"))
            h, w = arr.shape[:2]
            rec.artifacts["natural_image"] = str(natural)
            rec.cases.append(
                run_case(rec, predictor, "natural_truck", arr, (w // 2, h // 2), gating=False)
            )
        else:
            rec.notes.append(f"secondary natural-image case skipped: {natural} not present")

        record_memory(rec, device)

        if device != "mps":
            rec.verdict = "PARTIAL"
        else:
            rec.verdict = "PASS"

    except GateFailure as exc:
        rec.failure_stage = exc.stage
        rec.failure_type = type(exc).__name__
        rec.failure_message = str(exc)
        rec.failure_traceback = traceback.format_exc()
        # An absent precondition is inconclusive, not negative. Reporting FAIL
        # here would assert that SAM 2 does not work on MPS -- a claim this run
        # produced no evidence for either way.
        rec.verdict = "PARTIAL" if exc.blocking else "FAIL"
        if exc.blocking:
            rec.notes.append(
                "BLOCKED, NOT REFUTED: a precondition was missing, so MPS capability "
                "was neither confirmed nor refuted. This is not evidence that SAM 2 "
                "works on MPS, and it is not evidence that it does not."
            )
    except BaseException as exc:  # noqa: BLE001 -- must capture the exact traceback
        tb = traceback.format_exc()
        rec.failure_stage = "inference" if rec.model_load else "setup"
        rec.failure_type = type(exc).__name__
        rec.failure_message = str(exc)
        rec.failure_traceback = tb
        rec.failing_operation = _identify_failing_op(exc, tb)
        rec.verdict = "FAIL"
        rec.notes.append(
            "Upstream SAM 2 was NOT modified in response to this failure, and no CPU "
            "fallback was silently substituted."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rec.notes.insert(
        0,
        {
            "PASS": "MPS segmentation inference actually completed and produced a non-empty mask.",
            "PARTIAL": "Inconclusive. Either a precondition was missing (no Metal device / no "
            "checkpoint) or CPU was used. MPS capability is NOT established.",
            "FAIL": "The pipeline ran and did not work -- a real negative result about this "
            "commit on this hardware.",
            "NOT_RUN": "Validation did not start.",
        }[rec.verdict],
    )
    (OUTPUT_DIR / "result.json").write_text(rec.to_json())
    rec.artifacts["result_json"] = str(OUTPUT_DIR / "result.json")
    (OUTPUT_DIR / "result.json").write_text(rec.to_json())
    return rec


def _print_summary(rec: Record) -> None:
    print("\n" + "=" * 68)
    print(f"  SAM 2.1 Hiera-Tiny MPS validation -- VERDICT: {rec.verdict}")
    print("=" * 68)
    print(f"  python        {rec.environment.get('python_version')}  ({rec.environment.get('conda_env')})")
    print(f"  torch         {rec.torch_info.get('torch_version')}   torchvision {rec.torch_info.get('torchvision_version')}")
    print(f"  mps built     {rec.torch_info.get('mps_is_built')}     available {rec.torch_info.get('mps_is_available')}")
    print(f"  sam2 commit   {rec.repository.get('observed_commit')}")
    print(f"  device        {rec.device.get('selected')}")
    if rec.model_load:
        print(f"  params        {rec.model_load.get('parameter_count'):,}")
        print(f"  model load    {rec.model_load.get('model_load_seconds')} s")
    for c in rec.cases:
        print(f"  -- case {c['name']} ({'gating' if c['gating'] else 'informational'})")
        print(f"       resolution {c['image_resolution']}  prompt {c['prompt_point_xy']}")
        print(f"       cold set_image {c.get('set_image_seconds_cold')} s + predict {c.get('predict_seconds_cold')} s")
        print(f"       warm best      {c.get('total_inference_seconds_warm_best')} s")
        print(f"       mask area      {c.get('mask_area_pixels')} px  ({c.get('mask_area_percent')}%)  score {c.get('returned_scores')}")
    if rec.memory:
        print(f"  peak RSS      {rec.memory.get('process_peak_rss_mib')} MiB")
    if rec.verdict != "PASS":
        print(f"\n  failure stage {rec.failure_stage}")
        print(f"  failure       {rec.failure_type}: {rec.failure_message}")
        if rec.failing_operation:
            print(f"  failing op    {rec.failing_operation}")
    print(f"\n  result.json   {OUTPUT_DIR / 'result.json'}")
    print("=" * 68 + "\n")


def test_sam2_mps_inference() -> None:
    """pytest entry point. Fails unless MPS inference actually succeeded."""
    rec = run_validation(allow_cpu_fallback=False)
    _print_summary(rec)
    assert rec.verdict == "PASS", (
        f"verdict={rec.verdict} stage={rec.failure_stage} "
        f"op={rec.failing_operation} msg={rec.failure_message}"
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
    return 0 if rec.verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
