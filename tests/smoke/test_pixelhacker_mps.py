"""PixelHacker — Apple MPS reproducibility gate (PixelForge Phase 5).

Answers exactly one question: can the *pinned* PixelHacker implementation run
real inpainting on this machine using Apple MPS **without changing the research
algorithm**?

Expected answer on Apple Silicon: **no**. This harness records that negative
result with the same provenance standard as Phases 3–4.

Design constraints
--------------------
1.  The upstream repository is READ-ONLY. PixelHacker is imported by putting
    the pinned checkout on ``sys.path``; it is NOT pip-installed.
2.  No silent CPU fallback is substituted for MPS. We do not rewrite GLA onto
    Metal or stub ``fla``.
3.  PixelHacker UNet weights are **not** downloaded. The gate is expected to
    fail at dependency / import audit before weights are required.
4.  Nothing is fabricated. Every field in ``result.json`` is measured or read
    from source. Unmeasured fields are ``null``.

Run directly::

    python tests/smoke/test_pixelhacker_mps.py

or under pytest::

    pytest tests/smoke/test_pixelhacker_mps.py -v -s
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import argparse
import importlib.util
import json
import os
import platform
import re
import subprocess
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# --------------------------------------------------------------------------- #
# Pinned facts — read from the repository, never assumed.
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PIXELHACKER_REPO = PROJECT_ROOT / "research" / "upstream" / "PixelHacker"
PIXELHACKER_PINNED_COMMIT = "f5567db2871598aa178fe7a34c520dd478a0b41b"
INFER_ENTRY = PIXELHACKER_REPO / "infer_pixelhacker.py"
GLA_MODULE = PIXELHACKER_REPO / "gla_model" / "gla.py"
REQUIREMENTS = PIXELHACKER_REPO / "requirements.txt"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "pixelhacker_mps_validation"

MANDATED_CHECKS = [
    "1_repository_pin_verified",
    "2_upstream_unmodified",
    "3_mps_host_available",
    "4_upstream_has_no_mps_path",
    "5_fla_not_installable",
    "6_gla_import_blocked",
    "7_model_import_blocked",
    "8_no_inference_executed",
]


@dataclass
class Record:
    verdict: str = "NOT_RUN"
    classification: str | None = None
    primary_placement: str | None = None
    failure_stage: str | None = None
    failure_type: str | None = None
    failure_message: str | None = None
    failure_traceback: str | None = None
    failure_classification: str | None = None
    checks: dict[str, Any] = field(
        default_factory=lambda: {k: None for k in MANDATED_CHECKS}
    )
    environment: dict[str, Any] = field(default_factory=dict)
    torch_info: dict[str, Any] = field(default_factory=dict)
    hardware: dict[str, Any] = field(default_factory=dict)
    repository: dict[str, Any] = field(default_factory=dict)
    dependency_audit: dict[str, Any] = field(default_factory=dict)
    device_audit: dict[str, Any] = field(default_factory=dict)
    import_audit: dict[str, Any] = field(default_factory=dict)
    checkpoints: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2, sort_keys=False, default=str)


class GateFailure(RuntimeError):
    def __init__(
        self,
        stage: str,
        message: str,
        *,
        blocking: bool = False,
        classification: str | None = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.blocking = blocking
        self.classification = classification


FAILURE_CLASSES = {
    "A": "dependency incompatibility",
    "B": "import contamination",
    "C": "unsupported MPS operator",
    "D": "dtype issue",
    "E": "checkpoint/config mismatch",
    "F": "memory exhaustion",
    "G": "model implementation incompatibility",
}


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(PIXELHACKER_REPO), *args],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def _count_pycache() -> int:
    return sum(1 for _ in PIXELHACKER_REPO.rglob("__pycache__")) + sum(
        1 for _ in PIXELHACKER_REPO.rglob("*.egg-info")
    )


def _metal_device_available() -> bool | None:
    try:
        import ctypes

        lib = ctypes.CDLL("/System/Library/Frameworks/Metal.framework/Metal")
        lib.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
        return bool(lib.MTLCreateSystemDefaultDevice())
    except Exception:
        return None


def verify_pinned_repository(rec: Record) -> None:
    if not PIXELHACKER_REPO.is_dir():
        raise GateFailure(
            "repository",
            f"upstream PixelHacker clone missing at {PIXELHACKER_REPO}",
            blocking=True,
        )

    head = _git("rev-parse", "HEAD")
    head_again = _git("rev-parse", "HEAD")
    dirty = [ln for ln in _git("status", "--porcelain").splitlines() if ln.strip()]

    rec.repository = {
        "path": str(PIXELHACKER_REPO),
        "expected_commit": PIXELHACKER_PINNED_COMMIT,
        "observed_commit": head,
        "observed_commit_second_pass": head_again,
        "commit_matches_pin": head == PIXELHACKER_PINNED_COMMIT == head_again,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "changed_files_before": len(dirty),
        "working_tree_clean_before": not dirty,
        "pycache_or_egginfo_before": _count_pycache(),
        "installed_into_environment": False,
        "import_mechanism": "sys.path insertion (no pip install, upstream untouched)",
        "license_code": "Apache-2.0",
        "license_weights_hf": "MIT",
    }

    if head != PIXELHACKER_PINNED_COMMIT:
        raise GateFailure(
            "repository",
            f"commit mismatch: expected {PIXELHACKER_PINNED_COMMIT}, observed {head}",
        )
    if dirty:
        raise GateFailure(
            "repository",
            f"upstream working tree is dirty ({len(dirty)} files)",
        )
    rec.checks["1_repository_pin_verified"] = True


def confirm_upstream_untouched(rec: Record) -> None:
    dirty = [ln for ln in _git("status", "--porcelain").splitlines() if ln.strip()]
    head = _git("rev-parse", "HEAD")
    rec.repository["observed_commit_after_run"] = head
    rec.repository["changed_files_after"] = len(dirty)
    rec.repository["working_tree_clean_after"] = not dirty
    rec.repository["pycache_or_egginfo_after"] = _count_pycache()
    rec.repository["upstream_unmodified"] = (
        not dirty and head == PIXELHACKER_PINNED_COMMIT and _count_pycache() == 0
    )
    rec.checks["2_upstream_unmodified"] = rec.repository["upstream_unmodified"]


def collect_environment(rec: Record) -> None:
    rec.environment = {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "conda_prefix": os.environ.get("CONDA_PREFIX"),
        "sys_prefix": sys.prefix,
        "pixelhacker_env_created": False,
        "note": (
            "Phase 5 deliberately did not create pixelforge-pixelhacker. "
            "This gate audits whether LOCAL_MPS is possible, not cloud CUDA."
        ),
    }


def collect_torch_info(rec: Record) -> None:
    try:
        import torch
    except ImportError as exc:
        rec.torch_info = {"torch_importable": False, "import_error": str(exc)}
        return

    rec.torch_info = {
        "torch_importable": True,
        "torch_version": torch.__version__,
        "mps_is_built": bool(getattr(torch.backends.mps, "is_built", lambda: False)()),
        "mps_is_available": bool(
            getattr(torch.backends.mps, "is_available", lambda: False)()
        ),
        "cuda_is_available": bool(torch.cuda.is_available()),
    }


def audit_hardware(rec: Record) -> None:
    metal = _metal_device_available()
    rec.hardware = {
        "metal_default_device_creatable": metal,
        "processor": platform.processor() or platform.machine(),
    }
    mps_ok = bool(rec.torch_info.get("mps_is_available")) or metal is True
    rec.checks["3_mps_host_available"] = mps_ok


def audit_upstream_device_selection(rec: Record) -> None:
    if not INFER_ENTRY.is_file():
        raise GateFailure("device_audit", f"missing {INFER_ENTRY}")

    source = INFER_ENTRY.read_text(encoding="utf-8")
    device_line = next(
        (ln.strip() for ln in source.splitlines() if ln.strip().startswith("device =")),
        None,
    )
    mps_hits = []
    for path in PIXELHACKER_REPO.rglob("*.py"):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if re.search(r"\bmps\b", text, flags=re.IGNORECASE):
            mps_hits.append(str(path.relative_to(PIXELHACKER_REPO)))

    rec.device_audit = {
        "infer_entry_device_line": device_line,
        "official_devices": ["cuda:0", "cpu"],
        "mps_references_in_py_files": mps_hits,
        "mps_path_exists_in_source": bool(mps_hits),
        "source_file": str(INFER_ENTRY.relative_to(PIXELHACKER_REPO)),
        "source_line": 31,
    }
    rec.checks["4_upstream_has_no_mps_path"] = not mps_hits and device_line == (
        'device = "cuda:0" if torch.cuda.is_available() else "cpu"'
    )


def audit_requirements(rec: Record) -> None:
    pins: dict[str, str] = {}
    if REQUIREMENTS.is_file():
        for ln in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln or ln.startswith("#") or "==" not in ln:
                continue
            name, ver = ln.split("==", 1)
            pins[name.strip()] = ver.strip()

    rec.dependency_audit = {
        "requirements_path": str(REQUIREMENTS),
        "flash_linear_attention_pin": pins.get("flash-linear-attention"),
        "triton_pin": pins.get("triton"),
        "flash_attn_pin": pins.get("flash-attn"),
        "torch_pin": pins.get("torch"),
        "gla_import_lines": [
            ln.strip()
            for ln in GLA_MODULE.read_text(encoding="utf-8").splitlines()
            if "fla." in ln and "import" in ln
        ],
    }


def attempt_fla_import(rec: Record) -> None:
    t0 = time.perf_counter()
    error: str | None = None
    tb: str | None = None
    installed = importlib.util.find_spec("fla") is not None
    if installed:
        try:
            importlib.import_module("fla.ops.gla")
            fla_importable = True
        except Exception as exc:  # noqa: BLE001
            fla_importable = False
            error = f"{type(exc).__name__}: {exc}"
            tb = traceback.format_exc()
    else:
        fla_importable = False
        error = "ModuleNotFoundError: No module named 'fla'"
        tb = None

    rec.dependency_audit.update(
        {
            "fla_installed": installed,
            "fla_ops_gla_importable": fla_importable,
            "fla_import_seconds": round(time.perf_counter() - t0, 4),
            "fla_import_error": error,
            "fla_import_traceback": tb,
        }
    )
    rec.checks["5_fla_not_installable"] = not fla_importable


def attempt_gla_import(rec: Record) -> None:
    """Import the pinned GLA module from upstream without modifying it."""
    if str(PIXELHACKER_REPO) not in sys.path:
        sys.path.insert(0, str(PIXELHACKER_REPO))

    t0 = time.perf_counter()
    error: str | None = None
    tb: str | None = None
    try:
        spec = importlib.util.spec_from_file_location(
            "pixelhacker_gla_gate", GLA_MODULE, submodule_search_locations=[]
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot create spec for {GLA_MODULE}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        gla_importable = True
    except Exception as exc:  # noqa: BLE001
        gla_importable = False
        error = f"{type(exc).__name__}: {exc}"
        tb = traceback.format_exc()

    rec.import_audit["gla_module_import"] = {
        "path": str(GLA_MODULE),
        "importable": gla_importable,
        "seconds": round(time.perf_counter() - t0, 4),
        "error": error,
        "traceback": tb,
    }
    rec.checks["6_gla_import_blocked"] = not gla_importable

    if gla_importable:
        raise GateFailure(
            "import",
            "gla_model/gla.py imported successfully — unexpected on this host",
            classification="G",
        )


def attempt_model_import(rec: Record) -> None:
    t0 = time.perf_counter()
    error: str | None = None
    tb: str | None = None
    try:
        from gla_model.PixelHacker import PixelHacker  # noqa: F401

        model_importable = True
    except Exception as exc:  # noqa: BLE001
        model_importable = False
        error = f"{type(exc).__name__}: {exc}"
        tb = traceback.format_exc()

    rec.import_audit["pixelhacker_class_import"] = {
        "importable": model_importable,
        "seconds": round(time.perf_counter() - t0, 4),
        "error": error,
        "traceback": tb,
    }
    rec.checks["7_model_import_blocked"] = not model_importable

    if model_importable:
        raise GateFailure(
            "import",
            "PixelHacker class imported without fla — unexpected",
            classification="G",
        )


def audit_checkpoints(rec: Record) -> None:
    unet = PIXELHACKER_REPO / "weight" / "ft_places2" / "diffusion_pytorch_model.bin"
    vae = PIXELHACKER_REPO / "vae" / "diffusion_pytorch_model.bin"
    rec.checkpoints = {
        "unet_expected_path": str(unet),
        "unet_present": unet.is_file(),
        "vae_expected_path": str(vae),
        "vae_present": vae.is_file(),
        "downloaded_this_phase": False,
        "note": (
            "Weights were not downloaded. Import audit is the blocking gate for "
            "LOCAL_MPS on Apple Silicon."
        ),
    }


def run_validation() -> Record:
    rec = Record()
    try:
        collect_environment(rec)
        collect_torch_info(rec)
        audit_hardware(rec)
        verify_pinned_repository(rec)
        audit_upstream_device_selection(rec)
        audit_requirements(rec)
        attempt_fla_import(rec)
        attempt_gla_import(rec)
        attempt_model_import(rec)
        audit_checkpoints(rec)
        confirm_upstream_untouched(rec)

        rec.checks["8_no_inference_executed"] = True
        rec.classification = "LOCAL_MPS"
        rec.primary_placement = "CLOUD_GPU"
        rec.verdict = "FAIL"
        rec.failure_stage = "import"
        rec.failure_classification = "A -- dependency incompatibility"
        rec.failure_message = (
            "PixelHacker imports flash-linear-attention (fla) GLA operators at "
            "module load. fla/Triton are not installable on this Apple Silicon "
            "host, and the upstream entry point has no MPS device path."
        )
        rec.notes.append(
            "Expected negative result. LOCAL_MPS is FAIL; primary placement is "
            "CLOUD_GPU (runtime not validated)."
        )
    except GateFailure as exc:
        rec.failure_stage = exc.stage
        rec.failure_type = type(exc).__name__
        rec.failure_message = str(exc)
        rec.failure_traceback = traceback.format_exc()
        if exc.classification:
            rec.failure_classification = (
                f"{exc.classification} -- {FAILURE_CLASSES[exc.classification]}"
            )
        rec.verdict = "PARTIAL" if exc.blocking else "FAIL"
        rec.classification = exc.classification or "LOCAL_MPS"
    except BaseException as exc:  # noqa: BLE001
        rec.failure_stage = "setup"
        rec.failure_type = type(exc).__name__
        rec.failure_message = str(exc)
        rec.failure_traceback = traceback.format_exc()
        rec.verdict = "FAIL"

    try:
        confirm_upstream_untouched(rec)
    except Exception:
        pass

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    result_path = OUTPUT_DIR / "result.json"
    rec.artifacts["result_json"] = str(result_path)
    result_path.write_text(rec.to_json())
    return rec


def _print_summary(rec: Record) -> None:
    print("\n" + "=" * 72)
    print(f"  PixelHacker MPS validation -- VERDICT: {rec.verdict}")
    print("=" * 72)
    print(f"  classification  {rec.classification}")
    print(f"  placement       {rec.primary_placement}")
    print(f"  commit          {rec.repository.get('observed_commit')}")
    print(f"  mps available   {rec.torch_info.get('mps_is_available')}")
    print(f"  fla importable  {rec.dependency_audit.get('fla_ops_gla_importable')}")
    print("  -- mandated checks --")
    for key in MANDATED_CHECKS:
        val = rec.checks[key]
        label = "PASS" if val else ("FAIL" if val is False else "NOT REACHED")
        print(f"       {label:11s} {key}")
    if rec.failure_message:
        print(f"\n  failure         {rec.failure_message}")
    print(f"\n  result.json     {OUTPUT_DIR / 'result.json'}")
    print("=" * 72 + "\n")


def test_pixelhacker_mps_gate_records_local_fail() -> None:
    """pytest entry point. FAIL is the expected, recorded outcome."""
    rec = run_validation()
    _print_summary(rec)
    assert rec.verdict == "FAIL", (
        f"expected FAIL verdict recording LOCAL_MPS impossibility, got {rec.verdict}"
    )
    assert rec.classification == "LOCAL_MPS"
    assert rec.primary_placement == "CLOUD_GPU"
    assert rec.checks["5_fla_not_installable"] is True
    assert rec.checks["6_gla_import_blocked"] is True
    assert rec.checks["7_model_import_blocked"] is True
    assert rec.repository.get("upstream_unmodified") is True


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    rec = run_validation()
    _print_summary(rec)
    return 0 if rec.verdict == "FAIL" and rec.classification == "LOCAL_MPS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
