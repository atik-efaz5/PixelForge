#!/usr/bin/env python3
"""Lightweight environment diagnostic — does not install anything."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

ENV_SPECS: tuple[tuple[str, str, str], ...] = (
    ("backend", "pixelforge-sam2-v2", "PIXELFORGE_BACKEND_PYTHON"),
    ("moebius", "pixelforge-moebius", "PIXELFORGE_MOEBIUS_PYTHON"),
    ("grounding_dino", "pixelforge-grounding-dino", "PIXELFORGE_GROUNDING_DINO_PYTHON"),
)

PACKAGE_PROBES: tuple[str, ...] = ("torch", "fastapi", "uvicorn", "numpy", "PIL")


def _run_version(cmd: list[str]) -> str | None:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return (out.stdout or out.stderr).strip().splitlines()[0] if (out.stdout or out.stderr).strip() else None


def _resolve_python(env_name: str, override_var: str) -> Path | None:
    override = os.environ.get(override_var, "").strip()
    if override:
        path = Path(override).expanduser()
        return path if path.is_file() else None
    if os.environ.get("CONDA_PREFIX") and Path(os.environ["CONDA_PREFIX"]).name == env_name:
        candidate = Path(os.environ["CONDA_PREFIX"]) / "bin" / "python"
        if candidate.is_file():
            return candidate
    default = Path(f"/opt/anaconda3/envs/{env_name}/bin/python")
    return default if default.is_file() else None


def _probe_python(path: Path) -> dict[str, object]:
    info: dict[str, object] = {"path": str(path), "exists": path.is_file()}
    if not path.is_file():
        info["status"] = "missing"
        return info
    version = _run_version([str(path), "--version"])
    info["version"] = version
    packages: dict[str, str | None] = {}
    for pkg in PACKAGE_PROBES:
        mod = "PIL" if pkg == "PIL" else pkg
        ver = _run_version([str(path), "-c", f"import {mod}; print(getattr({mod}, '__version__', 'ok'))"])
        packages[pkg] = ver
    info["packages"] = packages
    mps = _run_version([
        str(path),
        "-c",
        "import torch; print('built=', torch.backends.mps.is_built(), "
        "'available=', torch.backends.mps.is_available())",
    ])
    info["mps"] = mps
    info["status"] = "ok"
    return info


def run() -> dict[str, object]:
    node = shutil.which("npm")
    npm_version = _run_version(["npm", "--version"]) if node else None
    node_version = _run_version(["node", "--version"]) if shutil.which("node") else None

    envs: dict[str, object] = {}
    for label, env_name, override_var in ENV_SPECS:
        py = _resolve_python(env_name, override_var)
        envs[label] = {
            "conda_env": env_name,
            "override_var": override_var,
            "python": _probe_python(py) if py else {"path": None, "status": "missing"},
        }

    return {
        "project_root": str(ROOT),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "node": node_version,
        "npm": npm_version,
        "e2e_deps_present": (ROOT / ".e2e_deps").is_dir(),
        "frontend_node_modules": (ROOT / "apps" / "frontend" / "node_modules").is_dir(),
        "environments": envs,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check PixelForge deployment environment.")
    parser.add_argument("--json", action="store_true", help="Emit JSON only.")
    args = parser.parse_args()
    report = run()

    backend_ok = (
        report["e2e_deps_present"]
        and report["environments"]["backend"]["python"].get("status") == "ok"  # type: ignore[index]
    )

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        plat = report["platform"]
        print(f"Project root: {report['project_root']}")
        print(f"OS: {plat['system']} {plat['release']} ({plat['machine']})")
        print(f"Node: {report['node'] or 'NOT FOUND'}")
        print(f"npm:  {report['npm'] or 'NOT FOUND'}")
        print(f".e2e_deps/: {'present' if report['e2e_deps_present'] else 'MISSING'}")
        print(f"frontend node_modules/: {'present' if report['frontend_node_modules'] else 'MISSING'}")
        print()
        for label, data in report["environments"].items():  # type: ignore[union-attr]
            py = data["python"]
            status = py.get("status", "?")
            path = py.get("path", "?")
            ver = py.get("version", "")
            tag = "OK" if status == "ok" else "MISS"
            print(f"[{tag}] {label} ({data['conda_env']}): {path} {ver}")
            if status == "ok" and py.get("mps"):
                print(f"       MPS: {py['mps']}")
        print()
        print(f"Minimum backend ready: {backend_ok}")

    return 0 if backend_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
