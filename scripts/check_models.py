#!/usr/bin/env python3
"""Report local checkpoint presence without loading model weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.config import model_entry, project_root, resolve_path  # noqa: E402

# Local re-acquisition anchors from validation docs (optional verification).
KNOWN_SHA256: dict[str, str] = {
    "sam2": "7402e0d864fa82708a20fbd15bc84245c2f26dff0eb43a4b5b93452deb34be69",
    "moebius_student": "6525afb888e55f9b5c74fa0a5d19ca0762d720d6c716fb0f8422fbeb6868a09a",
    "grounding_dino": "3b3ca2563c77c69f651d7bd133e97139c186df06231157a64c507099c52bc799",
}

CHECKPOINT_ENV: dict[str, str] = {
    "sam2": "PIXELFORGE_SAM2_CHECKPOINT",
    "moebius_student": "PIXELFORGE_MOEBIUS_CHECKPOINT",
    "moebius_vae": "PIXELFORGE_MOEBIUS_VAE",
    "grounding_dino": "PIXELFORGE_GROUNDING_DINO_CHECKPOINT",
}


def _human_size(num_bytes: int) -> str:
    if num_bytes < 1024:
        return f"{num_bytes} B"
    if num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KiB"
    return f"{num_bytes / (1024 * 1024):.2f} MiB"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_file(
    key: str,
    *,
    model_name: str,
    config_key: str = "checkpoint",
    compute_sha: bool,
) -> dict[str, object]:
    entry = model_entry(model_name)
    env_var = CHECKPOINT_ENV.get(key, "")
    raw = os.environ.get(env_var, "").strip() if env_var else ""
    if not raw:
        raw = str(entry.get(config_key) or "")
    path = resolve_path(raw)
    result: dict[str, object] = {
        "key": key,
        "model": entry.get("display_name") or model_name,
        "path": str(path) if path else None,
        "env_override": env_var or None,
        "present": False,
    }
    if path is None:
        result["status"] = "missing_config"
        return result
    if not path.is_file():
        result["status"] = "missing"
        return result
    size = path.stat().st_size
    result["present"] = True
    result["size_bytes"] = size
    result["size_human"] = _human_size(size)
    result["status"] = "ok"
    expected = KNOWN_SHA256.get(key)
    if compute_sha:
        digest = _sha256(path)
        result["sha256"] = digest
        if expected:
            result["expected_sha256"] = expected
            result["sha256_match"] = digest == expected
    elif expected:
        result["expected_sha256"] = expected
    return result


def _check_vae_dir(*, compute_sha: bool) -> dict[str, object]:
    entry = model_entry("moebius")
    raw = os.environ.get("PIXELFORGE_MOEBIUS_VAE", "").strip() or str(entry.get("vae_dir") or "")
    path = resolve_path(raw)
    result: dict[str, object] = {
        "key": "moebius_vae",
        "model": "Moebius VAE",
        "path": str(path) if path else None,
        "env_override": "PIXELFORGE_MOEBIUS_VAE",
        "present": False,
    }
    if path is None or not path.is_dir():
        result["status"] = "missing"
        return result
    files = sorted(p for p in path.rglob("*") if p.is_file())
    if not files:
        result["status"] = "empty"
        return result
    total = sum(p.stat().st_size for p in files)
    result["present"] = True
    result["file_count"] = len(files)
    result["size_bytes"] = total
    result["size_human"] = _human_size(total)
    result["status"] = "ok"
    if compute_sha:
        result["files"] = [
            {"name": p.name, "sha256": _sha256(p), "size_bytes": p.stat().st_size}
            for p in files[:20]
        ]
        if len(files) > 20:
            result["note"] = f"SHA shown for first 20 of {len(files)} files"
    return result


def run(*, compute_sha: bool) -> dict[str, object]:
    checks = [
        _check_file("sam2", model_name="sam2", compute_sha=compute_sha),
        _check_file("moebius_student", model_name="moebius", compute_sha=compute_sha),
        _check_vae_dir(compute_sha=compute_sha),
        _check_file("grounding_dino", model_name="grounding_dino", compute_sha=compute_sha),
    ]
    required = {"sam2", "moebius_student", "moebius_vae"}
    required_ok = all(c["key"] in required and c.get("present") for c in checks if c["key"] in required)
    return {
        "project_root": str(project_root()),
        "required_present": required_ok,
        "checkpoints": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check PixelForge checkpoint files.")
    parser.add_argument(
        "--sha",
        action="store_true",
        help="Compute SHA-256 (slow for large files; does not load model weights).",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON only.")
    args = parser.parse_args()
    report = run(compute_sha=args.sha)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Project root: {report['project_root']}")
        print()
        for item in report["checkpoints"]:
            key = item["key"]
            status = item.get("status", "?")
            path = item.get("path", "?")
            if item.get("present"):
                line = f"[OK]   {key}: {path} ({item.get('size_human')})"
                if item.get("sha256_match") is True:
                    line += " sha256=match"
                elif item.get("sha256_match") is False:
                    line += " sha256=MISMATCH"
                print(line)
            else:
                optional = key == "grounding_dino"
                tag = "OPT" if optional else "MISS"
                print(f"[{tag}] {key}: {path} ({status})")
        print()
        print(f"Required checkpoints ready: {report['required_present']}")

    return 0 if report["required_present"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
