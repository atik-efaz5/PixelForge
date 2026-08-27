#!/usr/bin/env bash
# Start the PixelForge FastAPI backend (SAM 2 in-process; Moebius via worker).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Optional: load deployment env file if present (not required)
if [[ -f "$ROOT/configs/deployment.env" ]]; then
  # shellcheck disable=SC1091
  set -a && source "$ROOT/configs/deployment.env" && set +a
fi

PYTHON="${PIXELFORGE_BACKEND_PYTHON:-}"
if [[ -z "$PYTHON" ]]; then
  if [[ -n "${CONDA_PREFIX:-}" && "$(basename "$CONDA_PREFIX")" == "pixelforge-sam2-v2" ]]; then
    PYTHON="$CONDA_PREFIX/bin/python"
  elif [[ -x "/opt/anaconda3/envs/pixelforge-sam2-v2/bin/python" ]]; then
    PYTHON="/opt/anaconda3/envs/pixelforge-sam2-v2/bin/python"
  fi
fi

if [[ -z "$PYTHON" || ! -x "$PYTHON" ]]; then
  echo "ERROR: Backend Python not found (pixelforge-sam2-v2)." >&2
  echo "  Set PIXELFORGE_BACKEND_PYTHON or activate: conda activate pixelforge-sam2-v2" >&2
  echo "  See docs/DEPLOYMENT.md" >&2
  exit 1
fi

if [[ ! -d "$ROOT/.e2e_deps" ]]; then
  echo "ERROR: .e2e_deps/ not found at $ROOT/.e2e_deps" >&2
  echo "  Install vendored backend deps — see docs/DEPLOYMENT.md §3." >&2
  exit 1
fi

HOST="${PIXELFORGE_API_HOST:-127.0.0.1}"
PORT="${PIXELFORGE_API_PORT:-8000}"

export PYTHONPATH="$ROOT/.e2e_deps:$ROOT"
export PIXELFORGE_ROOT="${PIXELFORGE_ROOT:-$ROOT}"

echo "Starting PixelForge backend on http://${HOST}:${PORT}"
echo "  Python: $PYTHON"
echo "  Root:   $ROOT"
echo "Press Ctrl+C to stop (persistent workers shut down on exit)."

exec "$PYTHON" -m uvicorn apps.backend.main:app --host "$HOST" --port "$PORT"
