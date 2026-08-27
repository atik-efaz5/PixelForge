#!/usr/bin/env bash
# Stop local PixelForge backend, frontend, and orphan model workers.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -f "$ROOT/configs/deployment.env" ]]; then
  # shellcheck disable=SC1091
  set -a && source "$ROOT/configs/deployment.env" && set +a
fi

PORT_BACKEND="${PIXELFORGE_API_PORT:-8000}"
PORT_FRONTEND="${PIXELFORGE_FRONTEND_PORT:-3000}"

stop_port() {
  local port="$1"
  local label="$2"
  local pids
  pids="$(lsof -ti ":$port" 2>/dev/null || true)"
  if [[ -n "$pids" ]]; then
    echo "Stopping $label on port $port (PIDs: $pids)"
    # shellcheck disable=SC2086
    kill $pids 2>/dev/null || true
    sleep 1
    # shellcheck disable=SC2086
    kill -9 $pids 2>/dev/null || true
  else
    echo "No process on port $port ($label)"
  fi
}

stop_port "$PORT_BACKEND" "backend"
stop_port "$PORT_FRONTEND" "frontend"

WORKERS="$(pgrep -f "moebius_persistent_worker|isolated_inpaint_worker|isolated_grounding_worker" 2>/dev/null || true)"
if [[ -n "$WORKERS" ]]; then
  echo "Stopping worker processes: $WORKERS"
  # shellcheck disable=SC2086
  kill $WORKERS 2>/dev/null || true
  sleep 1
  # shellcheck disable=SC2086
  kill -9 $WORKERS 2>/dev/null || true
else
  echo "No orphan worker processes"
fi

REMAINING="$(pgrep -f "uvicorn apps.backend.main|moebius_persistent_worker|next dev" 2>/dev/null || true)"
if [[ -n "$REMAINING" ]]; then
  echo "WARNING: processes still running: $REMAINING" >&2
  exit 1
fi

echo "Local PixelForge stopped."
