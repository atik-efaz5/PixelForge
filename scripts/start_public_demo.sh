#!/usr/bin/env bash
# Start backend + Cloudflare tunnel for the Vercel-hosted frontend.
#
# Stable API (named tunnel): copy configs/tunnel.env.example → configs/tunnel.env
#   and set CLOUDFLARE_TUNNEL_TOKEN or PIXELFORGE_CLOUDFLARED_CONFIG.
# Quick tunnel (ephemeral URL): used when no named config/token is present.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ENV_FILE="${PIXELFORGE_TUNNEL_ENV:-$ROOT/configs/tunnel.env}"
if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  set -a && source "$ENV_FILE" && set +a
fi

VERCEL_ORIGIN="${PIXELFORGE_VERCEL_ORIGIN:-https://frontend-mu-two-wzuqjziue7.vercel.app}"
export PIXELFORGE_CORS_ORIGINS="${PIXELFORGE_CORS_ORIGINS:-${VERCEL_ORIGIN},http://localhost:3000,http://127.0.0.1:3000}"

if ! command -v cloudflared >/dev/null 2>&1; then
  echo "ERROR: cloudflared not found. Install: brew install cloudflared" >&2
  exit 1
fi

BACKEND_PID=""
TUNNEL_PID=""

cleanup() {
  [[ -n "$TUNNEL_PID" ]] && kill "$TUNNEL_PID" 2>/dev/null || true
  [[ -n "$BACKEND_PID" ]] && kill "$BACKEND_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

PORT_STATUS=0
./scripts/check_pixelforge_port.sh || PORT_STATUS=$?

if [[ "$PORT_STATUS" -eq 1 ]]; then
  exit 1
elif [[ "$PORT_STATUS" -eq 0 ]]; then
  echo "Starting backend (CORS includes ${VERCEL_ORIGIN})..."
  ./scripts/start_backend.sh &
  BACKEND_PID=$!
  sleep 3
else
  echo "Backend already running; skipping start."
fi

start_named_tunnel() {
  if [[ -n "${CLOUDFLARE_TUNNEL_TOKEN:-}" ]]; then
    echo "Starting named Cloudflare tunnel (token)..."
    exec cloudflared tunnel run --token "$CLOUDFLARE_TUNNEL_TOKEN"
  fi

  local cfg="${PIXELFORGE_CLOUDFLARED_CONFIG:-$HOME/.cloudflared/config.yml}"
  if [[ -f "$cfg" ]]; then
    echo "Starting named Cloudflare tunnel ($cfg)..."
    exec cloudflared tunnel --config "$cfg" run
  fi

  return 1
}

if start_named_tunnel; then
  exit 0
fi

echo "WARNING: No named tunnel configured. Using ephemeral trycloudflare.com URL."
echo "  For a stable hostname run: ./scripts/setup_named_tunnel.sh"
echo "Starting Cloudflare quick tunnel to http://127.0.0.1:8000 ..."
exec cloudflared tunnel --url http://127.0.0.1:8000
