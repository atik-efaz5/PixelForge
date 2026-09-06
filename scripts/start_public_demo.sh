#!/usr/bin/env bash
# Start backend + Cloudflare quick tunnel for the Vercel-hosted frontend.
# Requires: cloudflared (brew install cloudflared), backend deps, checkpoints.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VERCEL_ORIGIN="${PIXELFORGE_VERCEL_ORIGIN:-https://frontend-mu-two-wzuqjziue7.vercel.app}"
export PIXELFORGE_CORS_ORIGINS="${PIXELFORGE_CORS_ORIGINS:-${VERCEL_ORIGIN},http://localhost:3000,http://127.0.0.1:3000}"

if ! command -v cloudflared >/dev/null 2>&1; then
  echo "ERROR: cloudflared not found. Install: brew install cloudflared" >&2
  exit 1
fi

echo "Starting backend (CORS includes ${VERCEL_ORIGIN})..."
./scripts/start_backend.sh &
BACKEND_PID=$!

cleanup() {
  kill "$BACKEND_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

sleep 3
echo "Starting Cloudflare quick tunnel to http://127.0.0.1:8000 ..."
echo "Set Vercel NEXT_PUBLIC_API_BASE_URL to the https://*.trycloudflare.com URL shown below."
exec cloudflared tunnel --url http://127.0.0.1:8000
