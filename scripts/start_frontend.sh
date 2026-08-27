#!/usr/bin/env bash
# Start the PixelForge Next.js frontend dev server.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRONTEND="$ROOT/apps/frontend"
cd "$FRONTEND"

# Optional: load deployment env file if present
if [[ -f "$ROOT/configs/deployment.env" ]]; then
  # shellcheck disable=SC1091
  set -a && source "$ROOT/configs/deployment.env" && set +a
fi

if ! command -v npm >/dev/null 2>&1; then
  echo "ERROR: npm not found. Install Node.js >= 18." >&2
  echo "  See docs/DEPLOYMENT.md" >&2
  exit 1
fi

if [[ ! -d "$FRONTEND/node_modules" ]]; then
  echo "ERROR: node_modules/ missing." >&2
  echo "  Run: cd apps/frontend && npm install" >&2
  exit 1
fi

PORT="${PIXELFORGE_FRONTEND_PORT:-3000}"
API_PORT="${PIXELFORGE_API_PORT:-8000}"
API_HOST="${PIXELFORGE_API_HOST:-127.0.0.1}"
export NEXT_PUBLIC_API_BASE_URL="${NEXT_PUBLIC_API_BASE_URL:-http://${API_HOST}:${API_PORT}}"

echo "Starting PixelForge frontend on http://127.0.0.1:${PORT}"
echo "  API: $NEXT_PUBLIC_API_BASE_URL"
echo "Press Ctrl+C to stop."

exec npm run dev -- --port "$PORT"
