#!/usr/bin/env bash
# Production Vercel deploy with NEXT_PUBLIC_API_BASE_URL baked in.
# Reads API URL from configs/tunnel.env or PIXELFORGE_PUBLIC_API_URL.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${PIXELFORGE_TUNNEL_ENV:-$ROOT/configs/tunnel.env}"

if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  set -a && source "$ENV_FILE" && set +a
fi

API_URL="${PIXELFORGE_PUBLIC_API_URL:-${NEXT_PUBLIC_API_BASE_URL:-}}"
if [[ -z "$API_URL" || "$API_URL" == *"YOUR_TUNNEL"* || "$API_URL" == *"yourdomain"* ]]; then
  echo "ERROR: Set PIXELFORGE_PUBLIC_API_URL or NEXT_PUBLIC_API_BASE_URL in $ENV_FILE" >&2
  exit 1
fi

echo "Deploying frontend with NEXT_PUBLIC_API_BASE_URL=$API_URL"
cd "$ROOT/apps/frontend"
export NEXT_PUBLIC_API_BASE_URL="$API_URL"
exec npx vercel@latest deploy --prod --yes
