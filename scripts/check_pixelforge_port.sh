#!/usr/bin/env bash
# Verify port 8000 (or PIXELFORGE_API_PORT) is free or already serves PixelForge /health.
set -euo pipefail

HOST="${PIXELFORGE_API_HOST:-127.0.0.1}"
PORT="${PIXELFORGE_API_PORT:-8000}"
URL="http://${HOST}:${PORT}/health"

if ! lsof -tiTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  exit 0
fi

HEALTH="$(curl -fsS --max-time 3 "$URL" 2>/dev/null || true)"
if [[ -z "$HEALTH" ]]; then
  echo "ERROR: Port $PORT is in use but /health did not respond." >&2
  echo "  Another app may be bound (e.g. Floodlens). Run it on another port (8010)." >&2
  lsof -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null | head -5 >&2 || true
  exit 1
fi

if ! printf '%s' "$HEALTH" | python3 -c '
import json, sys
data = json.load(sys.stdin)
required = ("status", "version", "max_upload_bytes", "max_concurrent_generations")
missing = [k for k in required if k not in data]
if missing:
    raise SystemExit(1)
if data.get("status") != "ok" or data.get("version") != "0.1.0":
    raise SystemExit(1)
' 2>/dev/null; then
  echo "ERROR: Port $PORT is serving a non-PixelForge app." >&2
  echo "  /health response: $HEALTH" >&2
  echo "  Stop the other process or run it on a different port (e.g. Floodlens on 8010)." >&2
  lsof -iTCP:"$PORT" -sTCP:LISTEN 2>/dev/null | head -5 >&2 || true
  exit 1
fi

echo "PixelForge already listening on $URL"
exit 2
