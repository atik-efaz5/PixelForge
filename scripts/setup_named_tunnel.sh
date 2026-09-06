#!/usr/bin/env bash
# One-time setup for a stable Cloudflare named tunnel (fixed hostname).
#
# Option A — Zero Trust tunnel token (recommended):
#   1. https://one.dash.cloudflare.com/ → Networks → Tunnels → Create tunnel
#   2. Public hostname → http://127.0.0.1:8000
#   3. Copy the install token into configs/tunnel.env:
#        CLOUDFLARE_TUNNEL_TOKEN=...
#        PIXELFORGE_PUBLIC_API_URL=https://api.yourdomain.com
#   4. Set Vercel NEXT_PUBLIC_API_BASE_URL to PIXELFORGE_PUBLIC_API_URL
#
# Option B — Local tunnel + DNS (requires a domain on Cloudflare):
#   1. cloudflared tunnel login
#   2. cloudflared tunnel create pixelforge
#   3. cloudflared tunnel route dns pixelforge api.yourdomain.com
#   4. Copy configs/cloudflared/config.yml.example → configs/cloudflared/config.yml
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${PIXELFORGE_TUNNEL_ENV:-$ROOT/configs/tunnel.env}"

if ! command -v cloudflared >/dev/null 2>&1; then
  echo "ERROR: cloudflared not found. Install: brew install cloudflared" >&2
  exit 1
fi

echo "PixelForge named tunnel setup"
echo "  Env file: $ENV_FILE"
echo ""

if [[ -f "$ENV_FILE" ]] && grep -q 'CLOUDFLARE_TUNNEL_TOKEN=' "$ENV_FILE" 2>/dev/null; then
  echo "CLOUDFLARE_TUNNEL_TOKEN already present in $ENV_FILE"
  echo "Run: ./scripts/start_public_demo.sh"
  exit 0
fi

if [[ ! -f "$HOME/.cloudflared/cert.pem" ]]; then
  echo "Step 1: Log in to Cloudflare (browser will open)..."
  cloudflared tunnel login
fi

TUNNEL_NAME="${PIXELFORGE_TUNNEL_NAME:-pixelforge}"
if ! cloudflared tunnel list 2>/dev/null | grep -q "$TUNNEL_NAME"; then
  echo "Step 2: Creating tunnel '$TUNNEL_NAME'..."
  cloudflared tunnel create "$TUNNEL_NAME"
fi

TUNNEL_ID="$(cloudflared tunnel list 2>/dev/null | awk -v n="$TUNNEL_NAME" '$0 ~ n {print $1; exit}')"
if [[ -z "$TUNNEL_ID" ]]; then
  echo "ERROR: Could not resolve tunnel id for $TUNNEL_NAME" >&2
  exit 1
fi

HOSTNAME="${PIXELFORGE_TUNNEL_HOSTNAME:-}"
if [[ -z "$HOSTNAME" ]]; then
  read -r -p "Public hostname (e.g. api.example.com): " HOSTNAME
fi

echo "Step 3: Routing DNS $HOSTNAME → tunnel $TUNNEL_NAME ($TUNNEL_ID)..."
cloudflared tunnel route dns "$TUNNEL_NAME" "$HOSTNAME"

mkdir -p "$ROOT/configs/cloudflared"
CFG="$ROOT/configs/cloudflared/config.yml"
cat >"$CFG" <<EOF
tunnel: $TUNNEL_NAME
credentials-file: $HOME/.cloudflared/${TUNNEL_ID}.json

ingress:
  - hostname: $HOSTNAME
    service: http://127.0.0.1:8000
  - service: http_status:404
EOF

mkdir -p "$(dirname "$ENV_FILE")"
touch "$ENV_FILE"
if ! grep -q 'PIXELFORGE_PUBLIC_API_URL=' "$ENV_FILE" 2>/dev/null; then
  echo "PIXELFORGE_PUBLIC_API_URL=https://${HOSTNAME}" >>"$ENV_FILE"
fi
if ! grep -q 'PIXELFORGE_CLOUDFLARED_CONFIG=' "$ENV_FILE" 2>/dev/null; then
  echo "PIXELFORGE_CLOUDFLARED_CONFIG=$CFG" >>"$ENV_FILE"
fi

echo ""
echo "Done. Next:"
echo "  1. Set Vercel production env:"
echo "       NEXT_PUBLIC_API_BASE_URL=https://${HOSTNAME}"
echo "  2. Redeploy frontend (cd apps/frontend && npx vercel deploy --prod)"
echo "  3. Start stack: ./scripts/start_public_demo.sh"
