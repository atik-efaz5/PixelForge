#!/usr/bin/env bash
# Install launchd agent to start PixelForge backend + tunnel at login (macOS).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="com.pixelforge.public-demo"
DEST="$HOME/Library/LaunchAgents/${LABEL}.plist"
TEMPLATE="$ROOT/scripts/com.pixelforge.public-demo.plist.template"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "ERROR: launchd install is macOS-only." >&2
  exit 1
fi

mkdir -p "$ROOT/logs"
sed "s|__PIXELFORGE_ROOT__|$ROOT|g" "$TEMPLATE" >"$DEST"

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" "$DEST"
launchctl enable "gui/$(id -u)/$LABEL"
launchctl kickstart -k "gui/$(id -u)/$LABEL"

echo "Installed $DEST"
echo "Logs: $ROOT/logs/public-demo.{stdout,stderr}.log"
echo "Unload: launchctl bootout gui/$(id -u)/$LABEL"
