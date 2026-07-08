#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PID_DIR="$ROOT_DIR/data/runtime/pids"
FUNNEL_PID_FILE="$PID_DIR/tailscale_funnel.pid"

if [[ -f "$FUNNEL_PID_FILE" ]]; then
  FUNNEL_PID="$(cat "$FUNNEL_PID_FILE")"
  if ps -p "$FUNNEL_PID" >/dev/null 2>&1; then
    echo "Stopping Tailscale Funnel with PID $FUNNEL_PID..."
    sudo kill "$FUNNEL_PID" 2>/dev/null || true
    sleep 1
    if ps -p "$FUNNEL_PID" >/dev/null 2>&1; then
      sudo kill -9 "$FUNNEL_PID" 2>/dev/null || true
    fi
  else
    echo "Tailscale Funnel PID file existed, but process $FUNNEL_PID is not running."
  fi
  rm -f "$FUNNEL_PID_FILE"
else
  echo "Tailscale Funnel is not running from this script."
fi

"$ROOT_DIR/scripts/stop_system.sh"
