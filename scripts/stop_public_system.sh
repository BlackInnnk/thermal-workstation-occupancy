#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PID_DIR="$ROOT_DIR/data/runtime/pids"
FUNNEL_PID_FILE="$PID_DIR/tailscale_funnel.pid"
stop_failed=false

if [[ -f "$FUNNEL_PID_FILE" ]]; then
  FUNNEL_PID="$(cat "$FUNNEL_PID_FILE")"
  FUNNEL_COMMAND="$(ps -p "$FUNNEL_PID" -o command= 2>/dev/null || true)"
  if [[ "$FUNNEL_PID" =~ ^[0-9]+$ ]] && [[ "$FUNNEL_COMMAND" == *"tailscale funnel"* ]]; then
    echo "Stopping Tailscale Funnel with PID $FUNNEL_PID..."
    if ! sudo kill "$FUNNEL_PID" 2>/dev/null; then
      echo "Initial Funnel stop signal failed; checking whether it already exited."
    fi
    sleep 1
    if ps -p "$FUNNEL_PID" >/dev/null 2>&1; then
      if ! sudo kill -9 "$FUNNEL_PID" 2>/dev/null; then
        echo "Unable to send a forced stop signal to Tailscale Funnel."
      fi
      sleep 0.5
    fi
    if ps -p "$FUNNEL_PID" >/dev/null 2>&1; then
      echo "ERROR: Tailscale Funnel is still running; PID file retained."
      stop_failed=true
    else
      rm -f "$FUNNEL_PID_FILE"
    fi
  elif [[ -z "$FUNNEL_COMMAND" ]]; then
    echo "Tailscale Funnel PID file existed, but process $FUNNEL_PID is not running."
  else
    echo "Tailscale Funnel PID file points to an unrelated process; not stopping it."
  fi
  if [[ "$stop_failed" != "true" ]]; then
    rm -f "$FUNNEL_PID_FILE"
  fi
else
  echo "Tailscale Funnel is not running from this script."
fi

if ! "$ROOT_DIR/scripts/stop_system.sh"; then
  stop_failed=true
fi

if [[ "$stop_failed" == "true" ]]; then
  echo "Public system stop was incomplete. Review the messages above."
  exit 1
fi
