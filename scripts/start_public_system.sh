#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUNTIME_DIR="$ROOT_DIR/data/runtime"
PID_DIR="$RUNTIME_DIR/pids"
LOG_DIR="$RUNTIME_DIR/logs"

DASHBOARD_PORT="${DASHBOARD_PORT:-8000}"
PUBLIC_DASHBOARD_URL="${PUBLIC_DASHBOARD_URL:-}"
FUNNEL_PID_FILE="$PID_DIR/tailscale_funnel.pid"

mkdir -p "$PID_DIR" "$LOG_DIR"

is_running() {
  local pid_file="$1"
  [[ -f "$pid_file" ]] && ps -p "$(cat "$pid_file")" >/dev/null 2>&1
}

echo "Starting local monitor and dashboard..."
"$ROOT_DIR/scripts/start_system.sh"

echo ""
echo "Checking local dashboard on port $DASHBOARD_PORT..."
for _ in $(seq 1 20); do
  if curl -fsS "http://127.0.0.1:$DASHBOARD_PORT/dashboard/" >/dev/null 2>&1; then
    echo "Local dashboard is ready."
    break
  fi
  sleep 0.5
done

if ! curl -fsS "http://127.0.0.1:$DASHBOARD_PORT/dashboard/" >/dev/null 2>&1; then
  echo "Dashboard did not become ready on http://127.0.0.1:$DASHBOARD_PORT/dashboard/"
  echo "Check logs:"
  echo "  $LOG_DIR/dashboard.log"
  exit 1
fi

if ! command -v tailscale >/dev/null 2>&1; then
  echo "tailscale command not found. Install/sign in to Tailscale before using public mode."
  exit 1
fi

if is_running "$FUNNEL_PID_FILE"; then
  echo "Tailscale Funnel already running with PID $(cat "$FUNNEL_PID_FILE")"
else
  echo "Starting Tailscale Funnel on port $DASHBOARD_PORT..."
  sudo -v
  nohup sudo tailscale funnel "$DASHBOARD_PORT" \
    > "$LOG_DIR/tailscale_funnel.log" 2>&1 &
  echo "$!" > "$FUNNEL_PID_FILE"
  sleep 2

  if ! is_running "$FUNNEL_PID_FILE"; then
    echo "Tailscale Funnel did not stay running."
    echo "Check log:"
    echo "  $LOG_DIR/tailscale_funnel.log"
    exit 1
  fi

  echo "Tailscale Funnel PID: $(cat "$FUNNEL_PID_FILE")"
fi

echo ""
echo "Public dashboard:"
if [[ -n "$PUBLIC_DASHBOARD_URL" ]]; then
  echo "  $PUBLIC_DASHBOARD_URL"
else
  echo "  Check the Funnel log below for the public https://...ts.net URL."
  echo "  Add /dashboard/ to the root Funnel URL if needed."
fi
echo ""
echo "Local dashboard:"
echo "  http://127.0.0.1:$DASHBOARD_PORT/dashboard/"
echo ""
echo "Logs:"
echo "  $LOG_DIR/monitor.log"
echo "  $LOG_DIR/dashboard.log"
echo "  $LOG_DIR/tailscale_funnel.log"
echo ""
echo "Stop everything with:"
echo "  ./scripts/stop_public_system.sh"

if [[ -f "$LOG_DIR/tailscale_funnel.log" ]]; then
  echo ""
  echo "Funnel log tail:"
  tail -20 "$LOG_DIR/tailscale_funnel.log"
fi
