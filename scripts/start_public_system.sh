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
MONITOR_PID_FILE="$PID_DIR/monitor.pid"
DASHBOARD_PID_FILE="$PID_DIR/dashboard.pid"

is_running() {
  local pid_file="$1"
  local expected="$2"
  [[ -f "$pid_file" ]] || return 1
  local pid command
  pid="$(cat "$pid_file")"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  command="$(ps -p "$pid" -o command= 2>/dev/null || true)"
  [[ -n "$command" && "$command" == *"$expected"* ]]
}

rollback_new_process() {
  local name="$1"
  local pid_file="$2"
  local expected="$3"
  local use_sudo="$4"
  local was_running="$5"

  if [[ "$was_running" == "true" ]] || [[ ! -f "$pid_file" ]]; then
    return
  fi
  if ! is_running "$pid_file" "$expected"; then
    rm -f "$pid_file"
    return
  fi

  local pid
  pid="$(cat "$pid_file")"
  echo "Rolling back newly started $name process $pid..."
  if [[ "$use_sudo" == "true" ]]; then
    sudo kill "$pid" 2>/dev/null || true
  else
    kill "$pid" 2>/dev/null || true
  fi
  sleep 0.5
  if ps -p "$pid" >/dev/null 2>&1; then
    echo "Warning: $name process $pid is still running; PID file retained."
  else
    rm -f "$pid_file"
  fi
}

rollback_local_start() {
  rollback_new_process "dashboard" "$DASHBOARD_PID_FILE" "dashboard_server.py" false "$dashboard_was_running"
  rollback_new_process "monitor" "$MONITOR_PID_FILE" "workstation_monitor.py" true "$monitor_was_running"
}

if ! command -v tailscale >/dev/null 2>&1; then
  echo "tailscale command not found. Install/sign in to Tailscale before using public mode."
  exit 1
fi

monitor_was_running=false
dashboard_was_running=false
if is_running "$MONITOR_PID_FILE" "workstation_monitor.py"; then
  monitor_was_running=true
fi
if is_running "$DASHBOARD_PID_FILE" "dashboard_server.py"; then
  dashboard_was_running=true
fi

echo "Starting local monitor and dashboard..."
"$ROOT_DIR/scripts/start_system.sh"
mkdir -p "$PID_DIR" "$LOG_DIR"

echo ""
echo "Checking local dashboard on port $DASHBOARD_PORT..."
for _ in $(seq 1 20); do
  if curl -fsS "http://127.0.0.1:$DASHBOARD_PORT/healthz" >/dev/null 2>&1; then
    echo "Local dashboard is ready."
    break
  fi
  sleep 0.5
done

if ! curl -fsS "http://127.0.0.1:$DASHBOARD_PORT/healthz" >/dev/null 2>&1; then
  echo "Dashboard did not become ready on http://127.0.0.1:$DASHBOARD_PORT/healthz"
  echo "Check logs:"
  echo "  $LOG_DIR/dashboard.log"
  rollback_local_start
  exit 1
fi

if [[ -f "$FUNNEL_PID_FILE" ]] && ! is_running "$FUNNEL_PID_FILE" "tailscale funnel"; then
  echo "Removing stale Tailscale Funnel PID file."
  rm -f "$FUNNEL_PID_FILE"
fi

if is_running "$FUNNEL_PID_FILE" "tailscale funnel"; then
  echo "Tailscale Funnel already running with PID $(cat "$FUNNEL_PID_FILE")"
else
  echo "Starting Tailscale Funnel on port $DASHBOARD_PORT..."
  if ! sudo -v; then
    echo "Unable to obtain sudo permission for Tailscale Funnel."
    rollback_local_start
    exit 1
  fi
  nohup sudo tailscale funnel "$DASHBOARD_PORT" \
    > "$LOG_DIR/tailscale_funnel.log" 2>&1 &
  echo "$!" > "$FUNNEL_PID_FILE"
  sleep 2

  if ! is_running "$FUNNEL_PID_FILE" "tailscale funnel"; then
    echo "Tailscale Funnel did not stay running."
    echo "Check log:"
    echo "  $LOG_DIR/tailscale_funnel.log"
    rm -f "$FUNNEL_PID_FILE"
    rollback_local_start
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
echo "  Project page: http://127.0.0.1:$DASHBOARD_PORT/dashboard/"
echo "  Live monitor: http://127.0.0.1:$DASHBOARD_PORT/dashboard/live/"
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
