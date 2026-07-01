#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUNTIME_DIR="$ROOT_DIR/data/runtime"
PID_DIR="$RUNTIME_DIR/pids"
LOG_DIR="$RUNTIME_DIR/logs"

MODEL_PATH="${OCCUPANCY_MODEL:-models/occupancy_mlp_train02_relabel/model.npz}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8000}"
OCCUPIED_CONFIRM="${OCCUPIED_CONFIRM:-3}"
LEAVE_CONFIRM="${LEAVE_CONFIRM:-10}"
RECENTLY_USED_MINUTES="${RECENTLY_USED_MINUTES:-15}"
SNAPSHOT_INTERVAL="${SNAPSHOT_INTERVAL:-30}"

MONITOR_PID_FILE="$PID_DIR/monitor.pid"
DASHBOARD_PID_FILE="$PID_DIR/dashboard.pid"

mkdir -p "$PID_DIR" "$LOG_DIR"

is_running() {
  local pid_file="$1"
  [[ -f "$pid_file" ]] && ps -p "$(cat "$pid_file")" >/dev/null 2>&1
}

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Missing occupancy model: $MODEL_PATH"
  echo "Copy model.npz to the Raspberry Pi before starting the system."
  exit 1
fi

if is_running "$MONITOR_PID_FILE"; then
  echo "Monitor already running with PID $(cat "$MONITOR_PID_FILE")"
else
  echo "Starting workstation monitor..."
  sudo -v
  nohup sudo python3 sensor/workstation_monitor.py \
    --occupancy-model "$MODEL_PATH" \
    --occupied-confirm "$OCCUPIED_CONFIRM" \
    --leave-confirm "$LEAVE_CONFIRM" \
    --recently-used-minutes "$RECENTLY_USED_MINUTES" \
    --snapshot-interval "$SNAPSHOT_INTERVAL" \
    --no-window \
    > "$LOG_DIR/monitor.log" 2>&1 &
  echo "$!" > "$MONITOR_PID_FILE"
  echo "Monitor PID: $(cat "$MONITOR_PID_FILE")"
fi

if is_running "$DASHBOARD_PID_FILE"; then
  echo "Dashboard server already running with PID $(cat "$DASHBOARD_PID_FILE")"
else
  echo "Starting dashboard server on port $DASHBOARD_PORT..."
  nohup python3 -m http.server "$DASHBOARD_PORT" --bind 0.0.0.0 \
    > "$LOG_DIR/dashboard.log" 2>&1 &
  echo "$!" > "$DASHBOARD_PID_FILE"
  echo "Dashboard PID: $(cat "$DASHBOARD_PID_FILE")"
fi

echo ""
echo "System started."
echo "Dashboard: http://<raspberry-pi-ip>:$DASHBOARD_PORT/dashboard/"
echo "Tailscale:  http://100.121.0.89:$DASHBOARD_PORT/dashboard/"
echo ""
echo "Logs:"
echo "  $LOG_DIR/monitor.log"
echo "  $LOG_DIR/dashboard.log"
echo ""
echo "Stop with:"
echo "  ./scripts/stop_system.sh"
