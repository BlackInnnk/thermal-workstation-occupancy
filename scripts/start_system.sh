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
STATUS_INTERVAL="${STATUS_INTERVAL:-1}"
MONITOR_LOG_INTERVAL="${MONITOR_LOG_INTERVAL:-10}"
TOOL_SAFE="${TOOL_SAFE:-38}"
TOOL_ALERT="${TOOL_ALERT:-45}"
COOLING_SLOPE="${COOLING_SLOPE:--0.5}"
COOLING_MIN_DROP="${COOLING_MIN_DROP:-2}"
TREND_MIN_SECONDS="${TREND_MIN_SECONDS:-45}"
UNATTENDED_DELAY_SECONDS="${UNATTENDED_DELAY_SECONDS:-180}"
SAFE_CONFIRM_SECONDS="${SAFE_CONFIRM_SECONDS:-60}"

MONITOR_PID_FILE="$PID_DIR/monitor.pid"
DASHBOARD_PID_FILE="$PID_DIR/dashboard.pid"

RUN_USER="${SUDO_USER:-$(id -un)}"
RUN_GROUP="$(id -gn "$RUN_USER")"

runtime_needs_repair=false
if ! mkdir -p "$PID_DIR" "$LOG_DIR" 2>/dev/null; then
  runtime_needs_repair=true
fi

for path in \
  "$RUNTIME_DIR" \
  "$PID_DIR" \
  "$LOG_DIR" \
  "$LOG_DIR/monitor.log" \
  "$LOG_DIR/dashboard.log" \
  "$MONITOR_PID_FILE" \
  "$DASHBOARD_PID_FILE"; do
  if [[ -e "$path" && ! -w "$path" ]]; then
    runtime_needs_repair=true
  fi
done

if [[ "$runtime_needs_repair" == "true" ]]; then
  echo "Repairing runtime directory ownership..."
  sudo -v
  sudo install -d -m 775 -o "$RUN_USER" -g "$RUN_GROUP" \
    "$RUNTIME_DIR" "$PID_DIR" "$LOG_DIR"
  sudo chown -R "$RUN_USER:$RUN_GROUP" "$RUNTIME_DIR"
fi

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

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Missing occupancy model: $MODEL_PATH"
  echo "Copy model.npz to the Raspberry Pi before starting the system."
  exit 1
fi

if [[ -f "$MONITOR_PID_FILE" ]] && ! is_running "$MONITOR_PID_FILE" "workstation_monitor.py"; then
  echo "Removing stale monitor PID file."
  rm -f "$MONITOR_PID_FILE"
fi

if [[ -f "$DASHBOARD_PID_FILE" ]] && ! is_running "$DASHBOARD_PID_FILE" "dashboard_server.py"; then
  echo "Removing stale dashboard PID file."
  rm -f "$DASHBOARD_PID_FILE"
fi

monitor_started=false
dashboard_started=false

cleanup_started_processes() {
  if [[ "$dashboard_started" == "true" ]] && [[ -f "$DASHBOARD_PID_FILE" ]]; then
    local dashboard_pid
    dashboard_pid="$(cat "$DASHBOARD_PID_FILE")"
    if is_running "$DASHBOARD_PID_FILE" "dashboard_server.py"; then
      kill "$dashboard_pid" 2>/dev/null || true
      sleep 0.5
    fi
    if ps -p "$dashboard_pid" >/dev/null 2>&1; then
      echo "Warning: dashboard process $dashboard_pid is still running; PID file retained."
    else
      rm -f "$DASHBOARD_PID_FILE"
    fi
  fi
  if [[ "$monitor_started" == "true" ]] && [[ -f "$MONITOR_PID_FILE" ]]; then
    local monitor_pid
    monitor_pid="$(cat "$MONITOR_PID_FILE")"
    if is_running "$MONITOR_PID_FILE" "workstation_monitor.py"; then
      sudo kill "$monitor_pid" 2>/dev/null || true
      sleep 0.5
    fi
    if ps -p "$monitor_pid" >/dev/null 2>&1; then
      echo "Warning: monitor process $monitor_pid is still running; PID file retained."
    else
      rm -f "$MONITOR_PID_FILE"
    fi
  fi
}

if is_running "$MONITOR_PID_FILE" "workstation_monitor.py"; then
  echo "Monitor already running with PID $(cat "$MONITOR_PID_FILE")"
else
  echo "Starting workstation monitor..."
  sudo -v
  rm -f "$RUNTIME_DIR/status.json" "$RUNTIME_DIR/status.json.tmp"
  nohup sudo python3 sensor/workstation_monitor.py \
    --occupancy-model "$MODEL_PATH" \
    --occupied-confirm "$OCCUPIED_CONFIRM" \
    --leave-confirm "$LEAVE_CONFIRM" \
    --recently-used-minutes "$RECENTLY_USED_MINUTES" \
    --snapshot-interval "$SNAPSHOT_INTERVAL" \
    --status-interval "$STATUS_INTERVAL" \
    --log-interval "$MONITOR_LOG_INTERVAL" \
    --tool-safe "$TOOL_SAFE" \
    --tool-alert "$TOOL_ALERT" \
    --cooling-slope "$COOLING_SLOPE" \
    --cooling-min-drop "$COOLING_MIN_DROP" \
    --trend-min-seconds "$TREND_MIN_SECONDS" \
    --unattended-delay-seconds "$UNATTENDED_DELAY_SECONDS" \
    --safe-confirm-seconds "$SAFE_CONFIRM_SECONDS" \
    --no-window \
    > "$LOG_DIR/monitor.log" 2>&1 &
  echo "$!" > "$MONITOR_PID_FILE"
  monitor_started=true
  echo "Monitor PID: $(cat "$MONITOR_PID_FILE")"
fi

if is_running "$DASHBOARD_PID_FILE" "dashboard_server.py"; then
  echo "Dashboard server already running with PID $(cat "$DASHBOARD_PID_FILE")"
else
  echo "Starting dashboard server on port $DASHBOARD_PORT..."
  nohup python3 scripts/dashboard_server.py --host 0.0.0.0 --port "$DASHBOARD_PORT" \
    > "$LOG_DIR/dashboard.log" 2>&1 &
  echo "$!" > "$DASHBOARD_PID_FILE"
  dashboard_started=true
  echo "Dashboard PID: $(cat "$DASHBOARD_PID_FILE")"
fi

echo "Checking dashboard and sensor health..."
dashboard_ready=false
sensor_ready=false
for _ in $(seq 1 60); do
  if is_running "$MONITOR_PID_FILE" "workstation_monitor.py" \
    && is_running "$DASHBOARD_PID_FILE" "dashboard_server.py"; then
    health="$(curl -fsS "http://127.0.0.1:$DASHBOARD_PORT/healthz" 2>/dev/null || true)"
    if [[ "$health" == *'"service":"hot-seat-dashboard"'* ]]; then
      dashboard_ready=true
      if [[ "$health" == *'"sensor":"fresh"'* ]]; then
        sensor_ready=true
        break
      fi
    fi
  fi
  sleep 0.5
done

if [[ "$dashboard_ready" != "true" || "$sensor_ready" != "true" ]]; then
  echo "System did not become healthy within 30 seconds."
  echo "Monitor log: $LOG_DIR/monitor.log"
  echo "Dashboard log: $LOG_DIR/dashboard.log"
  cleanup_started_processes
  exit 1
fi

echo ""
echo "System started."
echo "Project page:   http://<raspberry-pi-ip>:$DASHBOARD_PORT/dashboard/"
echo "Live dashboard: http://<raspberry-pi-ip>:$DASHBOARD_PORT/dashboard/live/"
echo "Tailscale live: http://<tailscale-ip>:$DASHBOARD_PORT/dashboard/live/"
echo "Public tunnel target: http://localhost:$DASHBOARD_PORT/"
echo ""
echo "Logs:"
echo "  $LOG_DIR/monitor.log"
echo "  $LOG_DIR/dashboard.log"
echo ""
echo "Stop with:"
echo "  ./scripts/stop_system.sh"
