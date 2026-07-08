#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUNTIME_DIR="$ROOT_DIR/data/runtime"
PID_DIR="$RUNTIME_DIR/pids"
LOG_DIR="$RUNTIME_DIR/logs"
DASHBOARD_PORT="${DASHBOARD_PORT:-8000}"
PUBLIC_DASHBOARD_URL="${PUBLIC_DASHBOARD_URL:-}"

print_process_status() {
  local name="$1"
  local pid_file="$2"

  if [[ ! -f "$pid_file" ]]; then
    printf "%-20s not running\n" "$name"
    return
  fi

  local pid
  pid="$(cat "$pid_file")"
  if ps -p "$pid" >/dev/null 2>&1; then
    printf "%-20s running, PID %s\n" "$name" "$pid"
  else
    printf "%-20s stale PID file, process %s not found\n" "$name" "$pid"
  fi
}

echo "Thermal workstation system status"
echo "================================="
print_process_status "Monitor" "$PID_DIR/monitor.pid"
print_process_status "Dashboard" "$PID_DIR/dashboard.pid"
print_process_status "Tailscale Funnel" "$PID_DIR/tailscale_funnel.pid"

echo ""
echo "Dashboard endpoints"
echo "-------------------"
if curl -fsS "http://127.0.0.1:$DASHBOARD_PORT/dashboard/" >/dev/null 2>&1; then
  echo "Local dashboard:  OK  http://127.0.0.1:$DASHBOARD_PORT/dashboard/"
else
  echo "Local dashboard:  FAIL http://127.0.0.1:$DASHBOARD_PORT/dashboard/"
fi
if [[ -n "$PUBLIC_DASHBOARD_URL" ]]; then
  echo "Public dashboard: $PUBLIC_DASHBOARD_URL"
else
  echo "Public dashboard: see $LOG_DIR/tailscale_funnel.log"
fi

echo ""
echo "Latest live state"
echo "-----------------"
if [[ -f "$RUNTIME_DIR/status.json" ]]; then
  python3 - <<'PY'
import json
from pathlib import Path

status_path = Path("data/runtime/status.json")
payload = json.loads(status_path.read_text(encoding="utf-8"))
occupancy = payload.get("occupancy", {})
safety = payload.get("safety", {})
model = payload.get("model") or {}
snapshot = payload.get("snapshot", {})

print(f"Timestamp:     {payload.get('timestamp', '--')}")
print(f"Occupancy:     {occupancy.get('state', '--')}")
print(f"Safety:        {safety.get('state', '--')}")
tool_temp = safety.get("tool_temperature_c")
if isinstance(tool_temp, (int, float)):
    print(f"Tool temp:     {tool_temp:.1f} C")
probability = model.get("occupied_probability")
if isinstance(probability, (int, float)):
    print(f"ML occupied:   {probability * 100:.1f}%")
print(f"Snapshot:      {snapshot.get('updated_at', '--')}")
PY
else
  echo "No status file yet: $RUNTIME_DIR/status.json"
fi

echo ""
echo "Recent logs"
echo "-----------"
echo "Monitor log:   $LOG_DIR/monitor.log"
echo "Dashboard log: $LOG_DIR/dashboard.log"
echo "Funnel log:    $LOG_DIR/tailscale_funnel.log"
