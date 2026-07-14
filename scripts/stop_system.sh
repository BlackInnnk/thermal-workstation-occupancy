#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PID_DIR="$ROOT_DIR/data/runtime/pids"
MONITOR_PID_FILE="$PID_DIR/monitor.pid"
DASHBOARD_PID_FILE="$PID_DIR/dashboard.pid"

stop_process() {
  local name="$1"
  local pid_file="$2"
  local expected="$3"
  local use_sudo="${4:-false}"

  if [[ ! -f "$pid_file" ]]; then
    echo "$name is not running."
    return
  fi

  local pid
  pid="$(cat "$pid_file")"
  if ! [[ "$pid" =~ ^[0-9]+$ ]] || ! ps -p "$pid" >/dev/null 2>&1; then
    echo "$name PID file existed, but process $pid is not running."
    rm -f "$pid_file"
    return
  fi

  local command
  command="$(ps -p "$pid" -o command= 2>/dev/null || true)"
  if [[ "$command" != *"$expected"* ]]; then
    echo "$name PID file points to an unrelated process; removing stale PID file."
    rm -f "$pid_file"
    return
  fi

  echo "Stopping $name with PID $pid..."
  if [[ "$use_sudo" == "true" ]]; then
    if ! sudo kill "$pid" 2>/dev/null; then
      echo "Initial stop signal for $name failed; checking whether it already exited."
    fi
  else
    if ! kill "$pid" 2>/dev/null; then
      echo "Initial stop signal for $name failed; checking whether it already exited."
    fi
  fi

  sleep 1
  if ps -p "$pid" >/dev/null 2>&1; then
    echo "$name did not stop cleanly; forcing shutdown."
    if [[ "$use_sudo" == "true" ]]; then
      if ! sudo kill -9 "$pid" 2>/dev/null; then
        echo "Unable to send a forced stop signal to $name."
      fi
    else
      if ! kill -9 "$pid" 2>/dev/null; then
        echo "Unable to send a forced stop signal to $name."
      fi
    fi
    sleep 0.5
  fi

  if ps -p "$pid" >/dev/null 2>&1; then
    echo "ERROR: $name is still running; PID file retained for recovery."
    return 1
  fi

  rm -f "$pid_file"
  echo "$name stopped."
}

stop_failed=false
monitor_stopped=true
if ! stop_process "workstation monitor" "$MONITOR_PID_FILE" "workstation_monitor.py" true; then
  stop_failed=true
  monitor_stopped=false
fi
if ! stop_process "dashboard server" "$DASHBOARD_PID_FILE" "dashboard_server.py" false; then
  stop_failed=true
fi

if [[ "$monitor_stopped" == "true" ]]; then
  if ! rm -f "$ROOT_DIR/data/runtime/status.json" "$ROOT_DIR/data/runtime/status.json.tmp" 2>/dev/null; then
    sudo rm -f "$ROOT_DIR/data/runtime/status.json" "$ROOT_DIR/data/runtime/status.json.tmp"
  fi
fi

if [[ "$stop_failed" == "true" ]]; then
  echo "System stop was incomplete. Check the retained PID files and running processes."
  exit 1
fi

echo "System stopped."
