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
  local use_sudo="${3:-false}"

  if [[ ! -f "$pid_file" ]]; then
    echo "$name is not running."
    return
  fi

  local pid
  pid="$(cat "$pid_file")"
  if ! ps -p "$pid" >/dev/null 2>&1; then
    echo "$name PID file existed, but process $pid is not running."
    rm -f "$pid_file"
    return
  fi

  echo "Stopping $name with PID $pid..."
  if [[ "$use_sudo" == "true" ]]; then
    sudo kill "$pid" 2>/dev/null || true
  else
    kill "$pid" 2>/dev/null || true
  fi

  sleep 1
  if ps -p "$pid" >/dev/null 2>&1; then
    echo "$name did not stop cleanly; forcing shutdown."
    if [[ "$use_sudo" == "true" ]]; then
      sudo kill -9 "$pid" 2>/dev/null || true
    else
      kill -9 "$pid" 2>/dev/null || true
    fi
  fi

  rm -f "$pid_file"
  echo "$name stopped."
}

stop_process "workstation monitor" "$MONITOR_PID_FILE" true
stop_process "dashboard server" "$DASHBOARD_PID_FILE" false

echo "System stopped."
