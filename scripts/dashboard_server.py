#!/usr/bin/env python3
"""Serve the landing page, dashboard, and only the live runtime files they need."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


STATUS_STALE_SECONDS = 15.0
OCCUPANCY_STATES = {"FREE", "OCCUPIED", "RECENTLY_USED"}
SAFETY_STATES = {"SAFE", "IN_USE", "MONITORING", "COOLING", "UNATTENDED_HOT"}


def status_timestamp(payload: object) -> datetime:
    """Validate the core monitor payload and return its UTC timestamp."""
    if not isinstance(payload, dict):
        raise ValueError("Status payload must be an object")

    occupancy = payload.get("occupancy")
    safety = payload.get("safety")
    if not isinstance(occupancy, dict) or occupancy.get("state") not in OCCUPANCY_STATES:
        raise ValueError("Status payload has no valid occupancy state")
    if not isinstance(safety, dict) or safety.get("state") not in SAFETY_STATES:
        raise ValueError("Status payload has no valid safety state")

    tool_temperature = safety.get("tool_temperature_c")
    if (
        isinstance(tool_temperature, bool)
        or not isinstance(tool_temperature, (int, float))
        or not math.isfinite(tool_temperature)
    ):
        raise ValueError("Status payload has no valid tool temperature")

    timestamp = datetime.fromisoformat(str(payload["timestamp"]))
    if timestamp.tzinfo is None:
        timestamp = timestamp.astimezone()
    return timestamp.astimezone(timezone.utc)


def runtime_health(runtime_dir: Path, now: datetime | None = None) -> dict[str, object]:
    """Return dashboard and sensor health without exposing runtime file contents."""
    health: dict[str, object] = {
        "service": "hot-seat-dashboard",
        "server": "ok",
        "sensor": "missing",
    }
    status_path = runtime_dir / "status.json"
    if not status_path.is_file():
        return health

    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
        timestamp = status_timestamp(payload)
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        health["sensor"] = "invalid"
        return health

    current_time = now or datetime.now(timezone.utc)
    if current_time.tzinfo is None:
        current_time = current_time.astimezone()
    current_time = current_time.astimezone(timezone.utc)
    age_seconds = (current_time - timestamp).total_seconds()
    health["status_age_seconds"] = round(max(0.0, age_seconds), 3)
    health["sensor"] = (
        "fresh" if -5.0 <= age_seconds <= STATUS_STALE_SECONDS else "stale"
    )
    return health


def safe_child(base_dir: Path, relative_path: str) -> Path | None:
    """Resolve a child path without allowing traversal outside its public root."""
    base_dir = base_dir.resolve()
    candidate = (base_dir / relative_path).resolve()
    try:
        candidate.relative_to(base_dir)
    except ValueError:
        return None
    return candidate


def resolve_file_request(
    path: str,
    root_dir: Path,
    dashboard_dir: Path,
    assets_dir: Path,
    runtime_dir: Path,
) -> tuple[Path, str] | None:
    """Map an allowed URL path to a file and cache policy."""
    if path in {"", "/", "/index.html", "/dashboard/", "/dashboard/index.html"}:
        return root_dir / "index.html", "no-cache"

    if path.startswith("/dashboard/live/"):
        relative = path.removeprefix("/dashboard/live/") or "index.html"
        file_path = safe_child(dashboard_dir, relative)
        return (file_path, "no-cache") if file_path is not None else None

    if path.startswith("/assets/"):
        relative = path.removeprefix("/assets/")
        file_path = safe_child(assets_dir, relative)
        return (file_path, "public, max-age=3600") if file_path is not None else None

    if path == "/data/runtime/status.json":
        return runtime_dir / "status.json", "no-store"
    if path == "/data/runtime/events.json":
        return runtime_dir / "events.json", "no-store"
    if path == "/data/runtime/thermal_view.jpg":
        return runtime_dir / "thermal_view.jpg", "no-store"
    return None


class DashboardRequestHandler(BaseHTTPRequestHandler):
    server_version = "ThermalDashboard/1.0"

    def do_GET(self) -> None:
        self._handle_request(send_body=True)

    def do_HEAD(self) -> None:
        self._handle_request(send_body=False)

    def _handle_request(self, send_body: bool) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        if path == "/healthz":
            self._send_json(runtime_health(self.server.runtime_dir), send_body)
            return

        if path in {"", "/", "/index.html"}:
            landing_page = self.server.root_dir / "index.html"
            if landing_page.is_file():
                self._send_file(landing_page, "no-cache", send_body)
                return
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", "/dashboard/")
            self.end_headers()
            return

        if path == "/dashboard":
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", "/dashboard/")
            self.end_headers()
            return

        if path == "/dashboard/live":
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", "/dashboard/live/")
            self.end_headers()
            return

        resolved = resolve_file_request(
            path,
            self.server.root_dir,
            self.server.dashboard_dir,
            self.server.assets_dir,
            self.server.runtime_dir,
        )
        if resolved is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        file_path, cache_control = resolved
        if not file_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        self._send_file(file_path, cache_control, send_body)

    def log_message(self, format: str, *args: object) -> None:
        return

    def _send_file(self, path: Path, cache_control: str, send_body: bool) -> None:
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        try:
            data = path.read_bytes()
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self._send_common_headers(cache_control)
        self.end_headers()
        if send_body:
            self.wfile.write(data)

    def _send_json(self, payload: dict[str, object], send_body: bool) -> None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self._send_common_headers("no-store")
        self.end_headers()
        if send_body:
            self.wfile.write(data)

    def _send_common_headers(self, cache_control: str) -> None:
        self.send_header("Cache-Control", cache_control)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header(
            "Permissions-Policy",
            "camera=(), microphone=(), geolocation=()",
        )
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; img-src 'self' data:; "
            "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; "
            "form-action 'none'; object-src 'none'",
        )


class DashboardServer(ThreadingHTTPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], handler_class: type[BaseHTTPRequestHandler], root_dir: Path):
        super().__init__(server_address, handler_class)
        self.root_dir = root_dir.resolve()
        self.dashboard_dir = (self.root_dir / "dashboard").resolve()
        self.assets_dir = (self.root_dir / "assets").resolve()
        self.runtime_dir = (self.root_dir / "data" / "runtime").resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the thermal workstation dashboard safely.")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root containing dashboard/ and data/runtime/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.port <= 65535:
        raise ValueError("--port must be between 1 and 65535")
    server = DashboardServer((args.host, args.port), DashboardRequestHandler, args.root)
    print(f"Serving landing page on http://{args.host}:{args.port}/dashboard/", flush=True)
    print(f"Serving live dashboard on http://{args.host}:{args.port}/dashboard/live/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
