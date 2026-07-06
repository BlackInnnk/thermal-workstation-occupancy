#!/usr/bin/env python3
"""Serve the dashboard and only the live runtime files it needs."""

from __future__ import annotations

import argparse
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


class DashboardRequestHandler(BaseHTTPRequestHandler):
    server_version = "ThermalDashboard/1.0"

    def do_GET(self) -> None:
        self._handle_request(send_body=True)

    def do_HEAD(self) -> None:
        self._handle_request(send_body=False)

    def _handle_request(self, send_body: bool) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        if path in {"", "/"}:
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", "/dashboard/")
            self.end_headers()
            return

        if path == "/dashboard":
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", "/dashboard/")
            self.end_headers()
            return

        file_path: Path | None = None
        cache_control = "no-cache"

        if path.startswith("/dashboard/"):
            relative = path.removeprefix("/dashboard/")
            if relative == "":
                relative = "index.html"
            file_path = self._safe_child(self.server.dashboard_dir, relative)
        elif path == "/data/runtime/status.json":
            file_path = self.server.runtime_dir / "status.json"
            cache_control = "no-store"
        elif path == "/data/runtime/thermal_view.jpg":
            file_path = self.server.runtime_dir / "thermal_view.jpg"
            cache_control = "no-store"

        if file_path is None or not file_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        self._send_file(file_path, cache_control, send_body)

    def log_message(self, format: str, *args: object) -> None:
        return

    def _safe_child(self, base_dir: Path, relative_path: str) -> Path | None:
        candidate = (base_dir / relative_path).resolve()
        try:
            candidate.relative_to(base_dir)
        except ValueError:
            return None
        return candidate

    def _send_file(self, path: Path, cache_control: str, send_body: bool) -> None:
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        data = path.read_bytes()

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", cache_control)
        self.end_headers()
        if send_body:
            self.wfile.write(data)


class DashboardServer(ThreadingHTTPServer):
    def __init__(self, server_address: tuple[str, int], handler_class: type[BaseHTTPRequestHandler], root_dir: Path):
        super().__init__(server_address, handler_class)
        self.root_dir = root_dir.resolve()
        self.dashboard_dir = (self.root_dir / "dashboard").resolve()
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
    server = DashboardServer((args.host, args.port), DashboardRequestHandler, args.root)
    print(f"Serving dashboard on http://{args.host}:{args.port}/dashboard/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
