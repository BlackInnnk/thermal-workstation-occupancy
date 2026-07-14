import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from dashboard_server import (  # noqa: E402
    DashboardRequestHandler,
    STATUS_STALE_SECONDS,
    resolve_file_request,
    runtime_health,
)


class RuntimeHealthTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = TemporaryDirectory()
        self.runtime_dir = Path(self.temporary_directory.name)
        self.now = datetime(2026, 7, 13, 12, 0, tzinfo=timezone.utc)

    def tearDown(self):
        self.temporary_directory.cleanup()

    def write_status(self, timestamp, **overrides):
        payload = {
            "timestamp": timestamp,
            "occupancy": {"state": "FREE"},
            "safety": {"state": "SAFE", "tool_temperature_c": 24.5},
        }
        payload.update(overrides)
        (self.runtime_dir / "status.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )

    def test_missing_status_is_reported(self):
        health = runtime_health(self.runtime_dir, now=self.now)
        self.assertEqual(health["server"], "ok")
        self.assertEqual(health["sensor"], "missing")

    def test_recent_status_is_fresh(self):
        timestamp = self.now - timedelta(seconds=5)
        self.write_status(timestamp.isoformat())

        health = runtime_health(self.runtime_dir, now=self.now)

        self.assertEqual(health["sensor"], "fresh")
        self.assertEqual(health["status_age_seconds"], 5.0)

    def test_old_status_is_stale(self):
        timestamp = self.now - timedelta(seconds=STATUS_STALE_SECONDS + 1)
        self.write_status(timestamp.isoformat())

        health = runtime_health(self.runtime_dir, now=self.now)

        self.assertEqual(health["sensor"], "stale")

    def test_invalid_status_is_reported(self):
        (self.runtime_dir / "status.json").write_text("not json", encoding="utf-8")

        health = runtime_health(self.runtime_dir, now=self.now)

        self.assertEqual(health["sensor"], "invalid")

    def test_incomplete_status_is_not_reported_as_fresh(self):
        (self.runtime_dir / "status.json").write_text(
            json.dumps({"timestamp": self.now.isoformat()}),
            encoding="utf-8",
        )

        health = runtime_health(self.runtime_dir, now=self.now)

        self.assertEqual(health["sensor"], "invalid")

    def test_unknown_state_is_not_reported_as_fresh(self):
        self.write_status(
            self.now.isoformat(),
            occupancy={"state": "MAYBE"},
        )

        health = runtime_health(self.runtime_dir, now=self.now)

        self.assertEqual(health["sensor"], "invalid")


class DashboardRouteTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        (self.root / "dashboard").mkdir()
        (self.root / "assets").mkdir()
        (self.root / "data" / "runtime").mkdir(parents=True)
        (self.root / "data" / "raw").mkdir(parents=True)
        (self.root / "index.html").write_text("landing", encoding="utf-8")
        (self.root / "dashboard" / "index.html").write_text("live", encoding="utf-8")
        (self.root / "data" / "raw" / "secret.txt").write_text("private", encoding="utf-8")

    def tearDown(self):
        self.temporary_directory.cleanup()

    def resolve(self, path):
        return resolve_file_request(
            path,
            self.root,
            self.root / "dashboard",
            self.root / "assets",
            self.root / "data" / "runtime",
        )

    def test_landing_and_live_routes_are_distinct(self):
        landing, landing_cache = self.resolve("/dashboard/")
        live, live_cache = self.resolve("/dashboard/live/")

        self.assertEqual(landing, self.root / "index.html")
        self.assertEqual(live, (self.root / "dashboard" / "index.html").resolve())
        self.assertEqual(landing_cache, "no-cache")
        self.assertEqual(live_cache, "no-cache")

    def test_raw_dataset_is_not_served(self):
        self.assertIsNone(self.resolve("/data/raw/secret.txt"))

    def test_runtime_events_file_is_served_without_cache(self):
        event_path, cache_policy = self.resolve("/data/runtime/events.json")

        self.assertEqual(event_path, self.root / "data" / "runtime" / "events.json")
        self.assertEqual(cache_policy, "no-store")

    def test_asset_traversal_is_rejected(self):
        self.assertIsNone(self.resolve("/assets/../data/raw/secret.txt"))

    def test_common_headers_restrict_browser_capabilities(self):
        handler = DashboardRequestHandler.__new__(DashboardRequestHandler)
        headers = {}
        handler.send_header = headers.__setitem__

        handler._send_common_headers("no-store")

        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertEqual(headers["X-Frame-Options"], "DENY")
        self.assertEqual(headers["Cross-Origin-Opener-Policy"], "same-origin")
        self.assertIn("camera=()", headers["Permissions-Policy"])
        self.assertIn("object-src 'none'", headers["Content-Security-Policy"])


if __name__ == "__main__":
    unittest.main()
