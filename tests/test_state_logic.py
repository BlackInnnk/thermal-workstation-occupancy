import sys
from pathlib import Path
import unittest

import numpy as np


SENSOR_DIR = Path(__file__).resolve().parents[1] / "sensor"
sys.path.insert(0, str(SENSOR_DIR))

from state_logic import (  # noqa: E402
    DetectionConfig,
    OCCUPANCY_FREE,
    OCCUPANCY_OCCUPIED,
    OCCUPANCY_RECENTLY_USED,
    SAFETY_COOLING,
    SAFETY_MONITORING,
    SAFETY_SAFE,
    SAFETY_UNATTENDED_HOT,
    OccupancyStateMachine,
    SafetyStateMachine,
    analyse_frame,
    top_fraction_mean,
)


HUMAN_ROI = (38, 4, 42, 55)
TOOL_ROI = (0, 25, 14, 9)


class FrameAnalysisTests(unittest.TestCase):
    def test_rejects_inconsistent_detection_config(self):
        with self.assertRaisesRegex(ValueError, "tool_alert_c"):
            DetectionConfig(tool_safe_c=45, tool_alert_c=40)

        with self.assertRaisesRegex(ValueError, "minimum drop positive"):
            DetectionConfig(cooling_min_drop_c=0)

    def test_detects_contiguous_human_heat_region(self):
        frame = np.full((60, 80), 22.0, dtype=np.float32)
        frame[20:30, 50:60] = 32.0
        metrics = analyse_frame(frame, HUMAN_ROI, TOOL_ROI, DetectionConfig())
        self.assertTrue(metrics.human_detected)
        self.assertGreaterEqual(metrics.human_component_pixels, 100)

    def test_ignores_small_hot_spot_in_human_area(self):
        frame = np.full((60, 80), 22.0, dtype=np.float32)
        frame[20:23, 50:53] = 60.0
        metrics = analyse_frame(frame, HUMAN_ROI, TOOL_ROI, DetectionConfig())
        self.assertFalse(metrics.human_detected)

    def test_rejects_non_finite_frame(self):
        frame = np.full((60, 80), 22.0, dtype=np.float32)
        frame[10, 10] = np.nan

        with self.assertRaisesRegex(ValueError, "non-finite"):
            analyse_frame(frame, HUMAN_ROI, TOOL_ROI, DetectionConfig())

    def test_rejects_invalid_top_fraction(self):
        with self.assertRaisesRegex(ValueError, "Top fraction"):
            top_fraction_mean(np.array([20.0, 21.0]), 0.0)


class OccupancyStateMachineTests(unittest.TestCase):
    def test_occupied_recently_used_and_free_transitions(self):
        config = DetectionConfig(
            occupied_confirm_seconds=5,
            leave_confirm_seconds=15,
            recently_used_seconds=900,
        )
        machine = OccupancyStateMachine(config, now=0)

        self.assertEqual(machine.update(True, 0).state, OCCUPANCY_FREE)
        self.assertEqual(machine.update(True, 5).state, OCCUPANCY_OCCUPIED)
        self.assertEqual(machine.update(False, 10).state, OCCUPANCY_OCCUPIED)
        self.assertEqual(machine.update(False, 25).state, OCCUPANCY_RECENTLY_USED)
        self.assertEqual(machine.update(False, 925).state, OCCUPANCY_FREE)

    def test_state_duration_resets_after_transition(self):
        config = DetectionConfig(occupied_confirm_seconds=3)
        machine = OccupancyStateMachine(config, now=10)

        machine.update(True, 10)
        result = machine.update(True, 13)

        self.assertTrue(result.changed)
        self.assertEqual(result.state, OCCUPANCY_OCCUPIED)
        self.assertEqual(result.state_seconds, 0)

    def test_sensor_gap_discards_pending_presence_confirmation(self):
        config = DetectionConfig(occupied_confirm_seconds=5)
        machine = OccupancyStateMachine(config, now=0)

        self.assertEqual(machine.update(True, 0).state, OCCUPANCY_FREE)
        machine.reset_observation_window()
        self.assertEqual(machine.update(True, 300).state, OCCUPANCY_FREE)
        self.assertEqual(machine.update(True, 305).state, OCCUPANCY_OCCUPIED)


class SafetyStateMachineTests(unittest.TestCase):
    def test_rejects_non_finite_temperature(self):
        machine = SafetyStateMachine(DetectionConfig(), now=0)

        with self.assertRaisesRegex(ValueError, "finite"):
            machine.update(float("nan"), occupied=False, now=0)

    def test_falling_temperature_is_cooling(self):
        config = DetectionConfig(
            trend_min_seconds=45,
            unattended_delay_seconds=180,
            tool_smoothing_samples=1,
        )
        machine = SafetyStateMachine(config, now=0)

        machine.update(70.0, occupied=False, now=0)
        result = machine.update(60.0, occupied=False, now=60)
        self.assertEqual(result.state, SAFETY_COOLING)
        self.assertLess(result.trend_c_per_min, 0)

    def test_cold_start_requires_safe_confirmation(self):
        config = DetectionConfig(safe_confirm_seconds=60, tool_smoothing_samples=1)
        machine = SafetyStateMachine(config, now=0)

        self.assertEqual(machine.update(30.0, occupied=False, now=0).state, SAFETY_MONITORING)
        self.assertEqual(machine.update(30.0, occupied=False, now=59).state, SAFETY_MONITORING)
        self.assertEqual(machine.update(30.0, occupied=False, now=60).state, SAFETY_SAFE)

    def test_sensor_gap_discards_pending_safe_confirmation(self):
        config = DetectionConfig(safe_confirm_seconds=60, tool_smoothing_samples=1)
        machine = SafetyStateMachine(config, now=0)

        self.assertEqual(machine.update(30.0, occupied=False, now=0).state, SAFETY_MONITORING)
        machine.reset_observation_window(now=300)
        self.assertEqual(machine.update(30.0, occupied=False, now=300).state, SAFETY_MONITORING)
        self.assertEqual(machine.update(30.0, occupied=False, now=359).state, SAFETY_MONITORING)
        self.assertEqual(machine.update(30.0, occupied=False, now=360).state, SAFETY_SAFE)

    def test_stable_high_temperature_becomes_unattended_alert(self):
        config = DetectionConfig(
            trend_min_seconds=45,
            unattended_delay_seconds=180,
            tool_smoothing_samples=1,
        )
        machine = SafetyStateMachine(config, now=0)

        machine.update(60.0, occupied=False, now=0)
        machine.update(60.0, occupied=False, now=60)
        machine.update(60.0, occupied=False, now=120)
        result = machine.update(60.0, occupied=False, now=180)
        self.assertEqual(result.state, SAFETY_UNATTENDED_HOT)

    def test_recently_hot_tool_remains_cooling_until_safe_is_confirmed(self):
        config = DetectionConfig(
            trend_min_seconds=30,
            safe_confirm_seconds=60,
            tool_smoothing_samples=1,
        )
        machine = SafetyStateMachine(config, now=0)

        machine.update(50.0, occupied=False, now=0)
        result = machine.update(42.0, occupied=False, now=40)
        self.assertEqual(result.state, SAFETY_COOLING)

        result = machine.update(37.5, occupied=False, now=70)
        self.assertEqual(result.state, SAFETY_COOLING)

        result = machine.update(37.0, occupied=False, now=130)
        self.assertEqual(result.state, SAFETY_SAFE)


if __name__ == "__main__":
    unittest.main()
