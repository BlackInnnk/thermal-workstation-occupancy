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
    SAFETY_UNATTENDED_HOT,
    OccupancyStateMachine,
    SafetyStateMachine,
    analyse_frame,
)


HUMAN_ROI = (38, 4, 42, 55)
TOOL_ROI = (0, 25, 14, 9)


class FrameAnalysisTests(unittest.TestCase):
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


class SafetyStateMachineTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
