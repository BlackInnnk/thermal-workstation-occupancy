#!/usr/bin/env python3

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np


OCCUPANCY_FREE = "FREE"
OCCUPANCY_OCCUPIED = "OCCUPIED"
OCCUPANCY_RECENTLY_USED = "RECENTLY_USED"

SAFETY_SAFE = "SAFE"
SAFETY_IN_USE = "IN_USE"
SAFETY_MONITORING = "MONITORING"
SAFETY_COOLING = "COOLING"
SAFETY_UNATTENDED_HOT = "UNATTENDED_HOT"


@dataclass(frozen=True)
class DetectionConfig:
    human_delta_c: float = 4.0
    human_floor_c: float = 27.0
    human_component_fraction: float = 0.025
    human_min_component_pixels: int = 20
    occupied_confirm_seconds: float = 5.0
    leave_confirm_seconds: float = 15.0
    recently_used_seconds: float = 15.0 * 60.0
    tool_top_fraction: float = 0.03
    tool_safe_c: float = 38.0
    tool_alert_c: float = 45.0
    cooling_slope_c_per_min: float = -0.5
    cooling_min_drop_c: float = 2.0
    trend_min_seconds: float = 45.0
    trend_window_seconds: float = 180.0
    unattended_delay_seconds: float = 180.0
    safe_confirm_seconds: float = 60.0
    tool_smoothing_samples: int = 5

    def __post_init__(self):
        finite_values = {
            "human_delta_c": self.human_delta_c,
            "human_floor_c": self.human_floor_c,
            "occupied_confirm_seconds": self.occupied_confirm_seconds,
            "leave_confirm_seconds": self.leave_confirm_seconds,
            "recently_used_seconds": self.recently_used_seconds,
            "tool_safe_c": self.tool_safe_c,
            "tool_alert_c": self.tool_alert_c,
            "cooling_slope_c_per_min": self.cooling_slope_c_per_min,
            "cooling_min_drop_c": self.cooling_min_drop_c,
            "trend_min_seconds": self.trend_min_seconds,
            "trend_window_seconds": self.trend_window_seconds,
            "unattended_delay_seconds": self.unattended_delay_seconds,
            "safe_confirm_seconds": self.safe_confirm_seconds,
        }
        invalid = [name for name, value in finite_values.items() if not math.isfinite(value)]
        if invalid:
            raise ValueError(f"Detection settings must be finite: {', '.join(invalid)}")
        if not 0 < self.human_component_fraction <= 1:
            raise ValueError("human_component_fraction must be in (0, 1]")
        if not 0 < self.tool_top_fraction <= 1:
            raise ValueError("tool_top_fraction must be in (0, 1]")
        if self.human_min_component_pixels < 1 or self.tool_smoothing_samples < 1:
            raise ValueError("Pixel and smoothing sample counts must be positive")
        duration_names = (
            "occupied_confirm_seconds",
            "leave_confirm_seconds",
            "recently_used_seconds",
            "trend_min_seconds",
            "trend_window_seconds",
            "unattended_delay_seconds",
            "safe_confirm_seconds",
        )
        if any(getattr(self, name) < 0 for name in duration_names):
            raise ValueError("Detection durations cannot be negative")
        if self.trend_window_seconds < self.trend_min_seconds:
            raise ValueError("trend_window_seconds must be at least trend_min_seconds")
        if self.tool_alert_c <= self.tool_safe_c:
            raise ValueError("tool_alert_c must be greater than tool_safe_c")
        if self.cooling_slope_c_per_min >= 0 or self.cooling_min_drop_c <= 0:
            raise ValueError("Cooling slope must be negative and minimum drop positive")


@dataclass(frozen=True)
class FrameMetrics:
    ambient_c: float
    human_threshold_c: float
    human_p95_c: float
    human_hot_pixels: int
    human_hot_fraction: float
    human_component_pixels: int
    human_component_fraction: float
    human_detected: bool
    tool_max_c: float
    tool_p95_c: float
    tool_hot_mean_c: float


@dataclass(frozen=True)
class OccupancyResult:
    state: str
    changed: bool
    state_seconds: float
    recently_used_remaining_seconds: float


@dataclass(frozen=True)
class SafetyResult:
    state: str
    changed: bool
    state_seconds: float
    tool_temperature_c: float
    trend_c_per_min: float | None
    unoccupied_seconds: float


def extract_roi(frame, roi):
    x, y, width, height = roi
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError(f"Invalid ROI: {roi}")
    if x + width > frame.shape[1] or y + height > frame.shape[0]:
        raise ValueError(f"ROI {roi} exceeds frame shape {frame.shape}")
    return frame[y : y + height, x : x + width]


def largest_component_size(mask):
    rows, cols = mask.shape
    visited = np.zeros(mask.shape, dtype=bool)
    largest = 0

    for row in range(rows):
        for col in range(cols):
            if not mask[row, col] or visited[row, col]:
                continue

            size = 0
            stack = [(row, col)]
            visited[row, col] = True

            while stack:
                current_row, current_col = stack.pop()
                size += 1

                for next_row, next_col in (
                    (current_row - 1, current_col),
                    (current_row + 1, current_col),
                    (current_row, current_col - 1),
                    (current_row, current_col + 1),
                ):
                    if (
                        0 <= next_row < rows
                        and 0 <= next_col < cols
                        and mask[next_row, next_col]
                        and not visited[next_row, next_col]
                    ):
                        visited[next_row, next_col] = True
                        stack.append((next_row, next_col))

            largest = max(largest, size)

    return largest


def top_fraction_mean(values, fraction):
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    if flat.size == 0:
        raise ValueError("Cannot calculate a top-fraction mean from an empty array.")
    if not 0 < fraction <= 1:
        raise ValueError(f"Top fraction must be in (0, 1], got {fraction}.")
    if not np.all(np.isfinite(flat)):
        raise ValueError("Temperature values must all be finite.")
    count = max(3, int(math.ceil(flat.size * fraction)))
    count = min(count, flat.size)
    top_values = np.partition(flat, flat.size - count)[-count:]
    return float(np.mean(top_values))


def analyse_frame(temp_c, human_roi, tool_roi, config):
    temp_c = np.asarray(temp_c, dtype=np.float32)
    if temp_c.ndim != 2 or temp_c.size == 0:
        raise ValueError(f"Thermal frame must be a non-empty 2D array, got {temp_c.shape}.")
    if not np.all(np.isfinite(temp_c)):
        raise ValueError("Thermal frame contains non-finite values.")

    human_region = extract_roi(temp_c, human_roi)
    tool_region = extract_roi(temp_c, tool_roi)

    ambient_c = float(np.percentile(temp_c, 30))
    human_threshold_c = max(config.human_floor_c, ambient_c + config.human_delta_c)
    human_mask = human_region >= human_threshold_c
    human_hot_pixels = int(np.sum(human_mask))
    human_component_pixels = largest_component_size(human_mask)
    human_hot_fraction = human_hot_pixels / human_region.size
    human_component_fraction = human_component_pixels / human_region.size
    required_component_pixels = max(
        config.human_min_component_pixels,
        int(math.ceil(human_region.size * config.human_component_fraction)),
    )

    return FrameMetrics(
        ambient_c=ambient_c,
        human_threshold_c=human_threshold_c,
        human_p95_c=float(np.percentile(human_region, 95)),
        human_hot_pixels=human_hot_pixels,
        human_hot_fraction=human_hot_fraction,
        human_component_pixels=human_component_pixels,
        human_component_fraction=human_component_fraction,
        human_detected=human_component_pixels >= required_component_pixels,
        tool_max_c=float(np.max(tool_region)),
        tool_p95_c=float(np.percentile(tool_region, 95)),
        tool_hot_mean_c=top_fraction_mean(tool_region, config.tool_top_fraction),
    )


class OccupancyStateMachine:
    def __init__(self, config, now=0.0):
        self.config = config
        self.state = OCCUPANCY_FREE
        self.state_since = now
        self.presence_since = None
        self.absence_since = now
        self.recently_used_until = None

    def update(self, human_detected, now):
        previous_state = self.state

        if human_detected:
            self.absence_since = None
            if self.presence_since is None:
                self.presence_since = now

            if now - self.presence_since >= self.config.occupied_confirm_seconds:
                self._set_state(OCCUPANCY_OCCUPIED, now)
                self.recently_used_until = None
        else:
            self.presence_since = None
            if self.absence_since is None:
                self.absence_since = now

            if (
                self.state == OCCUPANCY_OCCUPIED
                and now - self.absence_since >= self.config.leave_confirm_seconds
            ):
                self._set_state(OCCUPANCY_RECENTLY_USED, now)
                self.recently_used_until = now + self.config.recently_used_seconds
            elif (
                self.state == OCCUPANCY_RECENTLY_USED
                and self.recently_used_until is not None
                and now >= self.recently_used_until
            ):
                self._set_state(OCCUPANCY_FREE, now)
                self.recently_used_until = None

        remaining = 0.0
        if self.state == OCCUPANCY_RECENTLY_USED and self.recently_used_until is not None:
            remaining = max(0.0, self.recently_used_until - now)

        return OccupancyResult(
            state=self.state,
            changed=self.state != previous_state,
            state_seconds=max(0.0, now - self.state_since),
            recently_used_remaining_seconds=remaining,
        )

    def reset_observation_window(self):
        """Discard pending enter/leave evidence after a sensor data gap."""
        self.presence_since = None
        self.absence_since = None

    def _set_state(self, state, now):
        if state != self.state:
            self.state = state
            self.state_since = now


class SafetyStateMachine:
    def __init__(self, config, now=0.0):
        self.config = config
        self.state = SAFETY_MONITORING
        self.state_since = now
        self.unoccupied_since = now
        self.history = deque()
        self.raw_samples = deque(maxlen=max(1, config.tool_smoothing_samples))
        self.safe_below_since = None

    def update(self, tool_temperature_c, occupied, now):
        if not math.isfinite(tool_temperature_c):
            raise ValueError("Tool temperature must be finite.")

        previous_state = self.state

        if occupied:
            self.history.clear()
            self.raw_samples.clear()
            self.unoccupied_since = None
            self.safe_below_since = None
            self._set_state(SAFETY_IN_USE, now)
            return SafetyResult(
                state=self.state,
                changed=self.state != previous_state,
                state_seconds=max(0.0, now - self.state_since),
                tool_temperature_c=tool_temperature_c,
                trend_c_per_min=None,
                unoccupied_seconds=0.0,
            )

        if self.unoccupied_since is None:
            self.unoccupied_since = now

        self.raw_samples.append(tool_temperature_c)
        smoothed_temperature = float(np.median(self.raw_samples))
        self.history.append((now, smoothed_temperature))
        self._prune_history(now)

        trend = self._calculate_trend()
        unoccupied_seconds = max(0.0, now - self.unoccupied_since)
        history_seconds = self.history[-1][0] - self.history[0][0] if len(self.history) >= 2 else 0.0
        recent_peak_c = max(temperature for _, temperature in self.history)
        drop_from_peak_c = recent_peak_c - smoothed_temperature
        has_recent_hot_peak = recent_peak_c >= self.config.tool_alert_c
        has_enough_history = history_seconds >= self.config.trend_min_seconds

        if smoothed_temperature < self.config.tool_safe_c:
            if self.safe_below_since is None:
                self.safe_below_since = now
        else:
            self.safe_below_since = None

        safe_confirmed = (
            self.safe_below_since is not None
            and now - self.safe_below_since >= self.config.safe_confirm_seconds
        )
        cooling_by_trend = (
            trend is not None
            and has_enough_history
            and trend <= self.config.cooling_slope_c_per_min
        )
        cooling_by_drop = (
            has_recent_hot_peak
            and has_enough_history
            and drop_from_peak_c >= self.config.cooling_min_drop_c
        )
        cooling_after_recent_hot = has_recent_hot_peak and smoothed_temperature < self.config.tool_safe_c

        if safe_confirmed:
            self._set_state(SAFETY_SAFE, now)
        elif cooling_by_trend or cooling_by_drop or cooling_after_recent_hot:
            self._set_state(SAFETY_COOLING, now)
        elif (
            smoothed_temperature >= self.config.tool_alert_c
            and unoccupied_seconds >= self.config.unattended_delay_seconds
            and has_enough_history
        ):
            self._set_state(SAFETY_UNATTENDED_HOT, now)
        else:
            self._set_state(SAFETY_MONITORING, now)

        return SafetyResult(
            state=self.state,
            changed=self.state != previous_state,
            state_seconds=max(0.0, now - self.state_since),
            tool_temperature_c=smoothed_temperature,
            trend_c_per_min=trend,
            unoccupied_seconds=unoccupied_seconds,
        )

    def reset_observation_window(self, now):
        """Require fresh trend and threshold evidence after a sensor data gap."""
        self.history.clear()
        self.raw_samples.clear()
        self.unoccupied_since = now
        self.safe_below_since = None

    def _prune_history(self, now):
        cutoff = now - self.config.trend_window_seconds
        while len(self.history) > 1 and self.history[0][0] < cutoff:
            self.history.popleft()

    def _calculate_trend(self):
        if len(self.history) < 2:
            return None

        start_time = self.history[0][0]
        times_minutes = np.array(
            [(timestamp - start_time) / 60.0 for timestamp, _ in self.history],
            dtype=np.float32,
        )
        temperatures = np.array([temperature for _, temperature in self.history], dtype=np.float32)

        if float(times_minutes[-1]) <= 0.0:
            return None

        return float(np.polyfit(times_minutes, temperatures, 1)[0])

    def _set_state(self, state, now):
        if state != self.state:
            self.state = state
            self.state_since = now
