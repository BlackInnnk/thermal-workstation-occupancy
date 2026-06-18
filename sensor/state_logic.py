#!/usr/bin/env python3

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
    trend_min_seconds: float = 45.0
    trend_window_seconds: float = 180.0
    unattended_delay_seconds: float = 180.0
    tool_smoothing_samples: int = 5


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
    count = max(3, int(math.ceil(flat.size * fraction)))
    count = min(count, flat.size)
    top_values = np.partition(flat, flat.size - count)[-count:]
    return float(np.mean(top_values))


def analyse_frame(temp_c, human_roi, tool_roi, config):
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

    def _set_state(self, state, now):
        if state != self.state:
            self.state = state
            self.state_since = now


class SafetyStateMachine:
    def __init__(self, config, now=0.0):
        self.config = config
        self.state = SAFETY_SAFE
        self.state_since = now
        self.unoccupied_since = now
        self.history = deque()
        self.raw_samples = deque(maxlen=max(1, config.tool_smoothing_samples))

    def update(self, tool_temperature_c, occupied, now):
        previous_state = self.state

        if occupied:
            self.history.clear()
            self.raw_samples.clear()
            self.unoccupied_since = None
            self._set_state(SAFETY_IN_USE, now)
            return SafetyResult(
                state=self.state,
                changed=self.state != previous_state,
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

        if smoothed_temperature < self.config.tool_safe_c:
            self._set_state(SAFETY_SAFE, now)
        elif (
            trend is not None
            and history_seconds >= self.config.trend_min_seconds
            and trend <= self.config.cooling_slope_c_per_min
        ):
            self._set_state(SAFETY_COOLING, now)
        elif (
            smoothed_temperature >= self.config.tool_alert_c
            and unoccupied_seconds >= self.config.unattended_delay_seconds
            and history_seconds >= self.config.trend_min_seconds
        ):
            self._set_state(SAFETY_UNATTENDED_HOT, now)
        else:
            self._set_state(SAFETY_MONITORING, now)

        return SafetyResult(
            state=self.state,
            changed=self.state != previous_state,
            tool_temperature_c=smoothed_temperature,
            trend_c_per_min=trend,
            unoccupied_seconds=unoccupied_seconds,
        )

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
