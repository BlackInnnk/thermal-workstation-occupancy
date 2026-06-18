#!/usr/bin/env python3

import argparse
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import time

import cv2
import numpy as np
from pylepton import Lepton

from state_logic import (
    DetectionConfig,
    OCCUPANCY_OCCUPIED,
    OCCUPANCY_RECENTLY_USED,
    SAFETY_COOLING,
    SAFETY_IN_USE,
    SAFETY_MONITORING,
    SAFETY_SAFE,
    SAFETY_UNATTENDED_HOT,
    OccupancyStateMachine,
    SafetyStateMachine,
    analyse_frame,
)
from thermal_roi_viewer import DEFAULT_ROIS, raw_to_celsius


FRAME_WIDTH = 80
FRAME_HEIGHT = 60

OCCUPANCY_COLORS = {
    "FREE": (80, 210, 130),
    "OCCUPIED": (50, 100, 255),
    "RECENTLY_USED": (80, 190, 255),
}

SAFETY_COLORS = {
    SAFETY_SAFE: (80, 210, 130),
    SAFETY_IN_USE: (100, 180, 255),
    SAFETY_MONITORING: (100, 190, 255),
    SAFETY_COOLING: (255, 190, 80),
    SAFETY_UNATTENDED_HOT: (40, 40, 255),
}


def make_heatmap(frame, scale):
    normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
    normalized = np.uint8(normalized)
    color = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
    return cv2.resize(color, (FRAME_WIDTH * scale, FRAME_HEIGHT * scale), interpolation=cv2.INTER_NEAREST)


def draw_roi(display, name, roi, scale, color):
    x, y, width, height = roi
    start = (x * scale, y * scale)
    end = ((x + width) * scale, (y + height) * scale)
    cv2.rectangle(display, start, end, color, 2)
    cv2.putText(
        display,
        name,
        (start[0] + 6, max(22, start[1] + 22)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        color,
        2,
        cv2.LINE_AA,
    )


def format_duration(seconds):
    seconds = max(0, int(seconds))
    minutes, remaining_seconds = divmod(seconds, 60)
    return f"{minutes:02d}:{remaining_seconds:02d}"


def put_panel_line(panel, text, row, color=(235, 235, 235), scale=0.54, thickness=1):
    cv2.putText(
        panel,
        text,
        (18, row),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def make_status_panel(height, occupancy, safety, metrics, config):
    panel = np.full((height, 380, 3), (24, 27, 32), dtype=np.uint8)
    occupancy_color = OCCUPANCY_COLORS[occupancy.state]
    safety_color = SAFETY_COLORS[safety.state]

    put_panel_line(panel, "WORKSTATION STATUS", 32, (180, 185, 195), 0.58, 2)
    put_panel_line(panel, occupancy.state.replace("_", " "), 72, occupancy_color, 0.9, 2)

    if occupancy.state == OCCUPANCY_RECENTLY_USED:
        put_panel_line(
            panel,
            f"History remaining: {format_duration(occupancy.recently_used_remaining_seconds)}",
            102,
            occupancy_color,
        )
    elif occupancy.state == OCCUPANCY_OCCUPIED:
        put_panel_line(panel, f"Occupied for: {format_duration(occupancy.state_seconds)}", 102)
    else:
        put_panel_line(panel, f"Free for: {format_duration(occupancy.state_seconds)}", 102)

    cv2.line(panel, (18, 124), (360, 124), (70, 74, 82), 1)
    put_panel_line(panel, "SAFETY", 152, (180, 185, 195), 0.58, 2)
    put_panel_line(panel, safety.state.replace("_", " "), 190, safety_color, 0.8, 2)
    put_panel_line(panel, f"Tool temperature: {safety.tool_temperature_c:.1f} C", 222)

    trend_text = "Trend: collecting data"
    if safety.trend_c_per_min is not None:
        trend_text = f"Trend: {safety.trend_c_per_min:+.2f} C/min"
    put_panel_line(panel, trend_text, 248)
    put_panel_line(panel, f"Unoccupied: {format_duration(safety.unoccupied_seconds)}", 274)

    cv2.line(panel, (18, 296), (360, 296), (70, 74, 82), 1)
    put_panel_line(panel, "DETECTION METRICS", 326, (180, 185, 195), 0.58, 2)
    put_panel_line(panel, f"Ambient: {metrics.ambient_c:.1f} C", 356)
    put_panel_line(panel, f"Human threshold: {metrics.human_threshold_c:.1f} C", 382)
    put_panel_line(
        panel,
        f"Human component: {metrics.human_component_pixels} px "
        f"({metrics.human_component_fraction * 100:.1f}%)",
        408,
        scale=0.47,
    )
    put_panel_line(panel, f"Human detected: {'YES' if metrics.human_detected else 'NO'}", 434)
    put_panel_line(panel, f"Tool P95: {metrics.tool_p95_c:.1f} C", 460)

    if height >= 520:
        cv2.line(panel, (18, 482), (360, 482), (70, 74, 82), 1)
        put_panel_line(
            panel,
            f"Enter {config.occupied_confirm_seconds:.0f}s | Leave {config.leave_confirm_seconds:.0f}s",
            510,
            (155, 160, 170),
            0.45,
        )
        put_panel_line(panel, "Press q to quit", 536, (155, 160, 170), 0.45)

    return panel


def write_status_file(path, occupancy, safety, metrics):
    payload = {
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        "occupancy": asdict(occupancy),
        "safety": asdict(safety),
        "metrics": asdict(metrics),
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary_path.replace(path)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Monitor workstation occupancy and unattended thermal safety."
    )
    parser.add_argument("--device", default="/dev/spidev0.0", help="Lepton SPI device.")
    parser.add_argument("--scale", type=int, default=8, help="Thermal preview scale factor.")
    parser.add_argument(
        "--status-file",
        type=Path,
        default=Path("data/runtime/status.json"),
        help="JSON file updated with current states and metrics.",
    )
    parser.add_argument("--human-delta", type=float, default=4.0)
    parser.add_argument("--human-floor", type=float, default=27.0)
    parser.add_argument("--human-component-fraction", type=float, default=0.025)
    parser.add_argument("--human-min-component-pixels", type=int, default=20)
    parser.add_argument("--occupied-confirm", type=float, default=5.0)
    parser.add_argument("--leave-confirm", type=float, default=15.0)
    parser.add_argument("--recently-used-minutes", type=float, default=15.0)
    parser.add_argument("--tool-safe", type=float, default=38.0)
    parser.add_argument("--tool-alert", type=float, default=45.0)
    parser.add_argument("--cooling-slope", type=float, default=-0.5)
    parser.add_argument("--trend-min-seconds", type=float, default=45.0)
    parser.add_argument("--trend-window-seconds", type=float, default=180.0)
    parser.add_argument("--unattended-delay-seconds", type=float, default=180.0)
    parser.add_argument("--log-interval", type=float, default=1.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if "Human Area" not in DEFAULT_ROIS or "Tool Area" not in DEFAULT_ROIS:
        raise KeyError("DEFAULT_ROIS must contain 'Human Area' and 'Tool Area'")

    config = DetectionConfig(
        human_delta_c=args.human_delta,
        human_floor_c=args.human_floor,
        human_component_fraction=args.human_component_fraction,
        human_min_component_pixels=args.human_min_component_pixels,
        occupied_confirm_seconds=args.occupied_confirm,
        leave_confirm_seconds=args.leave_confirm,
        recently_used_seconds=args.recently_used_minutes * 60.0,
        tool_safe_c=args.tool_safe,
        tool_alert_c=args.tool_alert,
        cooling_slope_c_per_min=args.cooling_slope,
        trend_min_seconds=args.trend_min_seconds,
        trend_window_seconds=args.trend_window_seconds,
        unattended_delay_seconds=args.unattended_delay_seconds,
    )

    start_time = time.monotonic()
    occupancy_machine = OccupancyStateMachine(config, now=start_time)
    safety_machine = SafetyStateMachine(config, now=start_time)
    last_log_time = 0.0

    print("Starting workstation monitor.")
    print(f"Human ROI: {DEFAULT_ROIS['Human Area']}")
    print(f"Tool ROI: {DEFAULT_ROIS['Tool Area']}")
    print("Press q in the OpenCV window to quit.")

    with Lepton(args.device) as lepton:
        while True:
            frame, _ = lepton.capture()
            frame = np.squeeze(frame).astype(np.uint16)
            temp_c = raw_to_celsius(frame)
            now = time.monotonic()

            metrics = analyse_frame(
                temp_c,
                human_roi=DEFAULT_ROIS["Human Area"],
                tool_roi=DEFAULT_ROIS["Tool Area"],
                config=config,
            )
            occupancy = occupancy_machine.update(metrics.human_detected, now)
            safety = safety_machine.update(
                metrics.tool_hot_mean_c,
                occupied=occupancy.state == OCCUPANCY_OCCUPIED,
                now=now,
            )

            heatmap = make_heatmap(frame, args.scale)
            draw_roi(
                heatmap,
                "Tool Area",
                DEFAULT_ROIS["Tool Area"],
                args.scale,
                SAFETY_COLORS[safety.state],
            )
            draw_roi(
                heatmap,
                "Human Area",
                DEFAULT_ROIS["Human Area"],
                args.scale,
                OCCUPANCY_COLORS[occupancy.state],
            )
            panel = make_status_panel(heatmap.shape[0], occupancy, safety, metrics, config)
            display = np.hstack((heatmap, panel))
            cv2.imshow("Workstation Occupancy and Safety Monitor", display)

            write_status_file(args.status_file, occupancy, safety, metrics)

            if occupancy.changed or safety.changed:
                print(
                    f"STATE occupancy={occupancy.state} safety={safety.state} "
                    f"human={metrics.human_detected} tool={safety.tool_temperature_c:.1f}C"
                )

            if now - last_log_time >= args.log_interval:
                trend = (
                    "collecting"
                    if safety.trend_c_per_min is None
                    else f"{safety.trend_c_per_min:+.2f}C/min"
                )
                print(
                    f"occupancy={occupancy.state} safety={safety.state} "
                    f"ambient={metrics.ambient_c:.1f}C "
                    f"human_component={metrics.human_component_pixels}px "
                    f"tool={safety.tool_temperature_c:.1f}C trend={trend}"
                )
                last_log_time = now

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
