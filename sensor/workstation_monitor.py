#!/usr/bin/env python3

import argparse
from dataclasses import asdict
from datetime import datetime, timedelta
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
INPUT_DIM = FRAME_WIDTH * FRAME_HEIGHT
OCCUPANCY_MODEL_LABELS = ["not_occupied", "occupied"]

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


def relu(x):
    return np.maximum(x, 0.0)


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def load_occupancy_model(model_path):
    data = np.load(model_path)
    labels = [str(label) for label in data["labels"]]
    if labels != OCCUPANCY_MODEL_LABELS:
        raise ValueError(
            f"{model_path} is not a binary occupancy model. "
            f"Expected labels {OCCUPANCY_MODEL_LABELS}, got {labels}."
        )

    return {
        "path": str(model_path),
        "labels": labels,
        "w1": data["w1"].astype(np.float32),
        "b1": data["b1"].astype(np.float32),
        "w2": data["w2"].astype(np.float32),
        "b2": data["b2"].astype(np.float32),
        "mean": data["mean"].astype(np.float32),
        "std": data["std"].astype(np.float32),
    }


def predict_occupancy(model, temp_c):
    x = temp_c.astype(np.float32).reshape(1, INPUT_DIM)
    x = (x - model["mean"]) / model["std"]
    hidden = relu(x @ model["w1"] + model["b1"])
    logits = hidden @ model["w2"] + model["b2"]
    probs = softmax(logits)[0]
    occupied_index = model["labels"].index("occupied")
    occupied_probability = float(probs[occupied_index])
    predicted_label = model["labels"][int(np.argmax(probs))]
    return predicted_label, occupied_probability


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


def make_status_panel(height, occupancy, safety, metrics, config, model_status):
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
    put_panel_line(panel, f"ROI human: {'YES' if metrics.human_detected else 'NO'}", 434)
    if model_status:
        put_panel_line(
            panel,
            f"ML occupied: {model_status['occupied_probability'] * 100:.1f}%",
            460,
        )
    else:
        put_panel_line(panel, f"Tool P95: {metrics.tool_p95_c:.1f} C", 460)

    if height >= 520:
        cv2.line(panel, (18, 482), (360, 482), (70, 74, 82), 1)
        if model_status:
            put_panel_line(panel, f"Tool P95: {metrics.tool_p95_c:.1f} C", 510, (155, 160, 170), 0.45)
            put_panel_line(panel, "Human source: deep learning", 536, (155, 160, 170), 0.45)
        else:
            put_panel_line(panel, "Human source: ROI rules", 510, (155, 160, 170), 0.45)
            put_panel_line(panel, "Press q to quit", 536, (155, 160, 170), 0.45)

    if height >= 580:
        put_panel_line(
            panel,
            f"Enter {config.occupied_confirm_seconds:.0f}s | Leave {config.leave_confirm_seconds:.0f}s",
            562,
            (155, 160, 170),
            0.45,
        )
        put_panel_line(panel, "Press q to quit", 588, (155, 160, 170), 0.45)

    return panel


def write_snapshot(path, image):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
    if not cv2.imwrite(str(temporary_path), image):
        raise RuntimeError(f"Failed to write thermal snapshot: {temporary_path}")
    temporary_path.replace(path)


def write_status_file(path, occupancy, safety, metrics, model_status, snapshot_status):
    timestamp = datetime.now()
    occupancy_payload = asdict(occupancy)
    safety_payload = asdict(safety)
    occupancy_payload["changed_at"] = (
        timestamp - timedelta(seconds=occupancy.state_seconds)
    ).isoformat(timespec="milliseconds")
    safety_payload["changed_at"] = (
        timestamp - timedelta(seconds=safety.state_seconds)
    ).isoformat(timespec="milliseconds")

    payload = {
        "timestamp": timestamp.isoformat(timespec="milliseconds"),
        "occupancy": occupancy_payload,
        "safety": safety_payload,
        "metrics": asdict(metrics),
        "model": model_status,
        "snapshot": snapshot_status,
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
        "--occupancy-model",
        type=Path,
        default=None,
        help="Optional binary occupancy model.npz. If omitted, human detection uses ROI rules.",
    )
    parser.add_argument(
        "--model-threshold",
        type=float,
        default=0.5,
        help="Occupied probability threshold when --occupancy-model is used.",
    )
    parser.add_argument(
        "--status-file",
        type=Path,
        default=Path("data/runtime/status.json"),
        help="JSON file updated with current states and metrics.",
    )
    parser.add_argument(
        "--snapshot-file",
        type=Path,
        default=Path("data/runtime/thermal_view.jpg"),
        help="Thermal preview image written for the web dashboard.",
    )
    parser.add_argument(
        "--snapshot-interval",
        type=float,
        default=30.0,
        help="Seconds between dashboard thermal preview image updates.",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Run without an OpenCV preview window. Useful for background dashboard service mode.",
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
    parser.add_argument("--cooling-min-drop", type=float, default=2.0)
    parser.add_argument("--trend-min-seconds", type=float, default=45.0)
    parser.add_argument("--trend-window-seconds", type=float, default=180.0)
    parser.add_argument("--unattended-delay-seconds", type=float, default=180.0)
    parser.add_argument("--safe-confirm-seconds", type=float, default=60.0)
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
        cooling_min_drop_c=args.cooling_min_drop,
        trend_min_seconds=args.trend_min_seconds,
        trend_window_seconds=args.trend_window_seconds,
        unattended_delay_seconds=args.unattended_delay_seconds,
        safe_confirm_seconds=args.safe_confirm_seconds,
    )

    start_time = time.monotonic()
    occupancy_machine = OccupancyStateMachine(config, now=start_time)
    safety_machine = SafetyStateMachine(config, now=start_time)
    last_log_time = 0.0
    last_snapshot_time = 0.0
    last_snapshot_timestamp = None

    print("Starting workstation monitor.")
    print(f"Human ROI: {DEFAULT_ROIS['Human Area']}")
    print(f"Tool ROI: {DEFAULT_ROIS['Tool Area']}")
    occupancy_model = None
    if args.occupancy_model:
        occupancy_model = load_occupancy_model(args.occupancy_model)
        print(f"Occupancy model: {args.occupancy_model}")
        print(f"Model threshold: {args.model_threshold:.2f}")
    else:
        print("Occupancy model: disabled; using ROI human detection.")
    print(f"Dashboard snapshot: {args.snapshot_file} every {args.snapshot_interval:.0f}s")
    if args.no_window:
        print("Running without OpenCV preview window. Press Ctrl+C to quit.")
    else:
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
            model_status = None
            human_detected = metrics.human_detected
            if occupancy_model:
                predicted_label, occupied_probability = predict_occupancy(occupancy_model, temp_c)
                human_detected = occupied_probability >= args.model_threshold
                model_status = {
                    "type": "binary_occupancy_mlp",
                    "path": occupancy_model["path"],
                    "predicted_label": predicted_label,
                    "occupied_probability": occupied_probability,
                    "threshold": args.model_threshold,
                    "occupied": human_detected,
                }

            occupancy = occupancy_machine.update(human_detected, now)
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
            if now - last_snapshot_time >= args.snapshot_interval:
                write_snapshot(args.snapshot_file, heatmap)
                last_snapshot_time = now
                last_snapshot_timestamp = datetime.now().isoformat(timespec="milliseconds")

            panel = make_status_panel(heatmap.shape[0], occupancy, safety, metrics, config, model_status)
            display = np.hstack((heatmap, panel))
            snapshot_status = {
                "path": str(args.snapshot_file),
                "url": "../data/runtime/thermal_view.jpg",
                "updated_at": last_snapshot_timestamp,
                "interval_seconds": args.snapshot_interval,
            }
            write_status_file(args.status_file, occupancy, safety, metrics, model_status, snapshot_status)

            if occupancy.changed or safety.changed:
                print(
                    f"STATE occupancy={occupancy.state} safety={safety.state} "
                    f"human={human_detected} tool={safety.tool_temperature_c:.1f}C"
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
                    f"human={human_detected} "
                    f"tool={safety.tool_temperature_c:.1f}C trend={trend}"
                )
                last_log_time = now

            if not args.no_window:
                cv2.imshow("Workstation Occupancy and Safety Monitor", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    if not args.no_window:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
