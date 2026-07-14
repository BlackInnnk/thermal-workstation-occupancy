#!/usr/bin/env python3

import argparse
import time

import cv2
import numpy as np
from pylepton import Lepton


# ROI format: x, y, width, height in the original 80x60 Lepton frame.
# Adjust these values after checking the real camera mounting position.
DEFAULT_ROIS = {
    "Tool Area": (0, 25, 14, 9),
    "Human Area": (38, 4, 42, 55),
}


def raw_to_celsius(frame):
    """Convert Lepton radiometric values to degrees Celsius."""
    return (frame.astype(np.float32) * 0.01) - 273.15


def make_display(frame, scale):
    normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
    normalized = np.uint8(normalized)
    color = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
    return cv2.resize(color, (80 * scale, 60 * scale), interpolation=cv2.INTER_NEAREST)


def analyse_roi(temp_c, roi, threshold_c):
    x, y, width, height = roi
    region = temp_c[y : y + height, x : x + width]
    hot_mask = region > threshold_c

    return {
        "max": float(np.max(region)),
        "mean": float(np.mean(region)),
        "hot_area": int(np.sum(hot_mask)),
    }


def draw_roi(display, name, roi, stats, scale, threshold_c):
    x, y, width, height = roi
    sx = x * scale
    sy = y * scale
    sw = width * scale
    sh = height * scale

    cv2.rectangle(display, (sx, sy), (sx + sw, sy + sh), (255, 255, 255), 2)

    label = f"{name}  max {stats['max']:.1f}C  hot {stats['hot_area']}"
    cv2.putText(
        display,
        label,
        (sx + 6, sy + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    threshold_label = f">{threshold_c:.1f}C"
    cv2.putText(
        display,
        threshold_label,
        (sx + 6, sy + sh - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Display Lepton thermal frames with two workstation ROI overlays."
    )
    parser.add_argument(
        "--device",
        default="/dev/spidev0.0",
        help="SPI device path for the Lepton module.",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=10,
        help="Display scale factor for the 80x60 thermal frame.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=30.0,
        help="Temperature threshold in Celsius used for hot-area counting.",
    )
    parser.add_argument(
        "--log-interval",
        type=float,
        default=1.0,
        help="Seconds between terminal statistics updates.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.scale < 1:
        raise ValueError("--scale must be at least 1")
    if args.log_interval <= 0:
        raise ValueError("--log-interval must be positive")
    if not np.isfinite(args.threshold):
        raise ValueError("--threshold must be finite")
    rois = DEFAULT_ROIS
    last_log_time = 0.0

    print("Starting thermal ROI viewer.")
    print(f"Device: {args.device}")
    print(f"Hot-area threshold: {args.threshold:.1f}C")
    print("Press q in the OpenCV window to quit.")

    with Lepton(args.device) as lepton:
        while True:
            try:
                frame, _ = lepton.capture()
                frame = np.squeeze(frame)
            except (AttributeError, OSError, TypeError, ValueError) as exc:
                print(f"Thermal capture failed; retrying: {exc}")
                time.sleep(0.2)
                continue
            if frame.shape != (60, 80):
                print(f"Ignoring frame with shape {frame.shape}; expected (60, 80).")
                continue

            temp_c = raw_to_celsius(frame)
            display = make_display(frame, args.scale)

            stats_by_roi = {}
            for name, roi in rois.items():
                stats = analyse_roi(temp_c, roi, args.threshold)
                stats_by_roi[name] = stats
                draw_roi(display, name, roi, stats, args.scale, args.threshold)

            now = time.time()
            if now - last_log_time >= args.log_interval:
                parts = []
                for name, stats in stats_by_roi.items():
                    parts.append(
                        f"{name}: max={stats['max']:.1f}C "
                        f"mean={stats['mean']:.1f}C "
                        f"hot_area={stats['hot_area']}"
                    )
                print(" | ".join(parts))
                last_log_time = now

            cv2.imshow("Thermal ROI Viewer", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
