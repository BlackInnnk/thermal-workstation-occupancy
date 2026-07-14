#!/usr/bin/env python3

import argparse
import csv
from datetime import datetime
from pathlib import Path
import time

import cv2
import numpy as np
from pylepton import Lepton


LABEL_KEYS = {
    ord("f"): "free",
    ord("o"): "occupied",
    ord("c"): "cooling",
    ord("h"): "hot_empty",
}

LABELS = ["free", "occupied", "cooling", "hot_empty"]


def raw_to_celsius(frame):
    return (frame.astype(np.float32) * 0.01) - 273.15


def make_preview(frame, label, frame_index, scale, is_recording):
    normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
    normalized = np.uint8(normalized)
    color = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
    preview = cv2.resize(color, (80 * scale, 60 * scale), interpolation=cv2.INTER_NEAREST)

    temp_c = raw_to_celsius(frame)
    max_temp = float(np.max(temp_c))
    min_temp = float(np.min(temp_c))

    status = "RECORDING" if is_recording else "PAUSED"
    status_color = (0, 80, 255) if is_recording else (180, 180, 180)

    cv2.rectangle(preview, (0, 0), (preview.shape[1], 96), (0, 0, 0), -1)
    cv2.putText(
        preview,
        status,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        status_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        preview,
        f"Label: {label}",
        (12, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        preview,
        f"Saved {frame_index:06d}   min {min_temp:.1f}C   max {max_temp:.1f}C",
        (12, 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    return preview


def create_session_dir(output_dir, session_name):
    if session_name is None:
        session_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    session_dir = output_dir / session_name
    frames_dir = session_dir / "frames"
    previews_dir = session_dir / "previews"

    if session_dir.exists():
        raise FileExistsError(
            f"Dataset session already exists: {session_dir}. "
            "Choose a new --session name or remove the empty session first."
        )
    frames_dir.mkdir(parents=True, exist_ok=False)
    previews_dir.mkdir(parents=True, exist_ok=True)

    return session_dir, frames_dir, previews_dir


def parse_args():
    parser = argparse.ArgumentParser(
        description="Collect labelled FLIR Lepton thermal frames for occupancy detection."
    )
    parser.add_argument(
        "--device",
        default="/dev/spidev0.0",
        help="SPI device path for the Lepton module.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/raw",
        type=Path,
        help="Directory where dataset sessions will be saved.",
    )
    parser.add_argument(
        "--session",
        default=None,
        help="Optional session folder name. Defaults to current timestamp.",
    )
    parser.add_argument(
        "--label",
        default="free",
        choices=LABELS,
        help="Initial label for saved frames.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Approximate saved frame rate.",
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=10,
        help="OpenCV preview scale factor for the 80x60 thermal frame.",
    )
    parser.add_argument(
        "--preview-every",
        type=int,
        default=20,
        help="Save one preview PNG every N saved frames. Use 0 to disable.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    if args.scale < 1:
        raise ValueError("--scale must be at least 1")
    if args.preview_every < 0:
        raise ValueError("--preview-every cannot be negative")
    if args.session is not None and Path(args.session).name != args.session:
        raise ValueError("--session must be a single folder name, not a path")

    # Fail before creating a dataset folder when no desktop/VNC display is available.
    cv2.namedWindow("Thermal Dataset Collector", cv2.WINDOW_NORMAL)
    try:
        session_dir, frames_dir, previews_dir = create_session_dir(args.output_dir, args.session)
    except FileExistsError as exc:
        raise SystemExit(str(exc)) from None
    labels_path = session_dir / "labels.csv"
    metadata_path = session_dir / "metadata.txt"
    current_label = args.label
    frame_interval = 1.0 / args.fps
    frame_index = 0

    metadata_path.write_text(
        "\n".join(
            [
                f"created_at={datetime.now().isoformat(timespec='seconds')}",
                f"device={args.device}",
                f"fps={args.fps}",
                "sensor=FLIR Lepton 2.5 radiometric",
                "frame_shape=60x80",
                "labels=free,occupied,cooling,hot_empty",
                "controls=enter toggles recording, f/o/c/h changes label, q quits",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print("Starting thermal dataset collection.")
    print(f"Session: {session_dir}")
    print("State: PAUSED. Press Enter in the preview window to start recording.")
    print(
        "Keys: Enter=start/stop recording, f=free, o=occupied, "
        "c=cooling, h=hot_empty, q=quit"
    )

    with labels_path.open("w", newline="", encoding="utf-8") as labels_file:
        writer = csv.DictWriter(
            labels_file,
            fieldnames=[
                "timestamp",
                "frame_index",
                "filename",
                "label",
                "min_c",
                "max_c",
                "mean_c",
            ],
        )
        writer.writeheader()

        with Lepton(args.device) as lepton:
            is_recording = False
            next_save_time = 0.0

            while True:
                try:
                    frame, _ = lepton.capture()
                    frame = np.squeeze(frame).astype(np.uint16)
                except (AttributeError, OSError, TypeError, ValueError) as exc:
                    print(f"Thermal capture failed; retrying: {exc}")
                    time.sleep(0.2)
                    continue
                if frame.shape != (60, 80):
                    print(f"Ignoring frame with shape {frame.shape}; expected (60, 80).")
                    continue
                if np.all(frame == frame.flat[0]) or np.mean((frame == 0) | (frame == 65535)) > 0.05:
                    print("Ignoring invalid thermal frame.")
                    continue
                temp_c = raw_to_celsius(frame)

                preview = make_preview(frame, current_label, frame_index, args.scale, is_recording)
                cv2.imshow("Thermal Dataset Collector", preview)

                now = time.monotonic()
                if is_recording and now >= next_save_time:
                    filename = f"frame_{frame_index:06d}.npy"
                    frame_path = frames_dir / filename
                    np.save(frame_path, frame)

                    timestamp = datetime.now().isoformat(timespec="milliseconds")
                    writer.writerow(
                        {
                            "timestamp": timestamp,
                            "frame_index": frame_index,
                            "filename": f"frames/{filename}",
                            "label": current_label,
                            "min_c": f"{float(np.min(temp_c)):.3f}",
                            "max_c": f"{float(np.max(temp_c)):.3f}",
                            "mean_c": f"{float(np.mean(temp_c)):.3f}",
                        }
                    )
                    labels_file.flush()

                    if args.preview_every > 0 and frame_index % args.preview_every == 0:
                        cv2.imwrite(str(previews_dir / f"preview_{frame_index:06d}.png"), preview)

                    print(
                        f"{frame_index:06d} {current_label:10s} "
                        f"min={float(np.min(temp_c)):.1f}C "
                        f"max={float(np.max(temp_c)):.1f}C "
                        f"mean={float(np.mean(temp_c)):.1f}C"
                    )

                    frame_index += 1
                    next_save_time = now + frame_interval

                key = cv2.waitKey(1) & 0xFF
                if key in LABEL_KEYS:
                    current_label = LABEL_KEYS[key]
                    print(f"Label changed to {current_label}")
                elif key in (10, 13):
                    is_recording = not is_recording
                    state = "RECORDING" if is_recording else "PAUSED"
                    print(f"State changed to {state} with label {current_label}")
                    next_save_time = 0.0
                elif key == ord("q"):
                    break

    cv2.destroyAllWindows()
    print(f"Saved {frame_index} frames to {session_dir}")


if __name__ == "__main__":
    main()
