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
    ord("r"): "residual",
    ord("p"): "passing",
}


def raw_to_celsius(frame):
    return (frame.astype(np.float32) * 0.01) - 273.15


def make_preview(frame, label, frame_index, scale):
    normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
    normalized = np.uint8(normalized)
    color = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
    preview = cv2.resize(color, (80 * scale, 60 * scale), interpolation=cv2.INTER_NEAREST)

    temp_c = raw_to_celsius(frame)
    max_temp = float(np.max(temp_c))
    min_temp = float(np.min(temp_c))

    cv2.rectangle(preview, (0, 0), (preview.shape[1], 72), (0, 0, 0), -1)
    cv2.putText(
        preview,
        f"Label: {label}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        preview,
        f"Frame {frame_index:06d}   min {min_temp:.1f}C   max {max_temp:.1f}C",
        (12, 56),
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
        choices=["free", "occupied", "residual", "passing"],
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
    session_dir, frames_dir, previews_dir = create_session_dir(args.output_dir, args.session)
    labels_path = session_dir / "labels.csv"
    metadata_path = session_dir / "metadata.txt"
    current_label = args.label
    frame_interval = 1.0 / max(args.fps, 0.1)
    frame_index = 0

    metadata_path.write_text(
        "\n".join(
            [
                f"created_at={datetime.now().isoformat(timespec='seconds')}",
                f"device={args.device}",
                f"fps={args.fps}",
                "sensor=FLIR Lepton 2.5 radiometric",
                "frame_shape=60x80",
                "labels=free,occupied,residual,passing",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print("Starting thermal dataset collection.")
    print(f"Session: {session_dir}")
    print("Keys: f=free, o=occupied, r=residual, p=passing, q=quit")

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
            next_capture_time = 0.0

            while True:
                now = time.time()
                if now < next_capture_time:
                    time.sleep(min(0.02, next_capture_time - now))
                    key = cv2.waitKey(1) & 0xFF
                    if key in LABEL_KEYS:
                        current_label = LABEL_KEYS[key]
                        print(f"Label changed to {current_label}")
                    elif key == ord("q"):
                        break
                    continue

                frame, _ = lepton.capture()
                frame = np.squeeze(frame).astype(np.uint16)
                temp_c = raw_to_celsius(frame)

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

                preview = make_preview(frame, current_label, frame_index, args.scale)
                cv2.imshow("Thermal Dataset Collector", preview)

                if args.preview_every > 0 and frame_index % args.preview_every == 0:
                    cv2.imwrite(str(previews_dir / f"preview_{frame_index:06d}.png"), preview)

                print(
                    f"{frame_index:06d} {current_label:9s} "
                    f"min={float(np.min(temp_c)):.1f}C "
                    f"max={float(np.max(temp_c)):.1f}C "
                    f"mean={float(np.mean(temp_c)):.1f}C"
                )

                frame_index += 1
                next_capture_time = time.time() + frame_interval

                key = cv2.waitKey(1) & 0xFF
                if key in LABEL_KEYS:
                    current_label = LABEL_KEYS[key]
                    print(f"Label changed to {current_label}")
                elif key == ord("q"):
                    break

    cv2.destroyAllWindows()
    print(f"Saved {frame_index} frames to {session_dir}")


if __name__ == "__main__":
    main()
