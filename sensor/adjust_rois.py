#!/usr/bin/env python3

import argparse
import ast
from pathlib import Path
import re

import cv2
import numpy as np
from pylepton import Lepton


FRAME_WIDTH = 80
FRAME_HEIGHT = 60
MIN_SIZE = 2
DEFAULT_ROIS = {
    "Tool Area": (0, 20, 28, 25),
    "Human Area": (38, 4, 42, 55),
}

LEFT_KEYS = {81, 2424832, 65361, 63234}
UP_KEYS = {82, 2490368, 65362, 63232}
RIGHT_KEYS = {83, 2555904, 65363, 63235}
DOWN_KEYS = {84, 2621440, 65364, 63233}


def make_display(frame, scale):
    normalized = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX)
    normalized = np.uint8(normalized)
    color = cv2.applyColorMap(normalized, cv2.COLORMAP_INFERNO)
    return cv2.resize(color, (FRAME_WIDTH * scale, FRAME_HEIGHT * scale), interpolation=cv2.INTER_NEAREST)


def clamp_roi(roi):
    x, y, width, height = roi
    width = max(MIN_SIZE, min(width, FRAME_WIDTH))
    height = max(MIN_SIZE, min(height, FRAME_HEIGHT))
    x = max(0, min(x, FRAME_WIDTH - width))
    y = max(0, min(y, FRAME_HEIGHT - height))
    return (x, y, width, height)


def move_roi(roi, dx, dy):
    x, y, width, height = roi
    return clamp_roi((x + dx, y + dy, width, height))


def resize_roi(roi, dw, dh):
    x, y, width, height = roi
    center_x = x + width / 2
    center_y = y + height / 2
    width = max(MIN_SIZE, width + dw)
    height = max(MIN_SIZE, height + dh)
    x = round(center_x - width / 2)
    y = round(center_y - height / 2)
    return clamp_roi((x, y, width, height))


def load_rois(viewer_file):
    if not viewer_file.exists():
        return dict(DEFAULT_ROIS)

    source = viewer_file.read_text(encoding="utf-8")
    match = re.search(r"DEFAULT_ROIS\s*=\s*(\{.*?\n\})", source, re.DOTALL)
    if not match:
        return dict(DEFAULT_ROIS)

    try:
        rois = ast.literal_eval(match.group(1))
    except (SyntaxError, ValueError):
        return dict(DEFAULT_ROIS)

    if not isinstance(rois, dict) or len(rois) < 1:
        return dict(DEFAULT_ROIS)

    return {str(name): clamp_roi(tuple(value)) for name, value in rois.items()}


def format_rois(rois):
    lines = ["DEFAULT_ROIS = {"]
    for name, roi in rois.items():
        lines.append(f'    "{name}": {tuple(roi)},')
    lines.append("}")
    return "\n".join(lines)


def save_rois(viewer_file, rois):
    source = viewer_file.read_text(encoding="utf-8")
    updated, count = re.subn(
        r"DEFAULT_ROIS\s*=\s*\{.*?\n\}",
        format_rois(rois),
        source,
        count=1,
        flags=re.DOTALL,
    )
    if count != 1:
        raise RuntimeError(f"Could not update DEFAULT_ROIS in {viewer_file}")
    viewer_file.write_text(updated, encoding="utf-8")


def draw_rois(display, rois, selected_name, scale):
    for name, roi in rois.items():
        x, y, width, height = roi
        sx = x * scale
        sy = y * scale
        sw = width * scale
        sh = height * scale
        selected = name == selected_name
        color = (0, 255, 120) if selected else (255, 255, 255)
        thickness = 3 if selected else 2

        cv2.rectangle(display, (sx, sy), (sx + sw, sy + sh), color, thickness)
        cv2.putText(
            display,
            f"{name}: {x},{y},{width},{height}",
            (sx + 6, max(22, sy + 22)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )


def draw_help(display, selected_name):
    cv2.rectangle(display, (0, 0), (display.shape[1], 116), (0, 0, 0), -1)
    lines = [
        f"Selected: {selected_name}",
        "1/2 select ROI | arrows or WASD move | +/- scale",
        "J/L width -/+ | K/I height -/+ | Enter save to thermal_roi_viewer.py | Q quit",
    ]
    for index, line in enumerate(lines):
        cv2.putText(
            display,
            line,
            (12, 28 + index * 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )


def handle_key(key, rois, selected_index, names):
    if key < 0:
        return selected_index, False, False

    selected_name = names[selected_index]
    roi = rois[selected_name]
    should_save = key in (10, 13)
    should_quit = key in (ord("q"), ord("Q"), 27)

    if should_save or should_quit:
        return selected_index, should_save, should_quit

    if key == ord("1") and len(names) >= 1:
        selected_index = 0
    elif key == ord("2") and len(names) >= 2:
        selected_index = 1
    elif key == ord("\t"):
        selected_index = (selected_index + 1) % len(names)
    elif key in LEFT_KEYS or key in (ord("a"), ord("A")):
        rois[selected_name] = move_roi(roi, -1, 0)
    elif key in RIGHT_KEYS or key in (ord("d"), ord("D")):
        rois[selected_name] = move_roi(roi, 1, 0)
    elif key in UP_KEYS or key in (ord("w"), ord("W")):
        rois[selected_name] = move_roi(roi, 0, -1)
    elif key in DOWN_KEYS or key in (ord("s"), ord("S")):
        rois[selected_name] = move_roi(roi, 0, 1)
    elif key in (ord("+"), ord("=")):
        rois[selected_name] = resize_roi(roi, 2, 2)
    elif key in (ord("-"), ord("_")):
        rois[selected_name] = resize_roi(roi, -2, -2)
    elif key in (ord("j"), ord("J")):
        rois[selected_name] = resize_roi(roi, -2, 0)
    elif key in (ord("l"), ord("L")):
        rois[selected_name] = resize_roi(roi, 2, 0)
    elif key in (ord("k"), ord("K")):
        rois[selected_name] = resize_roi(roi, 0, -2)
    elif key in (ord("i"), ord("I")):
        rois[selected_name] = resize_roi(roi, 0, 2)

    return selected_index, False, False


def parse_args():
    parser = argparse.ArgumentParser(description="Interactively adjust ROI boxes for the Lepton viewer.")
    parser.add_argument("--device", default="/dev/spidev0.0", help="SPI device path for the Lepton module.")
    parser.add_argument("--scale", type=int, default=10, help="Display scale factor.")
    parser.add_argument(
        "--viewer-file",
        type=Path,
        default=Path(__file__).with_name("thermal_roi_viewer.py"),
        help="thermal_roi_viewer.py file to update when Enter is pressed.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    rois = load_rois(args.viewer_file)
    names = list(rois.keys())
    selected_index = 0

    print("Starting ROI adjuster.")
    print(f"Viewer file: {args.viewer_file}")
    print("Controls: 1/2 select, arrows/WASD move, +/- scale, J/L width, K/I height.")
    print("Press Enter in the OpenCV window to save and exit. Press q to quit without saving.")

    with Lepton(args.device) as lepton:
        while True:
            frame, _ = lepton.capture()
            frame = np.squeeze(frame).astype(np.uint16)
            display = make_display(frame, args.scale)
            selected_name = names[selected_index]

            draw_rois(display, rois, selected_name, args.scale)
            draw_help(display, selected_name)
            cv2.imshow("ROI Adjuster", display)

            key = cv2.waitKeyEx(1)
            selected_index, should_save, should_quit = handle_key(key, rois, selected_index, names)

            if should_save:
                save_rois(args.viewer_file, rois)
                print("Saved ROI positions:")
                for name, roi in rois.items():
                    print(f"  {name}: {roi}")
                break

            if should_quit:
                print("Quit without saving.")
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
