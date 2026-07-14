#!/usr/bin/env python3

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - handled at runtime
    Image = None
    ImageDraw = None
    ImageFont = None


LABEL_ORDER = ["free", "occupied", "cooling", "hot_empty"]

LABEL_COLORS = {
    "free": (28, 132, 128),
    "occupied": (212, 92, 121),
    "cooling": (56, 110, 190),
    "hot_empty": (210, 82, 47),
}

NEUTRAL = {
    "ink": (31, 35, 40),
    "muted": (105, 113, 122),
    "grid": (220, 225, 230),
    "background": (250, 251, 252),
    "panel": (255, 255, 255),
}


def raw_to_celsius(frame):
    return (frame.astype(np.float32) * 0.01) - 273.15


def load_font(size, bold=False):
    if ImageFont is None:
        return None

    candidates = []
    if bold:
        candidates.extend(
            [
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
                "/Library/Fonts/Arial Bold.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            ]
        )
    candidates.extend(
        [
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
    )

    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def text_width(draw, text, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]


def latest_session(raw_dir):
    sessions = [path for path in raw_dir.iterdir() if (path / "labels.csv").exists()]
    if not sessions:
        raise FileNotFoundError(f"No dataset sessions with labels.csv found in {raw_dir}")
    return max(sessions, key=lambda path: path.stat().st_mtime)


def parse_timestamp(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def read_label_rows(labels_path):
    rows = []
    with labels_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"timestamp", "frame_index", "filename", "label", "min_c", "max_c", "mean_c"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"{labels_path} is missing columns: {missing_list}")

        for row in reader:
            parsed = dict(row)
            parsed["frame_index"] = int(parsed["frame_index"])
            parsed["min_c"] = float(parsed["min_c"])
            parsed["max_c"] = float(parsed["max_c"])
            parsed["mean_c"] = float(parsed["mean_c"])
            temperatures = (parsed["min_c"], parsed["max_c"], parsed["mean_c"])
            if not all(math.isfinite(value) for value in temperatures):
                raise ValueError(
                    f"{labels_path} contains non-finite summary temperatures "
                    f"at frame {parsed['frame_index']}"
                )
            parsed["timestamp_dt"] = parse_timestamp(parsed["timestamp"])
            rows.append(parsed)
    return rows


def ordered_labels(counter):
    known = [label for label in LABEL_ORDER if counter.get(label, 0)]
    extra = sorted(label for label in counter if label not in LABEL_ORDER)
    return known + extra


def mean(values):
    return sum(values) / len(values) if values else 0.0


def percentile(values, pct):
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float32), pct))


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_count_rows(rows):
    counter = Counter(row["label"] for row in rows)
    total = sum(counter.values())
    return [
        {
            "label": label,
            "count": counter[label],
            "share_pct": f"{(counter[label] / total * 100):.2f}" if total else "0.00",
        }
        for label in ordered_labels(counter)
    ]


def build_temperature_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["label"]].append(row)

    out = []
    for label in ordered_labels(Counter(row["label"] for row in rows)):
        label_rows = grouped[label]
        mins = [row["min_c"] for row in label_rows]
        maxs = [row["max_c"] for row in label_rows]
        means = [row["mean_c"] for row in label_rows]
        out.append(
            {
                "label": label,
                "count": len(label_rows),
                "min_c_mean": f"{mean(mins):.3f}",
                "mean_c_mean": f"{mean(means):.3f}",
                "max_c_mean": f"{mean(maxs):.3f}",
                "max_c_p50": f"{percentile(maxs, 50):.3f}",
                "max_c_p90": f"{percentile(maxs, 90):.3f}",
                "max_c_p95": f"{percentile(maxs, 95):.3f}",
                "max_c_max": f"{max(maxs):.3f}",
            }
        )
    return out


def infer_duration(rows):
    timestamps = [row["timestamp_dt"] for row in rows if row["timestamp_dt"] is not None]
    if len(timestamps) < 2:
        return None
    return (max(timestamps) - min(timestamps)).total_seconds()


def validate_dataset(session_dir, rows, min_label_count):
    warnings = []
    if not rows:
        warnings.append("No labelled frames were found.")
        return warnings

    counter = Counter(row["label"] for row in rows)
    unknown = sorted(label for label in counter if label not in LABEL_ORDER)
    if unknown:
        warnings.append(f"Unknown labels found: {', '.join(unknown)}")

    missing_labels = [label for label in LABEL_ORDER if counter.get(label, 0) == 0]
    if missing_labels:
        warnings.append(f"Missing expected labels: {', '.join(missing_labels)}")

    low_labels = [
        f"{label}={counter[label]}"
        for label in LABEL_ORDER
        if 0 < counter.get(label, 0) < min_label_count
    ]
    if low_labels:
        warnings.append(
            "Low sample count for training in: "
            + ", ".join(low_labels)
            + f" (target at least {min_label_count} each)."
        )

    non_zero_counts = [count for count in counter.values() if count > 0]
    if non_zero_counts and min(non_zero_counts) > 0 and max(non_zero_counts) / min(non_zero_counts) > 3:
        warnings.append("Class imbalance is high; use class weighting or collect more minority-class data.")

    missing_files = []
    for row in rows:
        frame_path = session_dir / row["filename"]
        if not frame_path.exists():
            missing_files.append(row["filename"])
            if len(missing_files) >= 5:
                break
    if missing_files:
        warnings.append(
            f"Some frame files referenced by labels.csv are missing, for example: {', '.join(missing_files)}"
        )

    return warnings


def make_palette():
    stops = [
        (0.00, (9, 8, 31)),
        (0.22, (54, 31, 112)),
        (0.45, (143, 44, 123)),
        (0.68, (221, 73, 77)),
        (0.85, (249, 143, 60)),
        (1.00, (253, 230, 149)),
    ]
    palette = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255
        for idx in range(len(stops) - 1):
            left_t, left_rgb = stops[idx]
            right_t, right_rgb = stops[idx + 1]
            if left_t <= t <= right_t:
                local_t = (t - left_t) / max(right_t - left_t, 1e-9)
                rgb = [
                    round(left_rgb[channel] + (right_rgb[channel] - left_rgb[channel]) * local_t)
                    for channel in range(3)
                ]
                palette[i] = rgb
                break
    return palette


def thermal_image(frame, display_min, display_max, scale):
    temp = raw_to_celsius(frame)
    normalized = (temp - display_min) / max(display_max - display_min, 1e-6)
    indices = np.clip(normalized * 255, 0, 255).astype(np.uint8)
    rgb = make_palette()[indices]
    image = Image.fromarray(rgb, mode="RGB")
    return image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)


def select_samples(label_rows, sample_count):
    if not label_rows:
        return []
    if len(label_rows) <= sample_count:
        return label_rows
    positions = np.linspace(0, len(label_rows) - 1, sample_count)
    return [label_rows[int(round(pos))] for pos in positions]


def draw_label_distribution(count_rows, output_path, title):
    if Image is None:
        return False

    width = 1100
    row_height = 78
    top = 110
    left = 190
    right = 90
    height = top + row_height * len(count_rows) + 70
    image = Image.new("RGB", (width, height), NEUTRAL["background"])
    draw = ImageDraw.Draw(image)
    title_font = load_font(32, bold=True)
    label_font = load_font(22, bold=True)
    body_font = load_font(20)
    small_font = load_font(16)

    draw.text((42, 32), title, fill=NEUTRAL["ink"], font=title_font)
    draw.text((42, 72), "Label count and dataset balance", fill=NEUTRAL["muted"], font=small_font)

    max_count = max(int(row["count"]) for row in count_rows) if count_rows else 1
    bar_max = width - left - right
    total = sum(int(row["count"]) for row in count_rows)

    for idx, row in enumerate(count_rows):
        label = row["label"]
        count = int(row["count"])
        share = count / total * 100 if total else 0.0
        y = top + idx * row_height
        color = LABEL_COLORS.get(label, (120, 120, 120))
        draw.text((42, y + 15), label, fill=NEUTRAL["ink"], font=label_font)
        draw.rounded_rectangle((left, y + 11, width - right, y + 48), radius=9, fill=(236, 240, 244))
        bar_width = max(4, int(bar_max * count / max_count))
        draw.rounded_rectangle((left, y + 11, left + bar_width, y + 48), radius=9, fill=color)
        value = f"{count:,} frames  ({share:.1f}%)"
        value_width = text_width(draw, value, body_font)
        outside_x = left + bar_width + 14
        if outside_x + value_width <= width - 34:
            draw.text((outside_x, y + 17), value, fill=NEUTRAL["ink"], font=body_font)
        elif bar_width > value_width + 28:
            draw.text(
                (left + bar_width - value_width - 14, y + 17),
                value,
                fill=(255, 255, 255),
                font=body_font,
            )
        else:
            draw.text((width - right - value_width, y + 17), value, fill=NEUTRAL["ink"], font=body_font)

    image.save(output_path)
    return True


def draw_label_timeline(rows, output_path, title):
    if Image is None:
        return False

    width = 1300
    height = 300
    left = 70
    right = 50
    top = 90
    band_height = 72
    image = Image.new("RGB", (width, height), NEUTRAL["background"])
    draw = ImageDraw.Draw(image)
    title_font = load_font(32, bold=True)
    body_font = load_font(18)
    small_font = load_font(15)

    draw.text((42, 28), title, fill=NEUTRAL["ink"], font=title_font)
    draw.text((42, 67), "Sequence of labels across the recorded session", fill=NEUTRAL["muted"], font=small_font)

    plot_width = width - left - right
    n = max(len(rows), 1)
    for idx, row in enumerate(rows):
        x0 = left + int(idx / n * plot_width)
        x1 = left + int((idx + 1) / n * plot_width) + 1
        color = LABEL_COLORS.get(row["label"], (120, 120, 120))
        draw.rectangle((x0, top, x1, top + band_height), fill=color)

    draw.rectangle((left, top, left + plot_width, top + band_height), outline=NEUTRAL["ink"], width=1)
    tick_labels = [
        (0, "start"),
        (len(rows) // 2, "middle"),
        (len(rows) - 1, "end"),
    ]
    for idx, label in tick_labels:
        x = left + int(idx / n * plot_width)
        draw.line((x, top + band_height, x, top + band_height + 8), fill=NEUTRAL["ink"], width=1)
        frame = rows[idx]["frame_index"] if rows else 0
        draw.text((x - 28, top + band_height + 13), f"{label}\n#{frame}", fill=NEUTRAL["muted"], font=small_font)

    legend_x = left
    legend_y = height - 62
    for label in ordered_labels(Counter(row["label"] for row in rows)):
        color = LABEL_COLORS.get(label, (120, 120, 120))
        draw.rounded_rectangle((legend_x, legend_y, legend_x + 24, legend_y + 18), radius=4, fill=color)
        draw.text((legend_x + 32, legend_y - 2), label, fill=NEUTRAL["ink"], font=body_font)
        legend_x += 32 + text_width(draw, label, body_font) + 36

    image.save(output_path)
    return True


def draw_max_temperature_timeline(rows, output_path, title):
    if Image is None:
        return False

    width = 1300
    height = 560
    left = 92
    right = 50
    top = 95
    bottom = 92
    plot_width = width - left - right
    plot_height = height - top - bottom
    values = [row["max_c"] for row in rows]
    if not values:
        return False

    y_min = math.floor((min(values) - 2) / 5) * 5
    y_max = math.ceil((max(values) + 2) / 5) * 5
    if y_max <= y_min:
        y_max = y_min + 5

    image = Image.new("RGB", (width, height), NEUTRAL["background"])
    draw = ImageDraw.Draw(image)
    title_font = load_font(32, bold=True)
    body_font = load_font(18)
    small_font = load_font(15)

    draw.text((42, 28), title, fill=NEUTRAL["ink"], font=title_font)
    draw.text((42, 67), "Maximum frame temperature over time", fill=NEUTRAL["muted"], font=small_font)

    for tick in np.linspace(y_min, y_max, 6):
        y = top + plot_height - int((tick - y_min) / (y_max - y_min) * plot_height)
        draw.line((left, y, left + plot_width, y), fill=NEUTRAL["grid"], width=1)
        draw.text((35, y - 10), f"{tick:.0f}C", fill=NEUTRAL["muted"], font=small_font)

    n = max(len(values), 1)
    points = []
    for idx, value in enumerate(values):
        x = left + int(idx / max(n - 1, 1) * plot_width)
        y = top + plot_height - int((value - y_min) / (y_max - y_min) * plot_height)
        points.append((x, y))

    if len(points) >= 2:
        draw.line(points, fill=(43, 98, 156), width=3)
    for point in points[:: max(len(points) // 90, 1)]:
        draw.ellipse((point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2), fill=(43, 98, 156))

    draw.line((left, top, left, top + plot_height), fill=NEUTRAL["ink"], width=1)
    draw.line((left, top + plot_height, left + plot_width, top + plot_height), fill=NEUTRAL["ink"], width=1)

    band_y = height - 54
    for idx, row in enumerate(rows):
        x0 = left + int(idx / n * plot_width)
        x1 = left + int((idx + 1) / n * plot_width) + 1
        color = LABEL_COLORS.get(row["label"], (120, 120, 120))
        draw.rectangle((x0, band_y, x1, band_y + 14), fill=color)
    draw.text((left, band_y + 22), "label sequence", fill=NEUTRAL["muted"], font=body_font)

    image.save(output_path)
    return True


def draw_sample_grid(session_dir, rows, output_path, title, samples_per_label, scale):
    if Image is None:
        return False

    grouped = defaultdict(list)
    for row in rows:
        grouped[row["label"]].append(row)

    labels = ordered_labels(Counter(row["label"] for row in rows))
    sample_rows = {label: select_samples(grouped[label], samples_per_label) for label in labels}

    loaded_samples = []
    for label in labels:
        for row in sample_rows[label]:
            frame_path = session_dir / row["filename"]
            frame = np.squeeze(np.load(frame_path, allow_pickle=False))
            if frame.shape != (60, 80):
                raise ValueError(f"{frame_path} has shape {frame.shape}, expected 60x80")
            if not np.issubdtype(frame.dtype, np.number) or not np.all(np.isfinite(frame)):
                raise ValueError(f"{frame_path} does not contain finite numeric temperatures")
            loaded_samples.append((label, row, frame))

    if not loaded_samples:
        return False

    sample_values = np.concatenate([raw_to_celsius(frame).reshape(-1) for _, _, frame in loaded_samples])
    row_max_values = [row["max_c"] for row in rows]
    display_min = float(np.percentile(sample_values, 2))
    display_max = float(np.percentile(row_max_values, 90))
    display_max = max(display_max, display_min + 6.0)

    cell_w = 80 * scale
    cell_h = 60 * scale
    left = 220
    top = 112
    gap = 18
    caption_h = 42
    width = left + samples_per_label * (cell_w + gap) + 42
    height = top + len(labels) * (cell_h + caption_h + gap) + 40

    image = Image.new("RGB", (width, height), NEUTRAL["background"])
    draw = ImageDraw.Draw(image)
    title_font = load_font(32, bold=True)
    label_font = load_font(22, bold=True)
    body_font = load_font(16)
    small_font = load_font(14)

    draw.text((42, 28), title, fill=NEUTRAL["ink"], font=title_font)
    subtitle = f"Evenly sampled frames by label, display range {display_min:.1f}C to {display_max:.1f}C"
    draw.text((42, 67), subtitle, fill=NEUTRAL["muted"], font=small_font)

    sample_lookup = defaultdict(list)
    for label, row, frame in loaded_samples:
        sample_lookup[label].append((row, frame))

    for label_idx, label in enumerate(labels):
        row_y = top + label_idx * (cell_h + caption_h + gap)
        color = LABEL_COLORS.get(label, (120, 120, 120))
        draw.rounded_rectangle((42, row_y, 58, row_y + cell_h), radius=6, fill=color)
        draw.text((70, row_y + 10), label, fill=NEUTRAL["ink"], font=label_font)
        draw.text((70, row_y + 40), f"{len(grouped[label])} frames", fill=NEUTRAL["muted"], font=body_font)

        for col, (sample_row, frame) in enumerate(sample_lookup[label]):
            x = left + col * (cell_w + gap)
            thumb = thermal_image(frame, display_min, display_max, scale)
            image.paste(thumb, (x, row_y))
            draw.rectangle((x, row_y, x + cell_w, row_y + cell_h), outline=(255, 255, 255), width=2)
            caption = f"#{sample_row['frame_index']}  max {sample_row['max_c']:.1f}C"
            draw.text((x, row_y + cell_h + 10), caption, fill=NEUTRAL["ink"], font=small_font)

    image.save(output_path)
    return True


def write_summary_text(path, session_dir, rows, count_rows, temp_rows, warnings, output_files):
    duration = infer_duration(rows)
    total = len(rows)
    lines = [
        "Thermal Dataset Summary",
        "=======================",
        "",
        f"Session: {session_dir}",
        f"Frames: {total}",
    ]

    if duration is not None:
        lines.append(f"Duration: {duration:.1f} seconds ({duration / 60:.1f} minutes)")

    lines.extend(["", "Label counts:"])
    for row in count_rows:
        lines.append(f"- {row['label']}: {row['count']} frames ({row['share_pct']}%)")

    lines.extend(["", "Temperature summary:"])
    for row in temp_rows:
        lines.append(
            "- {label}: mean={mean_c_mean}C, max mean={max_c_mean}C, "
            "max p95={max_c_p95}C, max observed={max_c_max}C".format(**row)
        )

    if warnings:
        lines.extend(["", "Warnings:"])
        lines.extend(f"- {warning}" for warning in warnings)
    else:
        lines.extend(["", "Warnings: none"])

    lines.extend(["", "Generated files:"])
    lines.extend(f"- {file_path}" for file_path in output_files)
    lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarise and visualise labelled thermal dataset sessions."
    )
    parser.add_argument(
        "session",
        nargs="?",
        type=Path,
        help="Dataset session directory. Defaults to latest session under data/raw.",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory containing dataset sessions.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/analysis"),
        help="Directory where analysis output will be written.",
    )
    parser.add_argument(
        "--samples-per-label",
        type=int,
        default=4,
        help="Number of example frames to include per label in the sample grid.",
    )
    parser.add_argument(
        "--thermal-scale",
        type=int,
        default=4,
        help="Pixel scale for sample thermal images.",
    )
    parser.add_argument(
        "--min-label-count",
        type=int,
        default=300,
        help="Warning threshold for low per-label frame counts.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    session_dir = args.session if args.session is not None else latest_session(args.raw_dir)
    session_dir = session_dir.resolve()
    labels_path = session_dir / "labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels.csv: {labels_path}")

    rows = read_label_rows(labels_path)
    session_name = session_dir.name
    output_dir = (args.output_dir / session_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    count_rows = build_count_rows(rows)
    temp_rows = build_temperature_rows(rows)
    warnings = validate_dataset(session_dir, rows, args.min_label_count)

    count_csv = output_dir / "label_counts.csv"
    temp_csv = output_dir / "temperature_summary.csv"
    metadata_json = output_dir / "summary.json"
    summary_txt = output_dir / "summary.txt"

    write_csv(count_csv, count_rows, ["label", "count", "share_pct"])
    write_csv(
        temp_csv,
        temp_rows,
        [
            "label",
            "count",
            "min_c_mean",
            "mean_c_mean",
            "max_c_mean",
            "max_c_p50",
            "max_c_p90",
            "max_c_p95",
            "max_c_max",
        ],
    )

    output_files = [count_csv, temp_csv]
    chart_title = f"Thermal dataset {session_name}"
    if Image is not None:
        charts = [
            (output_dir / "label_distribution.png", draw_label_distribution(count_rows, output_dir / "label_distribution.png", chart_title)),
            (output_dir / "label_timeline.png", draw_label_timeline(rows, output_dir / "label_timeline.png", chart_title)),
            (
                output_dir / "max_temperature_timeline.png",
                draw_max_temperature_timeline(rows, output_dir / "max_temperature_timeline.png", chart_title),
            ),
            (
                output_dir / "sample_grid.png",
                draw_sample_grid(
                    session_dir,
                    rows,
                    output_dir / "sample_grid.png",
                    chart_title,
                    max(args.samples_per_label, 1),
                    max(args.thermal_scale, 1),
                ),
            ),
        ]
        output_files.extend(path for path, created in charts if created)
    else:
        warnings.append("Pillow is not installed, so PNG visualisations were skipped.")

    duration = infer_duration(rows)
    metadata = {
        "session": str(session_dir),
        "frames": len(rows),
        "duration_seconds": duration,
        "label_counts": count_rows,
        "temperature_summary": temp_rows,
        "warnings": warnings,
        "generated_files": [str(path) for path in output_files],
    }
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    output_files.append(metadata_json)

    write_summary_text(summary_txt, session_dir, rows, count_rows, temp_rows, warnings, output_files)
    output_files.append(summary_txt)

    print(f"Analysed session: {session_dir}")
    print(f"Frames: {len(rows)}")
    for row in count_rows:
        print(f"  {row['label']}: {row['count']} frames ({row['share_pct']}%)")
    if warnings:
        print("Warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
