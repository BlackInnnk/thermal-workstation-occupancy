#!/usr/bin/env python3

import argparse
import csv
import shutil
from collections import Counter
from pathlib import Path


def read_rows(labels_path):
    with labels_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"No CSV header found in {labels_path}")
        rows = list(reader)
        return reader.fieldnames, rows


def in_range(frame_index, start_frame, end_frame):
    if start_frame is not None and frame_index < start_frame:
        return False
    if end_frame is not None and frame_index > end_frame:
        return False
    return True


def write_rows(labels_path, fieldnames, rows):
    with labels_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def label_counts(rows):
    return Counter(row["label"] for row in rows)


def print_counts(title, counter):
    print(title)
    for label, count in sorted(counter.items()):
        print(f"  {label}: {count}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Relabel a frame range inside a thermal dataset labels.csv file."
    )
    parser.add_argument("session", type=Path, help="Dataset session directory, e.g. data/raw/eval_01.")
    parser.add_argument("--from-frame", type=int, default=None, help="First frame index to relabel.")
    parser.add_argument("--to-frame", type=int, default=None, help="Last frame index to relabel.")
    parser.add_argument("--from-label", default=None, help="Only relabel rows currently using this label.")
    parser.add_argument("--to-label", required=True, help="New label to write.")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without editing labels.csv.")
    parser.add_argument(
        "--backup",
        default=None,
        help="Backup path. Defaults to labels.csv.bak if a backup does not already exist.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    labels_path = args.session / "labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"Missing labels.csv: {labels_path}")

    fieldnames, rows = read_rows(labels_path)
    if "frame_index" not in fieldnames or "label" not in fieldnames:
        raise ValueError("labels.csv must contain frame_index and label columns")

    before_counts = label_counts(rows)
    changed = 0

    for row in rows:
        frame_index = int(row["frame_index"])
        if not in_range(frame_index, args.from_frame, args.to_frame):
            continue
        if args.from_label is not None and row["label"] != args.from_label:
            continue
        if row["label"] == args.to_label:
            continue
        row["label"] = args.to_label
        changed += 1

    after_counts = label_counts(rows)

    print(f"Session: {args.session}")
    print(f"Labels file: {labels_path}")
    print(f"Rows changed: {changed}")
    print_counts("Before:", before_counts)
    print_counts("After:", after_counts)

    if args.dry_run:
        print("Dry run only. labels.csv was not modified.")
        return

    if changed == 0:
        print("No changes to write.")
        return

    if args.backup is None:
        backup_path = labels_path.with_suffix(labels_path.suffix + ".bak")
    else:
        backup_path = Path(args.backup)

    if not backup_path.exists():
        shutil.copy2(labels_path, backup_path)
        print(f"Backup written: {backup_path}")
    else:
        print(f"Backup already exists, leaving it unchanged: {backup_path}")

    write_rows(labels_path, fieldnames, rows)
    print("labels.csv updated.")


if __name__ == "__main__":
    main()
