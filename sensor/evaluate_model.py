#!/usr/bin/env python3

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np

from train_thermal_mlp import (
    LABEL_ORDER,
    confusion_matrix,
    draw_confusion_matrix,
    load_frames,
    predict,
    read_dataset_rows,
    unweighted_loss,
    write_confusion_csv,
)


def load_model(model_path):
    data = np.load(model_path)
    labels = [str(label) for label in data["labels"]]
    if labels != LABEL_ORDER:
        raise ValueError(f"Model labels {labels} do not match expected labels {LABEL_ORDER}")

    model = {
        "w1": data["w1"].astype(np.float32),
        "b1": data["b1"].astype(np.float32),
        "w2": data["w2"].astype(np.float32),
        "b2": data["b2"].astype(np.float32),
    }
    mean = data["mean"].astype(np.float32)
    std = data["std"].astype(np.float32)
    return model, mean, std


def default_output_dir(model_path, session_dirs):
    session_name = "_".join(path.name for path in session_dirs)
    safe_name = "".join(char if char.isalnum() or char in ("_", "-") else "_" for char in session_name)
    return model_path.parent / "evaluations" / safe_name


def per_class_metrics(matrix):
    rows = []
    for idx, label in enumerate(LABEL_ORDER):
        tp = int(matrix[idx, idx])
        fp = int(matrix[:, idx].sum() - tp)
        fn = int(matrix[idx, :].sum() - tp)
        support = int(matrix[idx, :].sum())

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "label": label,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )
    return rows


def write_class_metrics_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["label", "precision", "recall", "f1", "support"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "label": row["label"],
                    "precision": f"{row['precision']:.6f}",
                    "recall": f"{row['recall']:.6f}",
                    "f1": f"{row['f1']:.6f}",
                    "support": row["support"],
                }
            )


def write_predictions_csv(path, rows, y_true, y_pred, probs):
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "session",
            "frame_index",
            "filename",
            "actual",
            "predicted",
            "correct",
            "confidence",
        ] + [f"prob_{label}" for label in LABEL_ORDER]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for row, true_id, pred_id, prob_row in zip(rows, y_true, y_pred, probs):
            out = {
                "session": row["session"],
                "frame_index": row["frame_index"],
                "filename": str(row["frame_path"]),
                "actual": LABEL_ORDER[int(true_id)],
                "predicted": LABEL_ORDER[int(pred_id)],
                "correct": int(true_id == pred_id),
                "confidence": f"{float(np.max(prob_row)):.6f}",
            }
            for label_id, label in enumerate(LABEL_ORDER):
                out[f"prob_{label}"] = f"{float(prob_row[label_id]):.6f}"
            writer.writerow(out)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained thermal MLP model on independent dataset sessions."
    )
    parser.add_argument("model", type=Path, help="Path to model.npz produced by train_thermal_mlp.py.")
    parser.add_argument(
        "sessions",
        nargs="+",
        type=Path,
        help="One or more dataset session folders to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for evaluation reports. Defaults to models/<run>/evaluations/<session>.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = args.model.resolve()
    session_dirs = [path.resolve() for path in args.sessions]
    output_dir = args.output_dir.resolve() if args.output_dir else default_output_dir(model_path, session_dirs)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, mean, std = load_model(model_path)
    rows = read_dataset_rows(session_dirs)
    x, y = load_frames(rows)
    x = (x - mean) / std

    y_pred, probs = predict(model, x)
    accuracy = float(np.mean(y_pred == y))
    loss = unweighted_loss(model, x, y)
    matrix = confusion_matrix(y, y_pred, len(LABEL_ORDER))
    class_rows = per_class_metrics(matrix)

    write_confusion_csv(output_dir / "confusion_matrix.csv", matrix)
    draw_confusion_matrix(output_dir / "confusion_matrix.png", matrix, "Independent evaluation confusion matrix")
    write_class_metrics_csv(output_dir / "class_metrics.csv", class_rows)
    write_predictions_csv(output_dir / "predictions.csv", rows, y, y_pred, probs)

    label_counts = Counter(row["label"] for row in rows)
    metrics = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_path": str(model_path),
        "sessions": [str(path) for path in session_dirs],
        "labels": LABEL_ORDER,
        "frames": len(rows),
        "accuracy": accuracy,
        "loss": loss,
        "label_counts": {label: int(label_counts.get(label, 0)) for label in LABEL_ORDER},
        "class_metrics": class_rows,
        "confusion_matrix": matrix.tolist(),
        "note": "This is an independent session-level evaluation when the evaluated sessions were not used for training.",
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Evaluated model: {model_path}")
    print(f"Frames: {len(rows)}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Loss: {loss:.6f}")
    print("Label counts:")
    for label in LABEL_ORDER:
        print(f"  {label}: {label_counts.get(label, 0)}")
    print("Confusion matrix rows=actual, columns=predicted:")
    print(matrix)
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
