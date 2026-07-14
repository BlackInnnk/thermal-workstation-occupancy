#!/usr/bin/env python3

import argparse
import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import platform

import numpy as np

from train_thermal_mlp import (
    LABEL_ORDER,
    OCCUPANCY_LABEL_ORDER,
    INPUT_DIM,
    confusion_matrix,
    dataset_fingerprint,
    draw_confusion_matrix,
    load_frames,
    predict,
    portable_path,
    read_dataset_rows,
    sha256_file,
    unweighted_loss,
    write_confusion_csv,
)


def load_model(model_path):
    with np.load(model_path, allow_pickle=False) as data:
        labels = [str(label) for label in data["labels"]]
        if labels == LABEL_ORDER:
            task = "state"
        elif labels == OCCUPANCY_LABEL_ORDER:
            task = "occupancy"
        else:
            raise ValueError(
                f"Model labels {labels} do not match expected labels {LABEL_ORDER} "
                f"or {OCCUPANCY_LABEL_ORDER}"
            )

        model = {
            "w1": data["w1"].astype(np.float32),
            "b1": data["b1"].astype(np.float32),
            "w2": data["w2"].astype(np.float32),
            "b2": data["b2"].astype(np.float32),
        }
        mean = data["mean"].astype(np.float32)
        std = data["std"].astype(np.float32)

    valid_normalisation_shapes = {(INPUT_DIM,), (1, INPUT_DIM)}
    hidden_units = model["w1"].shape[1] if model["w1"].ndim == 2 else None
    if (
        hidden_units is None
        or model["w1"].shape[0] != INPUT_DIM
        or model["b1"].shape not in {(hidden_units,), (1, hidden_units)}
        or model["w2"].shape != (hidden_units, len(labels))
        or model["b2"].shape not in {(len(labels),), (1, len(labels))}
        or mean.shape not in valid_normalisation_shapes
        or std.shape not in valid_normalisation_shapes
    ):
        raise ValueError(f"Model {model_path} contains incompatible array dimensions")
    arrays = (*model.values(), mean, std)
    if not all(np.all(np.isfinite(values)) for values in arrays) or np.any(std <= 0):
        raise ValueError(f"Model {model_path} contains invalid numeric values")
    return model, mean.reshape(1, INPUT_DIM), std.reshape(1, INPUT_DIM), labels, task


def default_output_dir(model_path, session_dirs):
    session_name = "_".join(path.name for path in session_dirs)
    safe_name = "".join(char if char.isalnum() or char in ("_", "-") else "_" for char in session_name)
    return model_path.parent / "evaluations" / safe_name


def per_class_metrics(matrix, labels):
    rows = []
    for idx, label in enumerate(labels):
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


def write_predictions_csv(path, rows, y_true, y_pred, probs, labels):
    with path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "session",
            "frame_index",
            "filename",
            "source_label",
            "actual",
            "predicted",
            "correct",
            "confidence",
        ] + [f"prob_{label}" for label in labels]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for row, true_id, pred_id, prob_row in zip(rows, y_true, y_pred, probs):
            out = {
                "session": row["session"],
                "frame_index": row["frame_index"],
                "filename": row["filename"],
                "source_label": row.get("source_label", row["label"]),
                "actual": labels[int(true_id)],
                "predicted": labels[int(pred_id)],
                "correct": int(true_id == pred_id),
                "confidence": f"{float(np.max(prob_row)):.6f}",
            }
            for label_id, label in enumerate(labels):
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

    model, mean, std, labels, task = load_model(model_path)
    rows = read_dataset_rows(session_dirs, task=task)
    if not rows:
        raise ValueError("No labelled frames found in the evaluation sessions")
    x, y = load_frames(rows, labels=labels)
    x = (x - mean) / std

    y_pred, probs = predict(model, x)
    accuracy = float(np.mean(y_pred == y))
    loss = unweighted_loss(model, x, y)
    matrix = confusion_matrix(y, y_pred, len(labels))
    class_rows = per_class_metrics(matrix, labels)

    write_confusion_csv(output_dir / "confusion_matrix.csv", matrix, labels=labels)
    draw_confusion_matrix(output_dir / "confusion_matrix.png", matrix, "Independent evaluation confusion matrix", labels=labels)
    write_class_metrics_csv(output_dir / "class_metrics.csv", class_rows)
    write_predictions_csv(output_dir / "predictions.csv", rows, y, y_pred, probs, labels=labels)

    label_counts = Counter(row["label"] for row in rows)
    source_counts = Counter(row["source_label"] for row in rows)
    metrics = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_path": portable_path(model_path),
        "model_sha256": sha256_file(model_path),
        "task": task,
        "sessions": [portable_path(path) for path in session_dirs],
        "dataset_fingerprints_sha256": {
            portable_path(path): dataset_fingerprint(path) for path in session_dirs
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "labels": labels,
        "frames": len(rows),
        "accuracy": accuracy,
        "loss": loss,
        "label_counts": {label: int(label_counts.get(label, 0)) for label in labels},
        "source_label_counts": {label: int(source_counts.get(label, 0)) for label in LABEL_ORDER},
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
    for label in labels:
        print(f"  {label}: {label_counts.get(label, 0)}")
    print("Confusion matrix rows=actual, columns=predicted:")
    print(matrix)
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
