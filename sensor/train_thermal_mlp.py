#!/usr/bin/env python3

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import platform

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - optional reporting dependency
    Image = None
    ImageDraw = None
    ImageFont = None


LABEL_ORDER = ["free", "occupied", "cooling", "hot_empty"]
OCCUPANCY_LABEL_ORDER = ["not_occupied", "occupied"]
TASK_LABELS = {
    "state": LABEL_ORDER,
    "occupancy": OCCUPANCY_LABEL_ORDER,
}
OCCUPANCY_LABEL_MAP = {
    "free": "not_occupied",
    "cooling": "not_occupied",
    "hot_empty": "not_occupied",
    "occupied": "occupied",
}
FRAME_HEIGHT = 60
FRAME_WIDTH = 80
INPUT_DIM = FRAME_HEIGHT * FRAME_WIDTH

LABEL_COLORS = {
    "free": (28, 132, 128),
    "occupied": (212, 92, 121),
    "cooling": (56, 110, 190),
    "hot_empty": (210, 82, 47),
    "not_occupied": (28, 132, 128),
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


def portable_path(path):
    """Prefer repository-relative paths in reports so they can be shared safely."""
    resolved = Path(path).resolve()
    try:
        return str(resolved.relative_to(Path.cwd().resolve()))
    except ValueError:
        return resolved.name


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_fingerprint(session_dir):
    """Hash labels and every referenced radiometric frame for provenance."""
    session_dir = Path(session_dir)
    labels_path = session_dir / "labels.csv"
    digest = hashlib.sha256()
    digest.update(b"labels.csv\0")
    digest.update(labels_path.read_bytes())

    with labels_path.open(newline="", encoding="utf-8") as handle:
        filenames = sorted({row["filename"] for row in csv.DictReader(handle)})
    for filename in filenames:
        frame_path = session_dir / filename
        digest.update(filename.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(frame_path).encode("ascii"))
    return digest.hexdigest()


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


def label_for_task(source_label, task):
    if task == "occupancy":
        return OCCUPANCY_LABEL_MAP[source_label]
    return source_label


def read_dataset_rows(session_dirs, task="state"):
    rows = []
    labels = TASK_LABELS[task]
    for session_dir in session_dirs:
        labels_path = session_dir / "labels.csv"
        if not labels_path.exists():
            raise FileNotFoundError(f"Missing labels.csv: {labels_path}")

        with labels_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            required = {"frame_index", "filename", "label"}
            missing = required.difference(reader.fieldnames or [])
            if missing:
                raise ValueError(f"{labels_path} is missing columns: {', '.join(sorted(missing))}")

            for row in reader:
                source_label = row["label"]
                if source_label not in LABEL_ORDER:
                    raise ValueError(f"Unknown label '{source_label}' in {labels_path}")
                label = label_for_task(source_label, task)
                if label not in labels:
                    raise ValueError(f"Unknown mapped label '{label}' in {labels_path}")

                frame_path = session_dir / row["filename"]
                if not frame_path.exists():
                    raise FileNotFoundError(f"Missing frame file referenced by labels.csv: {frame_path}")

                rows.append(
                    {
                        "session": session_dir.name,
                        "frame_index": int(row["frame_index"]),
                        "filename": row["filename"],
                        "frame_path": frame_path,
                        "source_label": source_label,
                        "label": label,
                    }
                )
    return rows


def load_frames(rows, labels=LABEL_ORDER):
    x = np.empty((len(rows), INPUT_DIM), dtype=np.float32)
    y = np.empty(len(rows), dtype=np.int64)

    for idx, row in enumerate(rows):
        frame = np.squeeze(np.load(row["frame_path"], allow_pickle=False))
        if frame.shape != (FRAME_HEIGHT, FRAME_WIDTH):
            raise ValueError(f"{row['frame_path']} has shape {frame.shape}, expected 60x80")
        if not np.issubdtype(frame.dtype, np.number):
            raise ValueError(f"{row['frame_path']} does not contain numeric temperatures")
        if not np.all(np.isfinite(frame)):
            raise ValueError(f"{row['frame_path']} contains non-finite temperatures")

        temperature_c = raw_to_celsius(frame)
        if not np.all(np.isfinite(temperature_c)):
            raise ValueError(f"{row['frame_path']} cannot be converted to finite Celsius values")

        x[idx] = temperature_c.reshape(-1)
        y[idx] = labels.index(row["label"])

    return x, y


def stratified_split(y, val_ratio, test_ratio, seed, labels=LABEL_ORDER):
    rng = np.random.default_rng(seed)
    train_indices = []
    val_indices = []
    test_indices = []

    for label_id in range(len(labels)):
        label_indices = np.where(y == label_id)[0]
        rng.shuffle(label_indices)

        n = len(label_indices)
        if n < 3:
            raise ValueError(f"Not enough samples for label {labels[label_id]}: {n}")

        n_test = max(1, int(round(n * test_ratio)))
        n_val = max(1, int(round(n * val_ratio)))
        if n_test + n_val >= n:
            n_test = 1
            n_val = 1

        test_indices.extend(label_indices[:n_test])
        val_indices.extend(label_indices[n_test : n_test + n_val])
        train_indices.extend(label_indices[n_test + n_val :])

    train_indices = np.array(train_indices, dtype=np.int64)
    val_indices = np.array(val_indices, dtype=np.int64)
    test_indices = np.array(test_indices, dtype=np.int64)
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    rng.shuffle(test_indices)

    return train_indices, val_indices, test_indices


def standardize(train_x, val_x, test_x):
    mean = train_x.mean(axis=0, keepdims=True)
    std = train_x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-3, 1.0, std)
    return (train_x - mean) / std, (val_x - mean) / std, (test_x - mean) / std, mean, std


def one_hot(y, num_classes):
    out = np.zeros((len(y), num_classes), dtype=np.float32)
    out[np.arange(len(y)), y] = 1.0
    return out


def relu(x):
    return np.maximum(x, 0.0)


def softmax(logits):
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def init_model(input_dim, hidden_dim, output_dim, seed):
    rng = np.random.default_rng(seed)
    w1 = rng.normal(0.0, math.sqrt(2.0 / input_dim), size=(input_dim, hidden_dim)).astype(np.float32)
    b1 = np.zeros((1, hidden_dim), dtype=np.float32)
    w2 = rng.normal(0.0, math.sqrt(2.0 / hidden_dim), size=(hidden_dim, output_dim)).astype(np.float32)
    b2 = np.zeros((1, output_dim), dtype=np.float32)
    return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}


def forward(model, x):
    hidden_pre = x @ model["w1"] + model["b1"]
    hidden = relu(hidden_pre)
    logits = hidden @ model["w2"] + model["b2"]
    return hidden_pre, hidden, logits


def weighted_loss_and_grads(model, x, y, class_weights, l2):
    hidden_pre, hidden, logits = forward(model, x)
    probs = softmax(logits)
    batch_size = len(y)
    weights = class_weights[y].reshape(-1, 1)
    normalizer = float(weights.sum())

    correct_probs = np.clip(probs[np.arange(batch_size), y], 1e-9, 1.0)
    loss = -float((weights.reshape(-1) * np.log(correct_probs)).sum() / normalizer)
    loss += 0.5 * l2 * (float(np.sum(model["w1"] ** 2)) + float(np.sum(model["w2"] ** 2)))

    dlogits = probs
    dlogits[np.arange(batch_size), y] -= 1.0
    dlogits *= weights / normalizer

    dw2 = hidden.T @ dlogits + l2 * model["w2"]
    db2 = dlogits.sum(axis=0, keepdims=True)
    dhidden = dlogits @ model["w2"].T
    dhidden_pre = dhidden * (hidden_pre > 0.0)
    dw1 = x.T @ dhidden_pre + l2 * model["w1"]
    db1 = dhidden_pre.sum(axis=0, keepdims=True)

    grads = {"w1": dw1, "b1": db1, "w2": dw2, "b2": db2}
    return loss, grads


def adam_update(model, grads, state, lr, beta1=0.9, beta2=0.999, eps=1e-8):
    state["t"] += 1
    t = state["t"]

    for key in model:
        state["m"][key] = beta1 * state["m"][key] + (1.0 - beta1) * grads[key]
        state["v"][key] = beta2 * state["v"][key] + (1.0 - beta2) * (grads[key] ** 2)
        m_hat = state["m"][key] / (1.0 - beta1**t)
        v_hat = state["v"][key] / (1.0 - beta2**t)
        model[key] -= lr * m_hat / (np.sqrt(v_hat) + eps)


def predict(model, x):
    _, _, logits = forward(model, x)
    return np.argmax(logits, axis=1), softmax(logits)


def accuracy(model, x, y):
    pred, _ = predict(model, x)
    return float(np.mean(pred == y))


def unweighted_loss(model, x, y):
    _, _, logits = forward(model, x)
    probs = softmax(logits)
    correct_probs = np.clip(probs[np.arange(len(y)), y], 1e-9, 1.0)
    return -float(np.mean(np.log(correct_probs)))


def confusion_matrix(y_true, y_pred, num_classes):
    matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_id, pred_id in zip(y_true, y_pred):
        matrix[true_id, pred_id] += 1
    return matrix


def class_weights(y, labels=LABEL_ORDER):
    counts = np.bincount(y, minlength=len(labels)).astype(np.float32)
    total = counts.sum()
    weights = total / (len(labels) * np.maximum(counts, 1.0))
    return weights.astype(np.float32)


def write_training_curves(path, history):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"],
        )
        writer.writeheader()
        writer.writerows(history)


def write_confusion_csv(path, matrix, labels=LABEL_ORDER):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["actual\\predicted"] + labels)
        for label, row in zip(labels, matrix):
            writer.writerow([label] + [int(value) for value in row])


def draw_confusion_matrix(path, matrix, title, labels=LABEL_ORDER):
    if Image is None:
        return False

    title_font = load_font(30, bold=True)
    label_font = load_font(18, bold=True)
    body_font = load_font(18)
    small_font = load_font(14)

    sizing_image = Image.new("RGB", (1, 1), NEUTRAL["background"])
    sizing_draw = ImageDraw.Draw(sizing_image)
    label_boxes = [sizing_draw.textbbox((0, 0), label, font=label_font) for label in labels]
    label_widths = [box[2] - box[0] for box in label_boxes]
    title_box = sizing_draw.textbbox((0, 0), title, font=title_font)
    title_width = title_box[2] - title_box[0]
    subtitle = "Rows are actual labels, columns are predicted labels"
    subtitle_box = sizing_draw.textbbox((0, 0), subtitle, font=small_font)
    subtitle_width = subtitle_box[2] - subtitle_box[0]

    cell = max(112, max(label_widths, default=0) + 30)
    left = max(190, max(label_widths, default=0) + 96)
    top = 130
    grid_width = cell * len(labels)
    width = max(left + grid_width + 60, title_width + 72, subtitle_width + 72)
    height = top + cell * len(labels) + 90
    image = Image.new("RGB", (width, height), NEUTRAL["background"])
    draw = ImageDraw.Draw(image)

    draw.text((36, 28), title, fill=NEUTRAL["ink"], font=title_font)
    draw.text((36, 67), subtitle, fill=NEUTRAL["muted"], font=small_font)

    max_value = int(matrix.max()) if matrix.size else 1
    max_value = max(max_value, 1)

    for col, label in enumerate(labels):
        x = left + col * cell
        label_width = label_widths[col]
        draw.text(
            (x + (cell - 8 - label_width) / 2, top - 38),
            label,
            fill=NEUTRAL["ink"],
            font=label_font,
        )

    for row, label in enumerate(labels):
        y = top + row * cell
        color = LABEL_COLORS.get(label, (120, 120, 120))
        draw.rounded_rectangle((36, y + 28, 52, y + 68), radius=5, fill=color)
        draw.text((64, y + 36), label, fill=NEUTRAL["ink"], font=label_font)

        row_total = int(matrix[row].sum())
        for col in range(len(labels)):
            x = left + col * cell
            value = int(matrix[row, col])
            intensity = value / max_value
            fill = (
                int(245 - 190 * intensity),
                int(248 - 130 * intensity),
                int(250 - 55 * intensity),
            )
            draw.rounded_rectangle((x, y, x + cell - 8, y + cell - 8), radius=10, fill=fill)
            text = str(value)
            text_bbox = draw.textbbox((0, 0), text, font=body_font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
            text_color = (255, 255, 255) if intensity > 0.52 else NEUTRAL["ink"]
            draw.text((x + (cell - 8 - text_w) / 2, y + 28), text, fill=text_color, font=body_font)
            if row_total:
                pct = f"{value / row_total * 100:.0f}%"
                pct_bbox = draw.textbbox((0, 0), pct, font=small_font)
                pct_w = pct_bbox[2] - pct_bbox[0]
                draw.text((x + (cell - 8 - pct_w) / 2, y + 53), pct, fill=text_color, font=small_font)

    image.save(path)
    return True


def train(args):
    if args.hidden < 1 or args.epochs < 1 or args.batch_size < 1:
        raise ValueError("Hidden units, epochs, and batch size must be positive")
    if args.learning_rate <= 0 or args.l2 < 0:
        raise ValueError("Learning rate must be positive and L2 cannot be negative")
    if not 0 < args.val_ratio < 1 or not 0 < args.test_ratio < 1:
        raise ValueError("Validation and test ratios must be between 0 and 1")
    if args.val_ratio + args.test_ratio >= 1:
        raise ValueError("Validation and test ratios must sum to less than 1")
    if args.patience < 1 or args.log_every < 1:
        raise ValueError("Patience and log interval must be positive")

    session_dirs = [path.resolve() for path in args.sessions]
    labels = TASK_LABELS[args.task]
    rows = read_dataset_rows(session_dirs, task=args.task)
    if not rows:
        raise ValueError("No labelled frames found.")

    x, y = load_frames(rows, labels=labels)
    train_idx, val_idx, test_idx = stratified_split(y, args.val_ratio, args.test_ratio, args.seed, labels=labels)

    train_x, val_x, test_x = x[train_idx], x[val_idx], x[test_idx]
    train_y, val_y, test_y = y[train_idx], y[val_idx], y[test_idx]
    train_x, val_x, test_x, mean, std = standardize(train_x, val_x, test_x)

    model = init_model(INPUT_DIM, args.hidden, len(labels), args.seed)
    weights = class_weights(train_y, labels=labels)
    rng = np.random.default_rng(args.seed)
    adam_state = {
        "t": 0,
        "m": {key: np.zeros_like(value) for key, value in model.items()},
        "v": {key: np.zeros_like(value) for key, value in model.items()},
    }

    history = []
    best_model = {key: value.copy() for key, value in model.items()}
    best_val_acc = -1.0
    best_val_loss = float("inf")
    patience_remaining = args.patience

    for epoch in range(1, args.epochs + 1):
        order = np.arange(len(train_x))
        rng.shuffle(order)
        losses = []

        for start in range(0, len(order), args.batch_size):
            batch_idx = order[start : start + args.batch_size]
            loss, grads = weighted_loss_and_grads(
                model,
                train_x[batch_idx],
                train_y[batch_idx],
                weights,
                args.l2,
            )
            adam_update(model, grads, adam_state, args.learning_rate)
            losses.append(loss)

        train_acc = accuracy(model, train_x, train_y)
        val_acc = accuracy(model, val_x, val_y)
        train_loss = float(np.mean(losses))
        val_loss = unweighted_loss(model, val_x, val_y)
        history.append(
            {
                "epoch": epoch,
                "train_loss": f"{train_loss:.6f}",
                "train_acc": f"{train_acc:.6f}",
                "val_loss": f"{val_loss:.6f}",
                "val_acc": f"{val_acc:.6f}",
            }
        )

        improved = (val_acc > best_val_acc) or (val_acc == best_val_acc and val_loss < best_val_loss)
        if improved:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_model = {key: value.copy() for key, value in model.items()}
            patience_remaining = args.patience
        else:
            patience_remaining -= 1

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(
                f"epoch {epoch:03d} | train_loss={train_loss:.4f} "
                f"train_acc={train_acc:.3f} val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
            )

        if patience_remaining <= 0:
            print(f"Early stopping at epoch {epoch}.")
            break

    model = best_model
    test_pred, _ = predict(model, test_x)
    test_acc = float(np.mean(test_pred == test_y))
    test_loss = unweighted_loss(model, test_x, test_y)
    matrix = confusion_matrix(test_y, test_pred, len(labels))

    run_name = args.run_name or datetime.now().strftime("thermal_mlp_%Y%m%d_%H%M%S")
    output_dir = (args.output_dir / run_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)

    model_path = output_dir / "model.npz"
    np.savez_compressed(
        model_path,
        w1=model["w1"],
        b1=model["b1"],
        w2=model["w2"],
        b2=model["b2"],
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        labels=np.array(labels),
        task=np.array([args.task]),
        frame_height=np.array([FRAME_HEIGHT], dtype=np.int64),
        frame_width=np.array([FRAME_WIDTH], dtype=np.int64),
    )

    write_training_curves(output_dir / "training_curves.csv", history)
    write_confusion_csv(output_dir / "confusion_matrix.csv", matrix, labels=labels)
    draw_confusion_matrix(output_dir / "confusion_matrix.png", matrix, "Thermal MLP test confusion matrix", labels=labels)

    split_counts = {
        "train": Counter(labels[idx] for idx in train_y),
        "val": Counter(labels[idx] for idx in val_y),
        "test": Counter(labels[idx] for idx in test_y),
    }
    overall_counts = Counter(row["label"] for row in rows)
    source_counts = Counter(row["source_label"] for row in rows)

    metrics = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": "one_hidden_layer_mlp",
        "note": "Initial neural-network baseline; random frame-level split can overestimate performance for time-correlated video.",
        "task": args.task,
        "sessions": [portable_path(path) for path in session_dirs],
        "labels": labels,
        "frame_shape": [FRAME_HEIGHT, FRAME_WIDTH],
        "hidden_units": args.hidden,
        "model_sha256": sha256_file(model_path),
        "dataset_fingerprints_sha256": {
            portable_path(path): dataset_fingerprint(path) for path in session_dirs
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "training_parameters": {
            "seed": args.seed,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "l2": args.l2,
            "validation_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "early_stopping_patience": args.patience,
        },
        "epochs_requested": args.epochs,
        "epochs_run": len(history),
        "best_val_accuracy": best_val_acc,
        "best_val_loss": best_val_loss,
        "test_accuracy": test_acc,
        "test_loss": test_loss,
        "class_weights": {label: float(weights[idx]) for idx, label in enumerate(labels)},
        "overall_counts": {label: int(overall_counts.get(label, 0)) for label in labels},
        "source_label_counts": {label: int(source_counts.get(label, 0)) for label in LABEL_ORDER},
        "split_counts": {
            split: {label: int(counter.get(label, 0)) for label in labels}
            for split, counter in split_counts.items()
        },
        "confusion_matrix": matrix.tolist(),
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("")
    print(f"Saved model and reports to: {output_dir}")
    print(f"Best validation accuracy: {best_val_acc:.3f}")
    print(f"Test accuracy: {test_acc:.3f}")
    print("Test confusion matrix rows=actual, columns=predicted:")
    print(matrix)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a small NumPy neural-network baseline for thermal workstation classification."
    )
    parser.add_argument(
        "sessions",
        nargs="+",
        type=Path,
        help="One or more dataset session folders containing labels.csv and frames/.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("models"), help="Model output directory.")
    parser.add_argument("--run-name", default=None, help="Optional model run folder name.")
    parser.add_argument(
        "--task",
        choices=sorted(TASK_LABELS),
        default="state",
        help="Training target: state=4 classes, occupancy=binary occupied/not_occupied.",
    )
    parser.add_argument("--hidden", type=int, default=64, help="Hidden units in the MLP.")
    parser.add_argument("--epochs", type=int, default=80, help="Maximum training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Adam learning rate.")
    parser.add_argument("--l2", type=float, default=0.0001, help="L2 weight decay.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation ratio per class.")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test ratio per class.")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--log-every", type=int, default=5, help="Print training metrics every N epochs.")
    return parser.parse_args()


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
