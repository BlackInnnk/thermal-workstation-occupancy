from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import types
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "occupancy_mlp_train02_relabel" / "model.npz"
SENSOR_DIR = ROOT / "sensor"
sys.path.insert(0, str(SENSOR_DIR))
sys.modules.setdefault("cv2", types.SimpleNamespace())
sys.modules.setdefault("pylepton", types.SimpleNamespace(Lepton=object))

from workstation_monitor import load_occupancy_model  # noqa: E402
from train_thermal_mlp import load_frames  # noqa: E402


class DeploymentModelTests(unittest.TestCase):
    def test_model_is_finite_and_has_expected_dimensions(self):
        with np.load(MODEL_PATH, allow_pickle=False) as data:
            labels = [str(label) for label in data["labels"]]
            w1 = data["w1"]
            b1 = data["b1"]
            w2 = data["w2"]
            b2 = data["b2"]
            mean = data["mean"]
            std = data["std"]

        self.assertEqual(labels, ["not_occupied", "occupied"])
        self.assertEqual(w1.shape, (4800, 64))
        self.assertIn(b1.shape, {(64,), (1, 64)})
        self.assertEqual(w2.shape, (64, 2))
        self.assertIn(b2.shape, {(2,), (1, 2)})
        self.assertIn(mean.shape, {(4800,), (1, 4800)})
        self.assertIn(std.shape, {(4800,), (1, 4800)})

        for values in (w1, b1, w2, b2, mean, std):
            self.assertTrue(np.all(np.isfinite(values)))
        self.assertTrue(np.all(std > 0))

    def test_model_forward_pass_produces_probabilities(self):
        with np.load(MODEL_PATH, allow_pickle=False) as data:
            hidden = np.maximum(data["b1"].astype(np.float32), 0.0)
            logits = hidden @ data["w2"].astype(np.float32) + data["b2"].astype(np.float32)

        shifted = logits - np.max(logits)
        probabilities = (np.exp(shifted) / np.exp(shifted).sum()).reshape(-1)

        self.assertEqual(probabilities.shape, (2,))
        self.assertTrue(np.all(np.isfinite(probabilities)))
        self.assertAlmostEqual(float(probabilities.sum()), 1.0, places=6)

    def test_loader_rejects_column_vector_normalisation(self):
        with TemporaryDirectory() as temporary_directory:
            malformed_path = Path(temporary_directory) / "model.npz"
            np.savez_compressed(
                malformed_path,
                w1=np.zeros((4800, 1), dtype=np.float32),
                b1=np.zeros((1, 1), dtype=np.float32),
                w2=np.zeros((1, 2), dtype=np.float32),
                b2=np.zeros((1, 2), dtype=np.float32),
                mean=np.zeros((4800, 1), dtype=np.float32),
                std=np.ones((4800, 1), dtype=np.float32),
                labels=np.array(["not_occupied", "occupied"]),
            )

            with self.assertRaisesRegex(ValueError, "normalisation"):
                load_occupancy_model(malformed_path)

    def test_training_loader_rejects_non_finite_frame(self):
        with TemporaryDirectory() as temporary_directory:
            frame_path = Path(temporary_directory) / "frame.npy"
            frame = np.full((60, 80), 30000.0, dtype=np.float32)
            frame[0, 0] = np.nan
            np.save(frame_path, frame)
            rows = [{"frame_path": frame_path, "label": "occupied"}]

            with self.assertRaisesRegex(ValueError, "non-finite"):
                load_frames(rows, labels=["not_occupied", "occupied"])


if __name__ == "__main__":
    unittest.main()
