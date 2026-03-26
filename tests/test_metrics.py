from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.metrics.classification import brier_score, expected_calibration_error, overconfident_error_mass


class MetricTests(unittest.TestCase):
    def test_brier_zero_for_perfect_predictions(self) -> None:
        probs = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        labels = np.array([1, 0], dtype=np.int64)
        self.assertAlmostEqual(brier_score(probs, labels, num_classes=2), 0.0)

    def test_ece_non_negative(self) -> None:
        probs = np.array([[0.7, 0.3], [0.6, 0.4], [0.2, 0.8]], dtype=np.float32)
        labels = np.array([0, 1, 1], dtype=np.int64)
        self.assertGreaterEqual(expected_calibration_error(probs, labels), 0.0)

    def test_overconfident_error_mass(self) -> None:
        probs = np.array([[0.95, 0.05], [0.1, 0.9]], dtype=np.float32)
        labels = np.array([1, 1], dtype=np.int64)
        self.assertAlmostEqual(overconfident_error_mass(probs, labels, threshold=0.9), 0.5)


if __name__ == "__main__":
    unittest.main()
