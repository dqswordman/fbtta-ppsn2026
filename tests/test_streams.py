from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.streams import build_contamination_stream


class StreamTests(unittest.TestCase):
    def test_contamination_ratio(self) -> None:
        id_images = np.zeros((10, 32, 32, 3), dtype=np.uint8)
        id_labels = np.arange(10, dtype=np.int64) % 10
        ood_images = np.ones((10, 32, 32, 3), dtype=np.uint8)
        images, labels, is_id = build_contamination_stream(id_images, id_labels, ood_images, ratio=0.2, seed=0, total_length=10)
        self.assertEqual(images.shape[0], 10)
        self.assertEqual(int((~is_id).sum()), 2)
        self.assertEqual(int((labels == -1).sum()), 2)


if __name__ == "__main__":
    unittest.main()
