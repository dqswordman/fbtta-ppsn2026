from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.methods.fb_gate_tent import FBGateTentAdapter


class DummyModel:
    def __init__(self) -> None:
        self.version = 0
        self.training = False

    def to(self, device):
        return self

    def state_dict(self):
        return {"version": self.version}

    def load_state_dict(self, state_dict):
        self.version = int(state_dict["version"])

    def train(self, mode: bool = True):
        self.training = mode
        return self

    def eval(self):
        self.training = False
        return self


def fake_predict_probabilities(model, images, device, microbatch_size):
    if model.version == 0:
        return np.tile(np.array([[1.0, 0.0]], dtype=np.float32), (len(images), 1))
    return np.tile(np.array([[0.0, 1.0]], dtype=np.float32), (len(images), 1))


def fake_chunked_entropy_update(model, optimizer, images, device, microbatch_size, steps):
    model.version = 1
    return 0.0


def fake_select_query_indices(probs, num_queries, strategy, rng):
    if num_queries <= 0:
        return np.zeros(0, dtype=np.int64)
    return np.array([0], dtype=np.int64)


class FBGateDelayTests(unittest.TestCase):
    @patch("src.methods.fb_gate_tent.build_bn_optimizer", return_value=object())
    @patch("src.methods.fb_gate_tent.chunked_entropy_update", side_effect=fake_chunked_entropy_update)
    @patch("src.methods.fb_gate_tent.predict_probabilities", side_effect=fake_predict_probabilities)
    @patch("src.methods.fb_gate_tent.select_query_indices", side_effect=fake_select_query_indices)
    def test_delayed_accept_applies_on_next_window(
        self,
        _select_query_indices,
        _predict_probabilities,
        _chunked_entropy_update,
        _build_bn_optimizer,
    ) -> None:
        adapter = FBGateTentAdapter(
            model=DummyModel(),
            device=torch.device("cpu"),
            config={
                "microbatch_size": 1,
                "steps": 1,
                "lr": 1e-3,
                "gate_threshold": 0.0,
                "feedback_delay_windows": 1,
            },
        )
        adapter.reset()

        images = torch.zeros((1, 3, 4, 4), dtype=torch.float32)
        labels = np.array([1], dtype=np.int64)
        is_id = np.array([True], dtype=bool)
        rng = np.random.RandomState(0)

        _, stats_first = adapter.process_window(images, labels, is_id, num_queries=1, rng=rng)
        self.assertEqual(adapter.model.version, 0)
        self.assertEqual(stats_first["accepted"], 1)
        self.assertEqual(stats_first["pending_candidate"], 1)

        _, stats_second = adapter.process_window(images, labels, is_id, num_queries=0, rng=rng)
        self.assertEqual(adapter.model.version, 1)
        self.assertEqual(stats_second["candidate_evaluated"], 0)
        self.assertEqual(stats_second["pending_candidate"], 0)


if __name__ == "__main__":
    unittest.main()
