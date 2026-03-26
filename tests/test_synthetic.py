from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.synthetic import SyntheticConfig, frozen_weights, generate_dual_environment, predict_linear, select_queries, zero_bit_candidate


class SyntheticTests(unittest.TestCase):
    def test_fb_gate_rejects_harmful_candidate_in_negative_env(self) -> None:
        config = SyntheticConfig(n_samples=2000, feature_shift=0.6, noise_std=0.6, candidate_step=0.8)
        features, labels = generate_dual_environment(env_sign=-1, seed=0, config=config)
        frozen_pred = predict_linear(features, frozen_weights())
        candidate_pred = predict_linear(features, zero_bit_candidate(config.candidate_step))
        self.assertLess((candidate_pred == labels).mean(), (frozen_pred == labels).mean())

        gate = select_queries(
            features,
            labels,
            weights_before=frozen_weights(),
            weights_after=zero_bit_candidate(config.candidate_step),
            query_rate=0.05,
            seed=0,
            feedback_noise=0.0,
        )
        self.assertFalse(gate["accept"])


if __name__ == "__main__":
    unittest.main()
