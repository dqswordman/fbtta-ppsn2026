from __future__ import annotations

import numpy as np

from src.methods.common import StreamingAdapter, max_class_share_from_probs, predict_probabilities


class FrozenAdapter(StreamingAdapter):
    def process_window(self, images, labels, is_id, num_queries, rng):
        probs = predict_probabilities(
            self.model,
            images,
            device=self.device,
            microbatch_size=self.microbatch_size,
        )
        predictions = probs.argmax(axis=1)
        stats = {
            "candidate_evaluated": 0,
            "accepted": 0,
            "num_queries": 0,
            "num_ood_queries": 0,
            "max_class_share": max_class_share_from_probs(probs),
        }
        return probs, stats
