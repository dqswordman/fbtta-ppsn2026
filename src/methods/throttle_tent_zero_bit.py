from __future__ import annotations

import numpy as np

from src.methods.common import (
    StreamingAdapter,
    build_bn_optimizer,
    chunked_entropy_update,
    max_class_share_from_probs,
    mean_confidence_from_probs,
    mean_entropy_from_probs,
    predict_probabilities,
    select_query_indices,
)


class ThrottleTentZeroBitAdapter(StreamingAdapter):
    def _refresh_optimizer(self) -> None:
        self.optimizer = build_bn_optimizer(
            self.model,
            lr=float(self.config.get("lr", 1e-3)),
            weight_decay=float(self.config.get("weight_decay", 0.0)),
        )

    def process_window(self, images, labels, is_id, num_queries, rng):
        baseline_probs = predict_probabilities(
            self.model,
            images,
            device=self.device,
            microbatch_size=self.microbatch_size,
        )
        query_indices = select_query_indices(
            baseline_probs,
            num_queries=num_queries,
            strategy=str(self.config.get("query_strategy", "entropy")),
            rng=rng,
        )
        stats_slice = baseline_probs[query_indices] if len(query_indices) else baseline_probs
        metric = str(self.config.get("throttle_metric", "entropy"))
        if metric == "confidence":
            score = mean_confidence_from_probs(stats_slice)
            threshold = float(self.config.get("confidence_threshold", 0.55))
            should_update = len(query_indices) > 0 and score <= threshold
        else:
            score = mean_entropy_from_probs(stats_slice)
            threshold = float(self.config.get("entropy_threshold", 1.75))
            should_update = len(query_indices) > 0 and score >= threshold

        tent_loss = None
        if should_update:
            if self.optimizer is None:
                self._refresh_optimizer()
            tent_loss = chunked_entropy_update(
                self.model,
                self.optimizer,
                images,
                device=self.device,
                microbatch_size=self.microbatch_size,
                steps=int(self.config.get("steps", 1)),
            )

        final_probs = predict_probabilities(
            self.model,
            images,
            device=self.device,
            microbatch_size=self.microbatch_size,
        )
        stats = {
            "candidate_evaluated": int(len(query_indices) > 0),
            "accepted": int(should_update),
            "throttle_score": score,
            "tent_loss": tent_loss,
            "num_queries": int(len(query_indices)),
            "num_ood_queries": int(np.sum(~is_id[query_indices])) if len(query_indices) else 0,
            "max_class_share": max_class_share_from_probs(final_probs),
        }
        return final_probs, stats
