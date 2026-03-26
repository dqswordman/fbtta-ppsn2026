from __future__ import annotations

from src.methods.common import (
    StreamingAdapter,
    build_bn_optimizer,
    chunked_dirichlet_entropy_update,
    max_class_share_from_probs,
    predict_probabilities,
)


class ComeZeroBitAdapter(StreamingAdapter):
    def _refresh_optimizer(self) -> None:
        self.optimizer = build_bn_optimizer(
            self.model,
            lr=float(self.config.get("lr", 5e-4)),
            weight_decay=float(self.config.get("weight_decay", 0.0)),
        )

    def process_window(self, images, labels, is_id, num_queries, rng):
        if self.optimizer is None:
            self._refresh_optimizer()
        loss_value = chunked_dirichlet_entropy_update(
            self.model,
            self.optimizer,
            images,
            device=self.device,
            microbatch_size=self.microbatch_size,
            steps=int(self.config.get("steps", 1)),
            prior_strength=float(self.config.get("prior_strength", 1000.0)),
        )
        probs = predict_probabilities(
            self.model,
            images,
            device=self.device,
            microbatch_size=self.microbatch_size,
        )
        stats = {
            "candidate_evaluated": 1,
            "accepted": 1,
            "dirichlet_loss": loss_value,
            "num_queries": 0,
            "num_ood_queries": 0,
            "max_class_share": max_class_share_from_probs(probs),
        }
        return probs, stats
