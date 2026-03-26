from __future__ import annotations

import copy

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


class SelfGateTentZeroBitAdapter(StreamingAdapter):
    def _refresh_optimizer(self) -> None:
        self.optimizer = build_bn_optimizer(
            self.model,
            lr=float(self.config.get("lr", 1e-3)),
            weight_decay=float(self.config.get("weight_decay", 0.0)),
        )

    def _proxy_score(self, probs: np.ndarray) -> float:
        proxy = str(self.config.get("self_gate_proxy", "confidence"))
        if proxy == "neg_entropy":
            return -mean_entropy_from_probs(probs)
        return mean_confidence_from_probs(probs)

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
        num_ood_queries = int(np.sum(~is_id[query_indices])) if len(query_indices) else 0
        snapshot = copy.deepcopy(self.model.state_dict())

        accepted = False
        delta = 0.0
        tent_loss = None
        proxy_before = 0.0
        proxy_after = 0.0

        if len(query_indices) > 0:
            if self.optimizer is None:
                self._refresh_optimizer()

            proxy_before = self._proxy_score(baseline_probs[query_indices])
            tent_loss = chunked_entropy_update(
                self.model,
                self.optimizer,
                images,
                device=self.device,
                microbatch_size=self.microbatch_size,
                steps=int(self.config.get("steps", 1)),
            )
            query_probs_after = predict_probabilities(
                self.model,
                images[query_indices],
                device=self.device,
                microbatch_size=self.microbatch_size,
            )
            proxy_after = self._proxy_score(query_probs_after)
            delta = float(proxy_after - proxy_before)
            accepted = delta > float(self.config.get("gate_threshold", 0.0))

        if accepted:
            self.safe_state = copy.deepcopy(self.model.state_dict())
            self.reject_streak = 0
        else:
            self.model.load_state_dict(snapshot)
            self._refresh_optimizer()
            if len(query_indices) > 0:
                self.reject_streak += 1
                if bool(self.config.get("rollback", True)) and self.reject_streak >= int(self.config.get("rollback_patience", 2)):
                    self.model.load_state_dict(copy.deepcopy(self.safe_state))
                    self._refresh_optimizer()
                    self.reject_streak = 0

        final_probs = predict_probabilities(
            self.model,
            images,
            device=self.device,
            microbatch_size=self.microbatch_size,
        )
        stats = {
            "candidate_evaluated": int(len(query_indices) > 0),
            "accepted": int(accepted),
            "gate_delta": delta,
            "proxy_before": proxy_before,
            "proxy_after": proxy_after,
            "tent_loss": tent_loss,
            "num_queries": int(len(query_indices)),
            "num_ood_queries": num_ood_queries,
            "max_class_share": max_class_share_from_probs(final_probs),
        }
        return final_probs, stats
