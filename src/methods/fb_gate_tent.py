from __future__ import annotations

import copy
import numpy as np

from src.methods.common import (
    StreamingAdapter,
    build_bn_optimizer,
    chunked_entropy_update,
    correctness_bits,
    max_class_share_from_probs,
    maybe_flip_bits,
    pairwise_gate_decision,
    predict_probabilities,
    select_query_indices,
)


class FBGateTentAdapter(StreamingAdapter):
    def __init__(self, model, device, config):
        super().__init__(model=model, device=device, config=config)
        self.pending_candidate = None

    def reset(self) -> None:
        super().reset()
        self.pending_candidate = None

    def _refresh_optimizer(self) -> None:
        self.optimizer = build_bn_optimizer(
            self.model,
            lr=float(self.config.get("lr", 1e-3)),
            weight_decay=float(self.config.get("weight_decay", 0.0)),
        )

    def process_window(self, images, labels, is_id, num_queries, rng):
        delay_windows = int(self.config.get("feedback_delay_windows", 0))
        if self.pending_candidate is not None and self.pending_candidate["remaining"] <= 0:
            if self.pending_candidate["accepted"]:
                self.model.load_state_dict(copy.deepcopy(self.pending_candidate["candidate_state"]))
                self.safe_state = copy.deepcopy(self.pending_candidate["candidate_state"])
                self._refresh_optimizer()
                self.reject_streak = 0
            self.pending_candidate = None

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
        before_bits = np.zeros(0, dtype=np.int64)
        after_bits = np.zeros(0, dtype=np.int64)

        can_propose = self.pending_candidate is None

        if len(query_indices) > 0 and can_propose:
            if self.optimizer is None:
                self._refresh_optimizer()

            baseline_predictions = baseline_probs.argmax(axis=1)
            before_bits = correctness_bits(baseline_predictions[query_indices], labels[query_indices], is_id[query_indices])
            before_bits = maybe_flip_bits(before_bits, float(self.config.get("feedback_noise", 0.0)), rng)

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
            after_predictions = query_probs_after.argmax(axis=1)
            after_bits = correctness_bits(after_predictions, labels[query_indices], is_id[query_indices])
            after_bits = maybe_flip_bits(after_bits, float(self.config.get("feedback_noise", 0.0)), rng)
            accepted, delta = pairwise_gate_decision(
                before_bits,
                after_bits,
                threshold=float(self.config.get("gate_threshold", 0.0)),
                criterion=str(self.config.get("gate_criterion", "delta_mean")),
            )

        if len(query_indices) > 0 and can_propose:
            if delay_windows > 0:
                candidate_state = copy.deepcopy(self.model.state_dict())
                self.model.load_state_dict(snapshot)
                self._refresh_optimizer()
                self.pending_candidate = {
                    "remaining": delay_windows,
                    "accepted": bool(accepted),
                    "candidate_state": candidate_state,
                }
            elif accepted:
                self.safe_state = copy.deepcopy(self.model.state_dict())
                self.reject_streak = 0
            else:
                self.model.load_state_dict(snapshot)
                self._refresh_optimizer()
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
        if self.pending_candidate is not None:
            self.pending_candidate["remaining"] -= 1
        stats = {
            "candidate_evaluated": int(len(query_indices) > 0 and can_propose),
            "accepted": int(accepted),
            "gate_delta": delta,
            "tent_loss": tent_loss,
            "num_queries": int(len(query_indices)),
            "num_ood_queries": num_ood_queries,
            "query_before_accuracy": float(before_bits.mean()) if len(before_bits) else 0.0,
            "query_after_accuracy": float(after_bits.mean()) if len(after_bits) else 0.0,
            "feedback_delay_windows": delay_windows,
            "pending_candidate": int(self.pending_candidate is not None),
            "max_class_share": max_class_share_from_probs(final_probs),
        }
        return final_probs, stats
