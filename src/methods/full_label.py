from __future__ import annotations

import numpy as np
import torch

from src.methods.common import (
    StreamingAdapter,
    build_bn_optimizer,
    chunked_supervised_update,
    max_class_share_from_probs,
    predict_probabilities,
    select_query_indices,
)


class FullLabelActiveAdapter(StreamingAdapter):
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
        id_query_indices = query_indices[is_id[query_indices]] if len(query_indices) else query_indices
        if self.optimizer is None:
            self._refresh_optimizer()

        supervised_loss = None
        if len(id_query_indices) > 0:
            supervised_loss = chunked_supervised_update(
                self.model,
                self.optimizer,
                images[id_query_indices],
                torch.from_numpy(labels[id_query_indices]).long(),
                device=self.device,
                microbatch_size=min(self.microbatch_size, len(id_query_indices)),
                steps=int(self.config.get("steps", 1)),
            )

        final_probs = predict_probabilities(
            self.model,
            images,
            device=self.device,
            microbatch_size=self.microbatch_size,
        )
        predictions = final_probs.argmax(axis=1)
        stats = {
            "candidate_evaluated": int(len(query_indices) > 0),
            "accepted": int(len(id_query_indices) > 0),
            "supervised_loss": supervised_loss,
            "num_queries": int(len(query_indices)),
            "num_ood_queries": int(np.sum(~is_id[query_indices])) if len(query_indices) else 0,
            "max_class_share": max_class_share_from_probs(final_probs),
        }
        return final_probs, stats
