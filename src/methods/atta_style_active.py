from __future__ import annotations

from collections import deque

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


class ATTAStyleActiveAdapter(StreamingAdapter):
    def __init__(self, model, device, config):
        super().__init__(model=model, device=device, config=config)
        self._reset_memory()

    def _reset_memory(self) -> None:
        capacity = int(self.config.get("memory_size", 2048))
        self.memory_images = deque(maxlen=capacity)
        self.memory_labels = deque(maxlen=capacity)

    def reset(self) -> None:
        super().reset()
        self._reset_memory()

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
        for index in id_query_indices:
            self.memory_images.append(images[index].detach().cpu())
            self.memory_labels.append(int(labels[index]))

        supervised_loss = None
        if self.memory_images:
            if self.optimizer is None:
                self._refresh_optimizer()
            memory_images = torch.stack(list(self.memory_images))
            memory_labels = torch.tensor(list(self.memory_labels), dtype=torch.long)
            supervised_loss = chunked_supervised_update(
                self.model,
                self.optimizer,
                memory_images,
                memory_labels,
                device=self.device,
                microbatch_size=min(self.microbatch_size, len(memory_labels)),
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
            "accepted": int(supervised_loss is not None),
            "supervised_loss": supervised_loss,
            "memory_size": int(len(self.memory_labels)),
            "num_queries": int(len(query_indices)),
            "num_ood_queries": int(np.sum(~is_id[query_indices])) if len(query_indices) else 0,
            "max_class_share": max_class_share_from_probs(final_probs),
        }
        return final_probs, stats
