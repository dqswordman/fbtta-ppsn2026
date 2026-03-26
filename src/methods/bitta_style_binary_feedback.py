from __future__ import annotations

from collections import deque

import numpy as np
import torch

from src.methods.common import (
    StreamingAdapter,
    build_bn_optimizer,
    chunked_negative_label_update,
    chunked_supervised_update,
    correctness_bits,
    max_class_share_from_probs,
    maybe_flip_bits,
    predict_probabilities,
    select_query_indices,
)


class BiTTAStyleBinaryFeedbackAdapter(StreamingAdapter):
    def __init__(self, model, device, config):
        super().__init__(model=model, device=device, config=config)
        self._reset_memory()

    def _reset_memory(self) -> None:
        capacity = int(self.config.get("memory_size", 2048))
        self.positive_images = deque(maxlen=capacity)
        self.positive_labels = deque(maxlen=capacity)
        self.negative_images = deque(maxlen=capacity)
        self.negative_labels = deque(maxlen=capacity)

    def reset(self) -> None:
        super().reset()
        self._reset_memory()

    def _refresh_optimizer(self) -> None:
        self.optimizer = build_bn_optimizer(
            self.model,
            lr=float(self.config.get("lr", 5e-4)),
            weight_decay=float(self.config.get("weight_decay", 0.0)),
        )

    def process_window(self, images, labels, is_id, num_queries, rng):
        baseline_probs = predict_probabilities(
            self.model,
            images,
            device=self.device,
            microbatch_size=self.microbatch_size,
        )
        baseline_predictions = baseline_probs.argmax(axis=1)
        query_indices = select_query_indices(
            baseline_probs,
            num_queries=num_queries,
            strategy=str(self.config.get("query_strategy", "entropy")),
            rng=rng,
        )
        bits = correctness_bits(baseline_predictions[query_indices], labels[query_indices], is_id[query_indices]) if len(query_indices) else np.zeros(0, dtype=np.int64)
        bits = maybe_flip_bits(bits, float(self.config.get("feedback_noise", 0.0)), rng) if len(bits) else bits

        positive_indices = query_indices[bits == 1]
        negative_indices = query_indices[bits == 0]
        for index in positive_indices:
            self.positive_images.append(images[index].detach().cpu())
            self.positive_labels.append(int(baseline_predictions[index]))
        for index in negative_indices:
            self.negative_images.append(images[index].detach().cpu())
            self.negative_labels.append(int(baseline_predictions[index]))

        positive_loss = None
        negative_loss = None
        if self.optimizer is None:
            self._refresh_optimizer()
        if self.positive_images:
            pos_images = torch.stack(list(self.positive_images))
            pos_labels = torch.tensor(list(self.positive_labels), dtype=torch.long)
            positive_loss = chunked_supervised_update(
                self.model,
                self.optimizer,
                pos_images,
                pos_labels,
                device=self.device,
                microbatch_size=min(self.microbatch_size, len(pos_labels)),
                steps=int(self.config.get("steps", 1)),
            )
        if self.negative_images:
            neg_images = torch.stack(list(self.negative_images))
            neg_labels = torch.tensor(list(self.negative_labels), dtype=torch.long)
            negative_loss = chunked_negative_label_update(
                self.model,
                self.optimizer,
                neg_images,
                neg_labels,
                device=self.device,
                microbatch_size=min(self.microbatch_size, len(neg_labels)),
                steps=int(self.config.get("steps", 1)),
                weight=float(self.config.get("negative_weight", 0.5)),
            )

        final_probs = predict_probabilities(
            self.model,
            images,
            device=self.device,
            microbatch_size=self.microbatch_size,
        )
        stats = {
            "candidate_evaluated": int(len(query_indices) > 0),
            "accepted": int((positive_loss is not None) or (negative_loss is not None)),
            "positive_loss": positive_loss,
            "negative_loss": negative_loss,
            "num_positive_bits": int(len(positive_indices)),
            "num_negative_bits": int(len(negative_indices)),
            "num_queries": int(len(query_indices)),
            "num_ood_queries": int(np.sum(~is_id[query_indices])) if len(query_indices) else 0,
            "max_class_share": max_class_share_from_probs(final_probs),
        }
        return final_probs, stats
