from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src.methods.common import (
    StreamingAdapter,
    build_bn_optimizer,
    chunked_entropy_update,
    chunked_supervised_update,
    max_class_share_from_probs,
    predict_probabilities,
    select_query_indices,
)


class EATTAStyleLowLabelAdapter(StreamingAdapter):
    def __init__(self, model, device, config):
        super().__init__(model=model, device=device, config=config)
        self.current_model_probs: torch.Tensor | None = None

    def reset(self) -> None:
        super().reset()
        self.current_model_probs = None

    def _refresh_optimizer(self) -> None:
        self.optimizer = build_bn_optimizer(
            self.model,
            lr=float(self.config.get("lr", 5e-4)),
            weight_decay=float(self.config.get("weight_decay", 0.0)),
        )

    def _reliable_indices(self, probs: np.ndarray) -> np.ndarray:
        entropies = -(probs * np.log(np.clip(probs, 1e-8, 1.0))).sum(axis=1)
        reliable = np.where(entropies < float(self.config.get("e_margin", 1.5)))[0]
        if len(reliable) == 0:
            return reliable
        reliable_probs = torch.from_numpy(probs[reliable]).float()
        if self.current_model_probs is not None:
            cosine = F.cosine_similarity(reliable_probs, self.current_model_probs.unsqueeze(0), dim=1)
            reliable = reliable[torch.abs(cosine) < float(self.config.get("d_margin", 0.05))]
            reliable_probs = torch.from_numpy(probs[reliable]).float() if len(reliable) else reliable_probs[:0]
        if len(reliable):
            mean_probs = reliable_probs.mean(dim=0)
            if self.current_model_probs is None:
                self.current_model_probs = mean_probs
            else:
                self.current_model_probs = 0.9 * self.current_model_probs + 0.1 * mean_probs
        return reliable

    def process_window(self, images, labels, is_id, num_queries, rng):
        baseline_probs = predict_probabilities(
            self.model,
            images,
            device=self.device,
            microbatch_size=self.microbatch_size,
        )
        reliable_indices = self._reliable_indices(baseline_probs)
        query_indices = select_query_indices(
            baseline_probs,
            num_queries=num_queries,
            strategy=str(self.config.get("query_strategy", "entropy")),
            rng=rng,
        )
        id_query_indices = query_indices[is_id[query_indices]] if len(query_indices) else query_indices

        if self.optimizer is None:
            self._refresh_optimizer()

        entropy_loss = None
        supervised_loss = None
        if len(reliable_indices) > 0:
            entropy_loss = chunked_entropy_update(
                self.model,
                self.optimizer,
                images[reliable_indices],
                device=self.device,
                microbatch_size=min(self.microbatch_size, len(reliable_indices)),
                steps=int(self.config.get("steps", 1)),
            )
        if len(id_query_indices) > 0:
            supervised_loss = chunked_supervised_update(
                self.model,
                self.optimizer,
                images[id_query_indices],
                torch.from_numpy(labels[id_query_indices]).long(),
                device=self.device,
                microbatch_size=min(self.microbatch_size, len(id_query_indices)),
                steps=max(1, int(self.config.get("supervised_steps", 1))),
            )

        final_probs = predict_probabilities(
            self.model,
            images,
            device=self.device,
            microbatch_size=self.microbatch_size,
        )
        stats = {
            "candidate_evaluated": int((len(reliable_indices) > 0) or (len(query_indices) > 0)),
            "accepted": int((entropy_loss is not None) or (supervised_loss is not None)),
            "entropy_loss": entropy_loss,
            "supervised_loss": supervised_loss,
            "num_reliable_samples": int(len(reliable_indices)),
            "num_queries": int(len(query_indices)),
            "num_ood_queries": int(np.sum(~is_id[query_indices])) if len(query_indices) else 0,
            "max_class_share": max_class_share_from_probs(final_probs),
        }
        return final_probs, stats
