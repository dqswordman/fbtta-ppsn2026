from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=1)
    return -(probabilities * torch.log(probabilities.clamp_min(1e-8))).sum(dim=1)


def dirichlet_entropy(logits: torch.Tensor, prior_strength: float = 1000.0) -> torch.Tensor:
    norms = torch.norm(logits, p=2, dim=-1, keepdim=True).clamp_min(1e-8)
    scaled = logits / norms * norms.detach()
    exp_logits = torch.exp(scaled)
    partition = exp_logits.sum(dim=1, keepdim=True) + prior_strength
    brief = exp_logits / partition
    uncertainty = prior_strength / partition
    probabilities = torch.cat([brief, uncertainty], dim=1).clamp_min(1e-7)
    return -(probabilities * torch.log(probabilities)).sum(dim=1)


def collect_bn_params(model: nn.Module) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            if module.weight is not None:
                params.append(module.weight)
            if module.bias is not None:
                params.append(module.bias)
    return params


def configure_model_for_tent(model: nn.Module) -> None:
    model.train()
    model.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.requires_grad_(True)
            module.track_running_stats = True
            module.running_mean = module.running_mean
            module.running_var = module.running_var


def build_bn_optimizer(model: nn.Module, lr: float, weight_decay: float = 0.0) -> optim.Optimizer:
    params = collect_bn_params(model)
    return optim.Adam(params, lr=lr, weight_decay=weight_decay)


def predict_probabilities(
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
    microbatch_size: int,
) -> np.ndarray:
    previous_mode = model.training
    model.eval()
    outputs: list[torch.Tensor] = []
    with torch.no_grad():
        for chunk in torch.split(images, microbatch_size):
            logits = model(chunk.to(device, non_blocking=True))
            outputs.append(torch.softmax(logits, dim=1).cpu())
    if previous_mode:
        model.train()
    return torch.cat(outputs, dim=0).numpy()


def mean_confidence_from_probs(probs: np.ndarray) -> float:
    return float(np.max(probs, axis=1).mean())


def mean_entropy_from_probs(probs: np.ndarray) -> float:
    return float((-(probs * np.log(np.clip(probs, 1e-8, 1.0))).sum(axis=1)).mean())


def max_class_share_from_probs(probs: np.ndarray) -> float:
    predictions = probs.argmax(axis=1)
    num_classes = probs.shape[1]
    return float(np.bincount(predictions, minlength=num_classes).max() / len(predictions))


def correctness_bits(predictions: np.ndarray, labels: np.ndarray, is_id: np.ndarray) -> np.ndarray:
    bits = np.zeros(len(predictions), dtype=np.int64)
    id_mask = is_id.astype(bool)
    bits[id_mask] = (predictions[id_mask] == labels[id_mask]).astype(np.int64)
    return bits


def maybe_flip_bits(bits: np.ndarray, noise: float, rng: np.random.RandomState) -> np.ndarray:
    if noise <= 0.0:
        return bits
    flips = rng.rand(len(bits)) < noise
    return np.where(flips, 1 - bits, bits)


def select_query_indices(
    probs: np.ndarray,
    num_queries: int,
    strategy: str,
    rng: np.random.RandomState,
) -> np.ndarray:
    if num_queries <= 0:
        return np.zeros(0, dtype=np.int64)

    if strategy == "random":
        indices = rng.permutation(len(probs))[:num_queries]
        return np.sort(indices.astype(np.int64))

    if strategy == "margin":
        sorted_probs = np.sort(probs, axis=1)
        scores = -(sorted_probs[:, -1] - sorted_probs[:, -2])
    else:
        entropy = -(probs * np.log(np.clip(probs, 1e-8, 1.0))).sum(axis=1)
        scores = entropy

    ranked = np.argsort(-scores)
    return np.sort(ranked[:num_queries].astype(np.int64))


def pairwise_gate_decision(
    before_bits: np.ndarray,
    after_bits: np.ndarray,
    threshold: float,
    criterion: str = "delta_mean",
) -> tuple[bool, float]:
    if len(before_bits) == 0:
        return False, 0.0
    delta = float(after_bits.mean() - before_bits.mean())
    if criterion == "wilson_lb":
        wins = int(np.sum(after_bits > before_bits))
        losses = int(np.sum(after_bits < before_bits))
        trials = wins + losses
        if trials == 0:
            return False, delta
        phat = wins / trials
        z = 1.96
        denom = 1.0 + (z**2 / trials)
        center = phat + (z**2 / (2.0 * trials))
        margin = z * math.sqrt((phat * (1 - phat) / trials) + (z**2 / (4.0 * trials**2)))
        lower_bound = (center - margin) / denom
        return lower_bound > 0.5 + threshold, delta
    return delta > threshold, delta


def chunked_entropy_update(
    model: nn.Module,
    optimizer: optim.Optimizer,
    images: torch.Tensor,
    device: torch.device,
    microbatch_size: int,
    steps: int,
) -> float:
    configure_model_for_tent(model)
    final_loss = 0.0
    for _ in range(steps):
        for chunk in torch.split(images, microbatch_size):
            optimizer.zero_grad(set_to_none=True)
            logits = model(chunk.to(device, non_blocking=True))
            loss = softmax_entropy(logits).mean()
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu())
    return final_loss


def chunked_dirichlet_entropy_update(
    model: nn.Module,
    optimizer: optim.Optimizer,
    images: torch.Tensor,
    device: torch.device,
    microbatch_size: int,
    steps: int,
    prior_strength: float = 1000.0,
) -> float:
    configure_model_for_tent(model)
    final_loss = 0.0
    for _ in range(steps):
        for chunk in torch.split(images, microbatch_size):
            optimizer.zero_grad(set_to_none=True)
            logits = model(chunk.to(device, non_blocking=True))
            loss = dirichlet_entropy(logits, prior_strength=prior_strength).mean()
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu())
    return final_loss


def chunked_supervised_update(
    model: nn.Module,
    optimizer: optim.Optimizer,
    images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    microbatch_size: int,
    steps: int,
) -> float:
    model.train()
    final_loss = 0.0
    for _ in range(steps):
        for image_chunk, label_chunk in zip(torch.split(images, microbatch_size), torch.split(labels, microbatch_size)):
            optimizer.zero_grad(set_to_none=True)
            logits = model(image_chunk.to(device, non_blocking=True))
            loss = F.cross_entropy(logits, label_chunk.to(device, non_blocking=True))
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu())
    return final_loss


def chunked_negative_label_update(
    model: nn.Module,
    optimizer: optim.Optimizer,
    images: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
    microbatch_size: int,
    steps: int,
    weight: float = 1.0,
) -> float:
    model.train()
    final_loss = 0.0
    for _ in range(steps):
        for image_chunk, label_chunk in zip(torch.split(images, microbatch_size), torch.split(labels, microbatch_size)):
            optimizer.zero_grad(set_to_none=True)
            logits = model(image_chunk.to(device, non_blocking=True))
            probs = torch.softmax(logits, dim=1)
            chosen = probs.gather(1, label_chunk.to(device, non_blocking=True).unsqueeze(1))
            loss = (-torch.log1p(-chosen.clamp(max=1.0 - 1e-6))).mean() * weight
            loss.backward()
            optimizer.step()
            final_loss = float(loss.detach().cpu())
    return final_loss


@dataclass
class QueryBudgetManager:
    query_rate: float
    carry: float = 0.0

    def allocate(self, window_size: int) -> int:
        self.carry += self.query_rate * window_size
        allocation = int(self.carry)
        self.carry -= allocation
        return allocation


class StreamingAdapter:
    def __init__(self, model: nn.Module, device: torch.device, config: dict):
        self.model = model.to(device)
        self.device = device
        self.config = config
        self.microbatch_size = int(config.get("microbatch_size", 256))
        self.base_state = copy.deepcopy(model.state_dict())
        self.safe_state = copy.deepcopy(self.base_state)
        self.reject_streak = 0
        self.optimizer = None

    def reset(self) -> None:
        self.model.load_state_dict(copy.deepcopy(self.base_state))
        self.safe_state = copy.deepcopy(self.base_state)
        self.reject_streak = 0
        self._refresh_optimizer()

    def _refresh_optimizer(self) -> None:
        self.optimizer = None

    def _load_state(self, state_dict) -> None:
        self.model.load_state_dict(copy.deepcopy(state_dict))
        self._refresh_optimizer()

    def process_window(
        self,
        images: torch.Tensor,
        labels: np.ndarray,
        is_id: np.ndarray,
        num_queries: int,
        rng: np.random.RandomState,
    ) -> tuple[np.ndarray, dict]:
        raise NotImplementedError
