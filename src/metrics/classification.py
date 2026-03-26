from __future__ import annotations

from typing import Iterable

import numpy as np


def _as_numpy(value: np.ndarray | Iterable[float]) -> np.ndarray:
    return np.asarray(value)


def accuracy_from_probs(probs: np.ndarray, labels: np.ndarray) -> float:
    predictions = probs.argmax(axis=1)
    return float((predictions == labels).mean())


def error_from_probs(probs: np.ndarray, labels: np.ndarray) -> float:
    return 1.0 - accuracy_from_probs(probs, labels)


def brier_score(probs: np.ndarray, labels: np.ndarray, num_classes: int | None = None) -> float:
    if num_classes is None:
        num_classes = int(probs.shape[1])
    one_hot = np.eye(num_classes, dtype=np.float32)[labels]
    return float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    num_bins: int = 15,
) -> float:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == labels).astype(np.float32)
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    ece = 0.0
    for left, right in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (confidences >= left) & (confidences < right if right < 1.0 else confidences <= right)
        if not mask.any():
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += float(mask.mean()) * abs(float(bin_acc - bin_conf))
    return float(ece)


def overconfident_error_mass(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.9,
) -> float:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    wrong = predictions != labels
    return float(np.mean(wrong & (confidences >= threshold)))


def summarize_stream_metrics(
    probs: np.ndarray,
    labels: np.ndarray,
    is_id: np.ndarray,
    window_stats: list[dict],
    ece_bins: int = 15,
) -> dict:
    id_probs = probs[is_id]
    id_labels = labels[is_id]
    if len(id_labels) == 0:
        raise ValueError("A stream summary requires at least one ID sample.")

    accepted_windows = [row.get("accepted", 0) for row in window_stats if row.get("candidate_evaluated", 0)]
    accepted_fraction = float(np.mean(accepted_windows)) if accepted_windows else 0.0
    max_class_shares = [row.get("max_class_share", 0.0) for row in window_stats]
    collapse = any(share >= 0.98 for share in max_class_shares)
    total_queries = int(sum(int(row.get("num_queries", 0)) for row in window_stats))
    total_ood_queries = int(sum(int(row.get("num_ood_queries", 0)) for row in window_stats))

    return {
        "id_accuracy": accuracy_from_probs(id_probs, id_labels),
        "id_error": error_from_probs(id_probs, id_labels),
        "ece": expected_calibration_error(id_probs, id_labels, num_bins=ece_bins),
        "brier": brier_score(id_probs, id_labels),
        "oem_0.9": overconfident_error_mass(id_probs, id_labels, threshold=0.9),
        "accepted_update_fraction": accepted_fraction,
        "collapse": int(collapse),
        "query_count": total_queries,
        "ood_query_count": total_ood_queries,
    }
