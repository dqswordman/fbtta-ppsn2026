from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.cifar import numpy_images_to_tensor
from src.data.streams import iter_windows
from src.metrics.classification import summarize_stream_metrics
from src.methods.common import QueryBudgetManager
from src.utils.repro import set_global_seed


def run_stream_experiment(
    adapter,
    images_uint8: np.ndarray,
    labels: np.ndarray,
    is_id: np.ndarray,
    metadata: dict,
    query_rate: float,
    seed: int,
    window_size: int,
    dataset_name: str = "cifar10",
) -> tuple[dict, pd.DataFrame]:
    set_global_seed(seed)
    adapter.reset()
    rng = np.random.RandomState(seed)
    budget = QueryBudgetManager(query_rate=float(query_rate))

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(adapter.device)

    start = time.time()
    images_tensor = numpy_images_to_tensor(images_uint8, dataset_name=dataset_name).contiguous()
    all_probs: list[np.ndarray] = []
    window_rows: list[dict] = []

    offset = 0
    for batch in iter_windows(images_uint8, labels, is_id, window_size=window_size):
        batch_size = len(batch.labels)
        image_slice = images_tensor[offset : offset + batch_size]
        num_queries = budget.allocate(batch_size)
        probs, stats = adapter.process_window(
            image_slice,
            batch.labels,
            batch.is_id,
            num_queries=num_queries,
            rng=rng,
        )
        stats["window_index"] = batch.index
        stats["window_start"] = offset
        stats["window_end"] = offset + batch_size
        window_rows.append(stats)
        all_probs.append(probs)
        offset += batch_size

    elapsed = time.time() - start
    probabilities = np.concatenate(all_probs, axis=0)
    summary = summarize_stream_metrics(probabilities, labels, is_id, window_rows)
    summary.update(metadata)
    summary.update(
        {
            "query_rate": float(query_rate),
            "seed": int(seed),
            "runtime_sec": elapsed,
            "peak_memory_mb": float(torch.cuda.max_memory_allocated(adapter.device) / (1024**2)) if torch.cuda.is_available() else 0.0,
            "actual_query_rate": float(summary["query_count"] / len(labels)),
        }
    )
    return summary, pd.DataFrame(window_rows)
