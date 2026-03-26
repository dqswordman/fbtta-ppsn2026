from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_device(batch: Iterable[torch.Tensor], device: torch.device) -> Iterable[torch.Tensor]:
    return [tensor.to(device, non_blocking=True) for tensor in batch]


def resolve_existing_path(candidates: list[str | Path]) -> Path:
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    joined = "\n".join(str(Path(candidate)) for candidate in candidates)
    raise FileNotFoundError(f"None of the expected paths exist:\n{joined}")
