from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from src.models.cifar_resnet import build_cifar_resnet18


CONSTANT_QUERY_METHODS = {"frozen", "tent", "sar_or_come_zero_bit_stable"}


def load_source_model(checkpoint_path: str | Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint["model_state"] if "model_state" in checkpoint else checkpoint
    num_classes = int(checkpoint.get("num_classes", state_dict["fc.weight"].shape[0]))
    model = build_cifar_resnet18(num_classes=num_classes)
    model.load_state_dict(state_dict)
    return model.to(device)


def attach_frozen_reference(frame: pd.DataFrame, key_columns: list[str]) -> pd.DataFrame:
    frozen = frame[frame["method"] == "frozen"][key_columns + ["id_accuracy", "ece"]].rename(
        columns={
            "id_accuracy": "frozen_id_accuracy",
            "ece": "frozen_ece",
        }
    )
    merged = frame.merge(frozen, on=key_columns, how="left")
    merged["gain_over_frozen"] = merged["id_accuracy"] - merged["frozen_id_accuracy"]
    merged["negative_transfer"] = (merged["id_accuracy"] < merged["frozen_id_accuracy"]).astype(int)
    merged["ece_delta_vs_frozen"] = merged["ece"] - merged["frozen_ece"]
    return merged


def grouped_mean_std(frame: pd.DataFrame, group_columns: list[str], value_columns: list[str]) -> pd.DataFrame:
    aggregate_map = {}
    for column in value_columns:
        aggregate_map[column] = ["mean", "std"]
    grouped = frame.groupby(group_columns).agg(aggregate_map)
    grouped.columns = [f"{name}_{stat}" for name, stat in grouped.columns]
    return grouped.reset_index()
