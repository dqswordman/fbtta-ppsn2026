from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.cifar import load_cifar_c_corruption
from src.experiment.runner import run_stream_experiment
from src.experiment.support import CONSTANT_QUERY_METHODS, attach_frozen_reference, load_source_model
from src.methods import build_adapter
from src.utils.config import load_merged_yaml
from src.utils.io import ensure_dir, save_resolved_config
from src.utils.repro import default_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CIFAR-10-C frontier study.")
    parser.add_argument("--config", default="configs/frontier_final.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_merged_yaml(ROOT / "configs" / "common.yaml", ROOT / args.config)
    device = default_device()

    data_root = ROOT / config["paths"]["data_root"]
    dataset_name = str(config.get("dataset", {}).get("name", "cifar10"))
    raw_dir = ensure_dir(ROOT / config["paths"]["results_raw"] / config["study"]["name"])
    processed_dir = ensure_dir(ROOT / config["paths"]["results_processed"])
    query_rates = [float(value) for value in config["query_rates"]]
    save_windows = bool(config.get("runtime", {}).get("save_windows", True))

    summary_rows: list[dict] = []
    window_dir = ensure_dir(raw_dir / "windows") if save_windows else None

    for corruption in config["corruptions"]:
        for severity in config["severities"]:
            images, labels = load_cifar_c_corruption(
                data_root,
                dataset_name=dataset_name,
                corruption=corruption,
                severity=int(severity),
            )
            is_id = (labels >= 0)
            for seed in config["seeds"]:
                checkpoint = ROOT / str(config["paths"]["source_checkpoint_pattern"]).format(seed=seed)
                if not checkpoint.exists():
                    raise FileNotFoundError(f"Missing source checkpoint: {checkpoint}")

                for method in config["method_names"]:
                    effective_query_rates = [0.0] if method in CONSTANT_QUERY_METHODS else query_rates
                    cached_summary = None
                    for query_rate in effective_query_rates:
                        model = load_source_model(checkpoint, device)
                        method_cfg = copy.deepcopy(config["methods"][method])
                        method_cfg["microbatch_size"] = int(config["runtime"]["microbatch_size"])
                        adapter = build_adapter(method, model, device, method_cfg)

                        metadata = {
                            "study": config["study"]["name"],
                            "dataset_name": dataset_name,
                            "method": method,
                            "corruption": corruption,
                            "severity": int(severity),
                            "source_checkpoint": str(checkpoint),
                        }
                        summary, windows = run_stream_experiment(
                            adapter=adapter,
                            images_uint8=images,
                            labels=labels,
                            is_id=is_id,
                            metadata=metadata,
                            query_rate=float(query_rate),
                            seed=int(seed),
                            window_size=int(config["runtime"]["window_size"]),
                            dataset_name=dataset_name,
                        )

                        run_name = f"{corruption}_sev{severity}_{method}_q{query_rate:.3f}_seed{seed}"
                        if save_windows and window_dir is not None:
                            windows.to_csv(window_dir / f"{run_name}.csv", index=False)
                        summary_rows.append(summary)
                        cached_summary = summary

                    if method in CONSTANT_QUERY_METHODS and cached_summary is not None:
                        for query_rate in query_rates[1:]:
                            clone = dict(cached_summary)
                            clone["query_rate"] = float(query_rate)
                            clone["actual_query_rate"] = 0.0
                            clone["reused_constant_run"] = 1
                            summary_rows.append(clone)

    raw_df = pd.DataFrame(summary_rows)
    raw_df = raw_df.sort_values(["corruption", "severity", "method", "query_rate", "seed"]).reset_index(drop=True)
    raw_df.to_csv(raw_dir / "frontier_runs.csv", index=False)
    save_resolved_config(raw_dir, config)

    annotated = attach_frozen_reference(
        raw_df,
        key_columns=["corruption", "severity", "seed", "query_rate"],
    )
    annotated.to_csv(processed_dir / f"{config['study']['name']}_annotated.csv", index=False)

    summary = (
        annotated.groupby(["method", "query_rate"])
        .agg(
            id_accuracy_mean=("id_accuracy", "mean"),
            id_accuracy_std=("id_accuracy", "std"),
            gain_mean=("gain_over_frozen", "mean"),
            gain_std=("gain_over_frozen", "std"),
            gain_worst=("gain_over_frozen", "min"),
            negative_transfer_rate=("negative_transfer", "mean"),
            ece_mean=("ece", "mean"),
            ece_std=("ece", "std"),
            brier_mean=("brier", "mean"),
            oem_0_9_mean=("oem_0.9", "mean"),
            accept_rate_mean=("accepted_update_fraction", "mean"),
            collapse_rate=("collapse", "mean"),
            runtime_sec_mean=("runtime_sec", "mean"),
            peak_memory_mb_mean=("peak_memory_mb", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(processed_dir / f"{config['study']['name']}_summary.csv", index=False)

    low_feedback = summary[summary["query_rate"].isin([0.001, 0.005, 0.01])].copy()
    low_feedback.to_csv(processed_dir / f"{config['study']['name']}_low_feedback.csv", index=False)
    print({"raw": str(raw_dir / "frontier_runs.csv"), "processed": str(processed_dir / f"{config['study']['name']}_summary.csv")})


if __name__ == "__main__":
    main()
