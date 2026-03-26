from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.cifar import load_cifar_c_corruption
from src.data.streams import build_contamination_stream
from src.data.svhn import sample_svhn_images
from src.experiment.runner import run_stream_experiment
from src.experiment.support import CONSTANT_QUERY_METHODS, attach_frozen_reference, load_source_model
from src.methods import build_adapter
from src.utils.config import load_merged_yaml
from src.utils.io import ensure_dir, save_resolved_config
from src.utils.repro import default_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CIFAR-10-C + SVHN contamination study.")
    parser.add_argument("--config", default="configs/contamination.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_merged_yaml(ROOT / "configs" / "common.yaml", ROOT / args.config)
    device = default_device()

    data_root = ROOT / config["paths"]["data_root"]
    dataset_name = str(config.get("dataset", {}).get("name", "cifar10"))
    raw_dir = ensure_dir(ROOT / config["paths"]["results_raw"] / config["study"]["name"])
    processed_dir = ensure_dir(ROOT / config["paths"]["results_processed"])
    save_windows = bool(config.get("runtime", {}).get("save_windows", True))
    window_dir = ensure_dir(raw_dir / "windows") if save_windows else None
    query_rates = [float(value) for value in config["query_rates"]]
    total_length = int(config.get("total_length", 10_000))

    summary_rows: list[dict] = []

    for corruption in config["corruptions"]:
        for severity in config["severities"]:
            id_images, id_labels = load_cifar_c_corruption(
                data_root,
                dataset_name=dataset_name,
                corruption=corruption,
                severity=int(severity),
            )
            for seed in config["seeds"]:
                checkpoint = ROOT / str(config["paths"]["source_checkpoint_pattern"]).format(seed=seed)
                if not checkpoint.exists():
                    raise FileNotFoundError(f"Missing source checkpoint: {checkpoint}")

                for ratio in config["contamination_ratios"]:
                    ratio_seed = int(seed * 1000 + round(float(ratio) * 100))
                    n_ood = int(round(total_length * float(ratio)))
                    ood_images = sample_svhn_images(data_root, count=max(n_ood, 1), seed=ratio_seed) if n_ood else id_images[:1]
                    mixed_images, mixed_labels, mixed_is_id = build_contamination_stream(
                        id_images=id_images,
                        id_labels=id_labels,
                        ood_images=ood_images,
                        ratio=float(ratio),
                        seed=ratio_seed,
                        total_length=total_length,
                    )

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
                                "contamination_ratio": float(ratio),
                                "source_checkpoint": str(checkpoint),
                            }
                            summary, windows = run_stream_experiment(
                                adapter=adapter,
                                images_uint8=mixed_images,
                                labels=mixed_labels,
                                is_id=mixed_is_id,
                                metadata=metadata,
                                query_rate=float(query_rate),
                                seed=ratio_seed,
                                window_size=int(config["runtime"]["window_size"]),
                                dataset_name=dataset_name,
                            )
                            run_name = f"{corruption}_sev{severity}_ratio{ratio:.2f}_{method}_q{query_rate:.3f}_seed{seed}"
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
    raw_df = raw_df.sort_values(
        ["corruption", "severity", "contamination_ratio", "method", "query_rate", "seed"]
    ).reset_index(drop=True)
    raw_df.to_csv(raw_dir / "contamination_runs.csv", index=False)
    save_resolved_config(raw_dir, config)

    annotated = attach_frozen_reference(
        raw_df,
        key_columns=["corruption", "severity", "contamination_ratio", "seed", "query_rate"],
    )
    annotated.to_csv(processed_dir / f"{config['study']['name']}_annotated.csv", index=False)

    summary = (
        annotated.groupby(["method", "contamination_ratio", "query_rate"])
        .agg(
            id_accuracy_mean=("id_accuracy", "mean"),
            id_accuracy_std=("id_accuracy", "std"),
            gain_mean=("gain_over_frozen", "mean"),
            gain_std=("gain_over_frozen", "std"),
            gain_worst=("gain_over_frozen", "min"),
            negative_transfer_rate=("negative_transfer", "mean"),
            ece_mean=("ece", "mean"),
            brier_mean=("brier", "mean"),
            oem_0_9_mean=("oem_0.9", "mean"),
            collapse_rate=("collapse", "mean"),
            accept_rate_mean=("accepted_update_fraction", "mean"),
            runtime_sec_mean=("runtime_sec", "mean"),
            peak_memory_mb_mean=("peak_memory_mb", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(processed_dir / f"{config['study']['name']}_summary.csv", index=False)
    print({"raw": str(raw_dir / "contamination_runs.csv"), "processed": str(processed_dir / f"{config['study']['name']}_summary.csv")})


if __name__ == "__main__":
    main()
