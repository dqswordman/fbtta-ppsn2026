from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.cifar import load_cifar10c_corruption
from src.data.streams import build_contamination_stream
from src.data.svhn import sample_svhn_images
from src.experiment.runner import run_stream_experiment
from src.experiment.support import load_source_model
from src.methods import build_adapter
from src.utils.config import load_merged_yaml
from src.utils.io import ensure_dir, save_resolved_config
from src.utils.repro import default_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FB-Gate-Tent ablations.")
    parser.add_argument("--config", default="configs/ablations.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_merged_yaml(ROOT / "configs" / "common.yaml", ROOT / args.config)
    device = default_device()

    data_root = ROOT / config["paths"]["data_root"]
    raw_dir = ensure_dir(ROOT / config["paths"]["results_raw"] / config["study"]["name"])
    processed_dir = ensure_dir(ROOT / config["paths"]["results_processed"])
    save_windows = bool(config.get("runtime", {}).get("save_windows", True))
    window_dir = ensure_dir(raw_dir / "windows") if save_windows else None

    base_method_cfg = copy.deepcopy(config["methods"]["fb_gate_tent"])
    records: list[dict] = []

    ablation_grid = []
    ablation_grid.extend([("query_strategy", value) for value in config["query_strategy_values"]])
    ablation_grid.extend([("gate_threshold", value) for value in config["gate_threshold_values"]])
    ablation_grid.extend([("gate_criterion", value) for value in config["gate_criterion_values"]])
    ablation_grid.extend([("feedback_noise", value) for value in config["feedback_noise_values"]])
    ablation_grid.extend([("rollback", value) for value in config["rollback_values"]])

    for corruption in config["corruptions"]:
        for severity in config["severities"]:
            id_images, id_labels = load_cifar10c_corruption(data_root, corruption=corruption, severity=int(severity))
            for seed in config["seeds"]:
                checkpoint = ROOT / str(config["paths"]["source_checkpoint_pattern"]).format(seed=seed)
                if not checkpoint.exists():
                    raise FileNotFoundError(f"Missing source checkpoint: {checkpoint}")

                ratio = float(config["contamination_ratio"])
                ratio_seed = int(seed * 1000 + round(ratio * 100))
                n_ood = int(round(10_000 * ratio))
                ood_images = sample_svhn_images(data_root, count=max(n_ood, 1), seed=ratio_seed) if n_ood else id_images[:1]
                mixed_images, mixed_labels, mixed_is_id = build_contamination_stream(
                    id_images=id_images,
                    id_labels=id_labels,
                    ood_images=ood_images,
                    ratio=ratio,
                    seed=ratio_seed,
                    total_length=10_000,
                )

                for ablation_name, ablation_value in ablation_grid:
                    method_cfg = copy.deepcopy(base_method_cfg)
                    method_cfg[ablation_name] = ablation_value
                    method_cfg["microbatch_size"] = int(config["runtime"]["microbatch_size"])

                    model = load_source_model(checkpoint, device)
                    adapter = build_adapter("fb_gate_tent", model, device, method_cfg)
                    metadata = {
                        "study": config["study"]["name"],
                        "method": "fb_gate_tent",
                        "corruption": corruption,
                        "severity": int(severity),
                        "contamination_ratio": ratio,
                        "ablation_name": ablation_name,
                        "ablation_value": ablation_value,
                        "source_checkpoint": str(checkpoint),
                    }
                    summary, windows = run_stream_experiment(
                        adapter=adapter,
                        images_uint8=mixed_images,
                        labels=mixed_labels,
                        is_id=mixed_is_id,
                        metadata=metadata,
                        query_rate=float(config["query_rate"]),
                        seed=ratio_seed,
                        window_size=int(config["runtime"]["window_size"]),
                    )
                    records.append(summary)
                    run_name = f"{corruption}_sev{severity}_{ablation_name}_{ablation_value}_seed{seed}"
                    if save_windows and window_dir is not None:
                        windows.to_csv(window_dir / f"{run_name}.csv", index=False)

    raw_df = pd.DataFrame(records)
    raw_df.to_csv(raw_dir / "ablation_runs.csv", index=False)
    save_resolved_config(raw_dir, config)

    summary = (
        raw_df.groupby(["ablation_name", "ablation_value"])
        .agg(
            id_accuracy_mean=("id_accuracy", "mean"),
            id_accuracy_std=("id_accuracy", "std"),
            ece_mean=("ece", "mean"),
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
    print({"raw": str(raw_dir / "ablation_runs.csv"), "processed": str(processed_dir / f"{config['study']['name']}_summary.csv")})


if __name__ == "__main__":
    main()
