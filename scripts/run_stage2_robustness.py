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
from src.experiment.support import attach_frozen_reference, load_source_model
from src.methods import build_adapter
from src.utils.config import load_merged_yaml
from src.utils.io import ensure_dir, save_resolved_config
from src.utils.repro import default_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 2 feedback-noise or delayed-feedback robustness sweeps.")
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def clone_reuse_rows(
    reuse_frame: pd.DataFrame,
    axis_name: str,
    axis_value: float | int,
    study_name: str,
    fixed_feedback_noise: float,
    fixed_feedback_delay_windows: int,
) -> pd.DataFrame:
    if axis_name not in {"feedback_noise", "feedback_delay_windows"}:
        raise ValueError(f"Unsupported robustness axis: {axis_name}")

    frozen_rows = reuse_frame[reuse_frame["method"] == "frozen"].copy()
    method_rows = reuse_frame[reuse_frame["method"] == "fb_gate_tent"].copy()

    if axis_name == "feedback_noise":
        if float(axis_value) == 0.0:
            selected_method_rows = method_rows
        else:
            selected_method_rows = method_rows[method_rows["query_rate"] == 0.0].copy()
        axis_feedback_noise = float(axis_value)
        axis_feedback_delay = int(fixed_feedback_delay_windows)
    else:
        if int(axis_value) == 0:
            selected_method_rows = method_rows
        else:
            selected_method_rows = method_rows[method_rows["query_rate"] == 0.0].copy()
        axis_feedback_noise = float(fixed_feedback_noise)
        axis_feedback_delay = int(axis_value)

    combined = pd.concat([frozen_rows, selected_method_rows], ignore_index=True)
    combined["study"] = study_name
    combined["feedback_noise"] = axis_feedback_noise
    combined["feedback_delay_windows"] = axis_feedback_delay
    combined["robustness_axis"] = axis_name
    combined["robustness_value"] = axis_value
    combined["reused_from_reference"] = 1
    return combined


def main() -> None:
    args = parse_args()
    config = load_merged_yaml(ROOT / "configs" / "common.yaml", ROOT / args.config)
    device = default_device()

    data_root = ROOT / config["paths"]["data_root"]
    dataset_name = str(config.get("dataset", {}).get("name", "cifar10"))
    raw_dir = ensure_dir(ROOT / config["paths"]["results_raw"] / config["study"]["name"])
    processed_dir = ensure_dir(ROOT / config["paths"]["results_processed"])
    save_windows = bool(config.get("runtime", {}).get("save_windows", False))
    window_dir = ensure_dir(raw_dir / "windows") if save_windows else None
    total_length = int(config.get("total_length", 10_000))

    axis_name = str(config["robustness"]["axis"])
    axis_values = list(config["robustness"]["values"])
    fixed_feedback_noise = float(config["robustness"].get("fixed_feedback_noise", 0.0))
    fixed_feedback_delay_windows = int(config["robustness"].get("fixed_feedback_delay_windows", 0))
    reuse_path = ROOT / str(config["robustness"]["reuse_reference_raw"])
    if not reuse_path.exists():
        raise FileNotFoundError(f"Missing robustness reference raw file: {reuse_path}")

    reuse_frame = pd.read_csv(reuse_path)
    expected_checkpoints = {
        str(ROOT / str(config["paths"]["source_checkpoint_pattern"]).format(seed=seed))
        for seed in config["seeds"]
    }
    if "dataset_name" in reuse_frame.columns:
        reuse_frame = reuse_frame[reuse_frame["dataset_name"] == dataset_name].copy()
    if "corruption" in reuse_frame.columns:
        reuse_frame = reuse_frame[reuse_frame["corruption"].isin(config["corruptions"])].copy()
    if "severity" in reuse_frame.columns:
        reuse_frame = reuse_frame[reuse_frame["severity"].isin([int(value) for value in config["severities"]])].copy()
    if "contamination_ratio" in reuse_frame.columns:
        reuse_frame = reuse_frame[
            reuse_frame["contamination_ratio"].isin([float(value) for value in config["contamination_ratios"]])
        ].copy()
    if "query_rate" in reuse_frame.columns:
        reuse_frame = reuse_frame[reuse_frame["query_rate"].isin([float(value) for value in config["query_rates"]])].copy()
    if "source_checkpoint" in reuse_frame.columns:
        reuse_frame = reuse_frame[reuse_frame["source_checkpoint"].isin(expected_checkpoints)].copy()
    summary_rows: list[dict] = []

    for axis_value in axis_values:
        reused_rows = clone_reuse_rows(
            reuse_frame=reuse_frame,
            axis_name=axis_name,
            axis_value=axis_value,
            study_name=str(config["study"]["name"]),
            fixed_feedback_noise=fixed_feedback_noise,
            fixed_feedback_delay_windows=fixed_feedback_delay_windows,
        )
        summary_rows.extend(reused_rows.to_dict(orient="records"))

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

                    for axis_value in axis_values:
                        is_baseline_axis = (axis_name == "feedback_noise" and float(axis_value) == 0.0) or (
                            axis_name == "feedback_delay_windows" and int(axis_value) == 0
                        )
                        feedback_noise = float(axis_value) if axis_name == "feedback_noise" else fixed_feedback_noise
                        feedback_delay_windows = int(axis_value) if axis_name == "feedback_delay_windows" else fixed_feedback_delay_windows

                        for query_rate in [float(value) for value in config["query_rates"] if float(value) > 0.0]:
                            if is_baseline_axis:
                                continue

                            model = load_source_model(checkpoint, device)
                            method_cfg = copy.deepcopy(config["methods"]["fb_gate_tent"])
                            method_cfg["microbatch_size"] = int(config["runtime"]["microbatch_size"])
                            method_cfg["feedback_noise"] = feedback_noise
                            method_cfg["feedback_delay_windows"] = feedback_delay_windows
                            adapter = build_adapter("fb_gate_tent", model, device, method_cfg)

                            metadata = {
                                "study": config["study"]["name"],
                                "dataset_name": dataset_name,
                                "method": "fb_gate_tent",
                                "corruption": corruption,
                                "severity": int(severity),
                                "contamination_ratio": float(ratio),
                                "source_checkpoint": str(checkpoint),
                                "feedback_noise": feedback_noise,
                                "feedback_delay_windows": feedback_delay_windows,
                                "robustness_axis": axis_name,
                                "robustness_value": axis_value,
                                "reused_from_reference": 0,
                            }
                            summary, windows = run_stream_experiment(
                                adapter=adapter,
                                images_uint8=mixed_images,
                                labels=mixed_labels,
                                is_id=mixed_is_id,
                                metadata=metadata,
                                query_rate=query_rate,
                                seed=ratio_seed,
                                window_size=int(config["runtime"]["window_size"]),
                                dataset_name=dataset_name,
                            )
                            summary_rows.append(summary)

                            if save_windows and window_dir is not None:
                                run_name = (
                                    f"{corruption}_sev{severity}_ratio{ratio:.2f}_{axis_name}_{axis_value}_"
                                    f"q{query_rate:.3f}_seed{seed}"
                                )
                                windows.to_csv(window_dir / f"{run_name}.csv", index=False)

    raw_df = pd.DataFrame(summary_rows)
    raw_df = raw_df.sort_values(
        [
            "robustness_axis",
            "robustness_value",
            "corruption",
            "severity",
            "contamination_ratio",
            "method",
            "query_rate",
            "seed",
        ]
    ).reset_index(drop=True)
    raw_df.to_csv(raw_dir / "robustness_runs.csv", index=False)
    save_resolved_config(raw_dir, config)

    annotated = attach_frozen_reference(
        raw_df,
        key_columns=[
            "corruption",
            "severity",
            "contamination_ratio",
            "seed",
            "query_rate",
            "feedback_noise",
            "feedback_delay_windows",
        ],
    )
    annotated.to_csv(processed_dir / f"{config['study']['name']}_annotated.csv", index=False)

    summary = (
        annotated.groupby(
            [
                "method",
                "robustness_axis",
                "robustness_value",
                "feedback_noise",
                "feedback_delay_windows",
                "contamination_ratio",
                "query_rate",
            ]
        )
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
    print({"raw": str(raw_dir / "robustness_runs.csv"), "processed": str(processed_dir / f"{config['study']['name']}_summary.csv")})


if __name__ == "__main__":
    main()
