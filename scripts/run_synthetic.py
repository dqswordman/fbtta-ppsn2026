from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.synthetic import (
    SyntheticConfig,
    frozen_weights,
    generate_dual_environment,
    predict_linear,
    select_queries,
    zero_bit_candidate,
)
from src.utils.config import load_yaml
from src.utils.io import ensure_dir, save_resolved_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the synthetic separation study.")
    parser.add_argument("--config", default="configs/synthetic.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    synth_params = dict(config["synthetic"])
    feedback_noise_values = synth_params.pop("feedback_noise_values")
    synth_cfg = SyntheticConfig(**synth_params)
    study_name = config["study"]["name"]

    raw_dir = ensure_dir(ROOT / "results" / "raw" / study_name)
    processed_dir = ensure_dir(ROOT / "results" / "processed")

    rows = []
    for feedback_noise in feedback_noise_values:
        for env_sign in config["env_signs"]:
            env_name = "env_plus" if env_sign > 0 else "env_minus"
            for seed in config["seeds"]:
                features, labels = generate_dual_environment(env_sign=env_sign, seed=seed, config=synth_cfg)
                frozen_pred = predict_linear(features, frozen_weights())
                candidate_weights = zero_bit_candidate(step=synth_cfg.candidate_step)
                zero_pred = predict_linear(features, candidate_weights)

                frozen_acc = float((frozen_pred == labels).mean())
                zero_acc = float((zero_pred == labels).mean())

                for query_rate in config["query_rates"]:
                    gate_stats = select_queries(
                        features,
                        labels,
                        weights_before=frozen_weights(),
                        weights_after=candidate_weights,
                        query_rate=float(query_rate),
                        seed=seed + int(query_rate * 1_000_000) + 17 * (1 if env_sign > 0 else 2),
                        feedback_noise=float(feedback_noise),
                    )
                    accepted = bool(gate_stats["accept"])
                    fb_acc = zero_acc if accepted else frozen_acc

                    common = {
                        "env_name": env_name,
                        "env_sign": env_sign,
                        "seed": seed,
                        "query_rate": float(query_rate),
                        "feedback_noise": float(feedback_noise),
                    }
                    rows.extend(
                        [
                            {
                                **common,
                                "method": "frozen",
                                "accuracy": frozen_acc,
                                "accepted": 0,
                                "n_queries": 0,
                            },
                            {
                                **common,
                                "method": "zero_bit_update",
                                "accuracy": zero_acc,
                                "accepted": 1,
                                "n_queries": 0,
                            },
                            {
                                **common,
                                "method": "fb_gate",
                                "accuracy": fb_acc,
                                "accepted": int(accepted),
                                "n_queries": int(gate_stats["n_queries"]),
                                "gate_delta": float(gate_stats["delta"]),
                            },
                        ]
                    )

    raw_df = pd.DataFrame(rows)
    raw_df.to_csv(raw_dir / f"{study_name}_runs.csv", index=False)
    save_resolved_config(raw_dir, config)

    merge_keys = ["env_name", "seed", "query_rate", "feedback_noise"]
    frozen = raw_df[raw_df["method"] == "frozen"][merge_keys + ["accuracy"]].rename(columns={"accuracy": "frozen_accuracy"})
    annotated = raw_df.merge(frozen, on=merge_keys, how="left")
    annotated["gain_over_frozen"] = annotated["accuracy"] - annotated["frozen_accuracy"]
    annotated["negative_transfer"] = (annotated["accuracy"] < annotated["frozen_accuracy"]).astype(int)
    annotated.to_csv(processed_dir / f"{study_name}_annotated.csv", index=False)

    summary = (
        annotated.groupby(["env_name", "method", "query_rate", "feedback_noise"])
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            gain_mean=("gain_over_frozen", "mean"),
            gain_std=("gain_over_frozen", "std"),
            negative_transfer_rate=("negative_transfer", "mean"),
            accept_rate=("accepted", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(processed_dir / f"{study_name}_summary.csv", index=False)

    table = summary[summary["feedback_noise"] == 0.0].pivot_table(
        index=["env_name", "method"],
        columns="query_rate",
        values=["accuracy_mean", "negative_transfer_rate"],
    )
    table.to_csv(processed_dir / f"{study_name}_table.csv")
    print({"raw": str(raw_dir / f"{study_name}_runs.csv"), "processed": str(processed_dir / f"{study_name}_summary.csv")})


if __name__ == "__main__":
    main()
