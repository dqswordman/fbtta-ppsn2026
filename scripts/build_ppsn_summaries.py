from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import ensure_dir


METHOD_ORDER = [
    "frozen",
    "tent",
    "fb_gate_tent",
    "self_gate_tent_zero_bit",
    "throttle_tent_zero_bit",
    "sar_or_come_zero_bit_stable",
    "bitta_style_binary_feedback",
    "atta_or_simatta_style_active",
    "eatta_style_low_label",
    "full_label_active_reference",
]

METHOD_LABELS = {
    "frozen": "Frozen",
    "tent": "Tent",
    "fb_gate_tent": "FB-Gate-Tent",
    "self_gate_tent_zero_bit": "Self-Gate Tent (0-bit)",
    "throttle_tent_zero_bit": "Throttle Tent (0-bit)",
    "sar_or_come_zero_bit_stable": "COME-style 0-bit",
    "bitta_style_binary_feedback": "BiTTA-style",
    "atta_or_simatta_style_active": "ATTA-style",
    "eatta_style_low_label": "EATTA-style",
    "full_label_active_reference": "Full-label active",
}

SUPERVISION_TYPE = {
    "frozen": "0-bit",
    "tent": "0-bit",
    "self_gate_tent_zero_bit": "0-bit",
    "throttle_tent_zero_bit": "0-bit",
    "sar_or_come_zero_bit_stable": "0-bit",
    "fb_gate_tent": "binary correctness events",
    "bitta_style_binary_feedback": "binary correctness events",
    "atta_or_simatta_style_active": "label queries",
    "eatta_style_low_label": "label queries",
    "full_label_active_reference": "label queries",
}

EVENT_MULTIPLIER = {
    "frozen": 0.0,
    "tent": 0.0,
    "self_gate_tent_zero_bit": 0.0,
    "throttle_tent_zero_bit": 0.0,
    "sar_or_come_zero_bit_stable": 0.0,
    "fb_gate_tent": 2.0,
    "bitta_style_binary_feedback": 1.0,
    "atta_or_simatta_style_active": 1.0,
    "eatta_style_low_label": 1.0,
    "full_label_active_reference": 1.0,
}

PRIMARY_METHODS = [
    "frozen",
    "tent",
    "fb_gate_tent",
    "self_gate_tent_zero_bit",
    "throttle_tent_zero_bit",
    "sar_or_come_zero_bit_stable",
    "bitta_style_binary_feedback",
    "eatta_style_low_label",
]


def save_table(numeric: pd.DataFrame, pretty: pd.DataFrame, stem: str, out_root: Path) -> None:
    numeric.to_csv(out_root / f"{stem}.csv", index=False)
    (out_root / f"{stem}.tex").write_text(pretty.to_latex(index=False, escape=False), encoding="utf-8")


def method_rank(method: str) -> int:
    return METHOD_ORDER.index(method)


def load_primary_dataset(processed_path: Path, raw_path: Path, dataset_label: str) -> pd.DataFrame:
    processed = pd.read_csv(processed_path)
    raw = pd.read_csv(raw_path)

    processed = processed[
        (processed["query_rate"] == 0.01)
        & (processed["contamination_ratio"] == 0.5)
        & (processed["method"].isin(METHOD_ORDER))
    ].copy()

    raw = raw[
        (raw["query_rate"] == 0.01)
        & (raw["contamination_ratio"] == 0.5)
        & (raw["method"].isin(METHOD_ORDER))
    ].copy()

    query_stats = (
        raw.groupby("method", as_index=False)
        .agg(
            actual_query_rate_mean=("actual_query_rate", "mean"),
            query_count_mean=("query_count", "mean"),
        )
    )

    frame = processed.merge(query_stats, on="method", how="left")
    frame["dataset"] = dataset_label
    frame["method_label"] = frame["method"].map(METHOD_LABELS)
    frame["supervision_type"] = frame["method"].map(SUPERVISION_TYPE)
    frame["supervision_events_per_1000"] = (
        frame["actual_query_rate_mean"].fillna(0.0) * 1000.0 * frame["method"].map(EVENT_MULTIPLIER)
    )
    frame["method_rank"] = frame["method"].map(method_rank)
    return frame.sort_values("method_rank").reset_index(drop=True)


def build_primary_cost_table(out_root: Path) -> None:
    c10 = load_primary_dataset(
        ROOT / "results/processed/contamination_stage2_final_cifar10_summary.csv",
        ROOT / "results/raw/contamination_stage2_final_cifar10/contamination_runs.csv",
        "CIFAR-10-C + SVHN",
    )
    c100 = load_primary_dataset(
        ROOT / "results/processed/contamination_stage2_final_cifar100_summary.csv",
        ROOT / "results/raw/contamination_stage2_final_cifar100/contamination_runs.csv",
        "CIFAR-100-C + SVHN",
    )
    numeric = pd.concat([c10, c100], ignore_index=True)
    numeric = numeric[numeric["method"].isin(PRIMARY_METHODS)].copy()

    pretty = numeric[
        [
            "dataset",
            "method_label",
            "supervision_type",
            "supervision_events_per_1000",
            "gain_mean",
            "negative_transfer_rate",
            "gain_worst",
            "accept_rate_mean",
        ]
    ].copy()
    pretty["Events / 1k"] = pretty["supervision_events_per_1000"].map(lambda value: f"{value:.1f}")
    pretty["Gain"] = pretty["gain_mean"].map(lambda value: f"{value:.3f}")
    pretty["NTR"] = pretty["negative_transfer_rate"].map(lambda value: f"{value:.3f}")
    pretty["Worst gain"] = pretty["gain_worst"].map(lambda value: f"{value:.3f}")
    pretty["Accept"] = pretty["accept_rate_mean"].map(lambda value: f"{value:.3f}")
    pretty = pretty[
        ["dataset", "method_label", "supervision_type", "Events / 1k", "Gain", "NTR", "Worst gain", "Accept"]
    ]
    pretty.columns = ["Dataset", "Method", "Supervision", "Events / 1k", "Gain", "NTR", "Worst gain", "Accept"]
    save_table(numeric, pretty, "ppsn_supervision_cost_primary", out_root)


def build_family_summary(out_root: Path) -> None:
    frontier_c10 = pd.read_csv(ROOT / "results/processed/frontier_stage2_final_cifar10_summary.csv")
    frontier_c100 = pd.read_csv(ROOT / "results/processed/frontier_stage2_final_cifar100_summary.csv")
    contam_c10 = pd.read_csv(ROOT / "results/processed/contamination_stage2_final_cifar10_summary.csv")
    contam_c100 = pd.read_csv(ROOT / "results/processed/contamination_stage2_final_cifar100_summary.csv")
    raw_c10 = pd.read_csv(ROOT / "results/raw/contamination_stage2_final_cifar10/contamination_runs.csv")
    raw_c100 = pd.read_csv(ROOT / "results/raw/contamination_stage2_final_cifar100/contamination_runs.csv")

    frontier_c10 = frontier_c10[frontier_c10["query_rate"] == 0.01][["method", "gain_mean", "negative_transfer_rate"]].copy()
    frontier_c10 = frontier_c10.rename(
        columns={"gain_mean": "c10_frontier_gain", "negative_transfer_rate": "c10_frontier_ntr"}
    )
    frontier_c100 = frontier_c100[frontier_c100["query_rate"] == 0.01][["method", "gain_mean", "negative_transfer_rate"]].copy()
    frontier_c100 = frontier_c100.rename(
        columns={"gain_mean": "c100_frontier_gain", "negative_transfer_rate": "c100_frontier_ntr"}
    )
    contam_c10 = contam_c10[
        (contam_c10["query_rate"] == 0.01) & (contam_c10["contamination_ratio"] == 0.5)
    ][["method", "gain_mean", "negative_transfer_rate"]].copy()
    contam_c10 = contam_c10.rename(
        columns={"gain_mean": "c10_contam_gain", "negative_transfer_rate": "c10_contam_ntr"}
    )
    contam_c100 = contam_c100[
        (contam_c100["query_rate"] == 0.01) & (contam_c100["contamination_ratio"] == 0.5)
    ][["method", "gain_mean", "negative_transfer_rate"]].copy()
    contam_c100 = contam_c100.rename(
        columns={"gain_mean": "c100_contam_gain", "negative_transfer_rate": "c100_contam_ntr"}
    )

    raw = pd.concat(
        [
            raw_c10[(raw_c10["query_rate"] == 0.01) & (raw_c10["contamination_ratio"] == 0.5)][
                ["method", "actual_query_rate"]
            ],
            raw_c100[(raw_c100["query_rate"] == 0.01) & (raw_c100["contamination_ratio"] == 0.5)][
                ["method", "actual_query_rate"]
            ],
        ],
        ignore_index=True,
    )
    events = raw.groupby("method", as_index=False).agg(actual_query_rate_mean=("actual_query_rate", "mean"))

    numeric = pd.DataFrame({"method": METHOD_ORDER})
    numeric["method_label"] = numeric["method"].map(METHOD_LABELS)
    numeric["supervision_type"] = numeric["method"].map(SUPERVISION_TYPE)
    numeric = numeric.merge(events, on="method", how="left")
    numeric["events_per_1000"] = numeric["actual_query_rate_mean"].fillna(0.0) * 1000.0 * numeric["method"].map(
        EVENT_MULTIPLIER
    )
    numeric = numeric.merge(frontier_c10, on="method", how="left")
    numeric = numeric.merge(contam_c10, on="method", how="left")
    numeric = numeric.merge(frontier_c100, on="method", how="left")
    numeric = numeric.merge(contam_c100, on="method", how="left")

    pretty = numeric[
        [
            "method_label",
            "supervision_type",
            "events_per_1000",
            "c10_frontier_gain",
            "c10_contam_gain",
            "c10_contam_ntr",
            "c100_frontier_gain",
            "c100_contam_gain",
            "c100_contam_ntr",
        ]
    ].copy()
    pretty["Events / 1k"] = pretty["events_per_1000"].map(lambda value: f"{value:.1f}")
    pretty["C10 frontier gain"] = pretty["c10_frontier_gain"].map(lambda value: f"{value:.3f}")
    pretty["C10 contam gain"] = pretty["c10_contam_gain"].map(lambda value: f"{value:.3f}")
    pretty["C10 contam NTR"] = pretty["c10_contam_ntr"].map(lambda value: f"{value:.3f}")
    pretty["C100 frontier gain"] = pretty["c100_frontier_gain"].map(lambda value: f"{value:.3f}")
    pretty["C100 contam gain"] = pretty["c100_contam_gain"].map(lambda value: f"{value:.3f}")
    pretty["C100 contam NTR"] = pretty["c100_contam_ntr"].map(lambda value: f"{value:.3f}")
    pretty = pretty[
        [
            "method_label",
            "supervision_type",
            "Events / 1k",
            "C10 frontier gain",
            "C10 contam gain",
            "C10 contam NTR",
            "C100 frontier gain",
            "C100 contam gain",
            "C100 contam NTR",
        ]
    ]
    pretty.columns = [
        "Method",
        "Supervision",
        "Events / 1k",
        "C10 frontier gain",
        "C10 contam gain",
        "C10 contam NTR",
        "C100 frontier gain",
        "C100 contam gain",
        "C100 contam NTR",
    ]
    save_table(numeric, pretty, "ppsn_supervision_family_summary", out_root)


def main() -> None:
    out_root = ROOT / "results/processed"
    ensure_dir(out_root)
    build_primary_cost_table(out_root)
    build_family_summary(out_root)


if __name__ == "__main__":
    main()
