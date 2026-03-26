from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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
METHOD_RANK = {method: index for index, method in enumerate(METHOD_ORDER)}

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

PLOT_METHODS = [
    "frozen",
    "tent",
    "fb_gate_tent",
    "self_gate_tent_zero_bit",
    "bitta_style_binary_feedback",
    "atta_or_simatta_style_active",
    "eatta_style_low_label",
]

MATCHED_CONTROL_METHODS = [
    "tent",
    "fb_gate_tent",
    "self_gate_tent_zero_bit",
    "throttle_tent_zero_bit",
]

COLORS = {
    "Frozen": "#7f8c8d",
    "Tent": "#1f77b4",
    "FB-Gate-Tent": "#d55e00",
    "Self-Gate Tent (0-bit)": "#2a9d8f",
    "Throttle Tent (0-bit)": "#8c564b",
    "COME-style 0-bit": "#6c757d",
    "BiTTA-style": "#9467bd",
    "ATTA-style": "#e9c46a",
    "EATTA-style": "#264653",
    "Full-label active": "#f4a261",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final Stage 2 figures and tables.")
    parser.add_argument("--processed-root", default="results/processed")
    parser.add_argument("--raw-root", default="results/raw")
    parser.add_argument("--figures-root", default="figures/final")
    parser.add_argument("--tables-root", default="results/processed/final_tables")
    return parser.parse_args()


def maybe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def bootstrap_mean_ci(values: pd.Series, seed: int = 0, n_boot: int = 2000) -> tuple[float, float]:
    array = values.to_numpy(dtype=float)
    if len(array) == 0:
        return float("nan"), float("nan")
    rng = np.random.RandomState(seed)
    samples = rng.randint(0, len(array), size=(n_boot, len(array)))
    means = array[samples].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def format_mean_std(mean: float, std: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} +/- {std:.{digits}f}"


def format_ci(mean: float, low: float, high: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def relpath_for_tex(path: Path) -> str:
    return path.as_posix()


def save_table(name: str, numeric_frame: pd.DataFrame, pretty_frame: pd.DataFrame, table_root: Path) -> None:
    csv_path = table_root / f"{name}.csv"
    tex_path = table_root / f"{name}.tex"
    numeric_frame.to_csv(csv_path, index=False)
    tex_path.write_text(pretty_frame.to_latex(index=False, escape=False), encoding="utf-8")


def save_figure(fig: plt.Figure, figures_root: Path, name: str) -> None:
    fig.savefig(figures_root / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(figures_root / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def build_primary_tables(processed_root: Path, table_root: Path) -> dict[str, Path]:
    frontier_c10 = maybe_read_csv(processed_root / "frontier_stage2_final_cifar10_summary.csv")
    frontier_c100 = maybe_read_csv(processed_root / "frontier_stage2_final_cifar100_summary.csv")
    frontier_c10_ann = maybe_read_csv(processed_root / "frontier_stage2_final_cifar10_annotated.csv")
    frontier_c100_ann = maybe_read_csv(processed_root / "frontier_stage2_final_cifar100_annotated.csv")

    contamination_c10 = maybe_read_csv(processed_root / "contamination_stage2_final_cifar10_summary.csv")
    contamination_c100 = maybe_read_csv(processed_root / "contamination_stage2_final_cifar100_summary.csv")
    contamination_c10_ann = maybe_read_csv(processed_root / "contamination_stage2_final_cifar10_annotated.csv")
    contamination_c100_ann = maybe_read_csv(processed_root / "contamination_stage2_final_cifar100_annotated.csv")

    primary_q = 0.01
    primary_ratio = 0.5

    def build_table(summary: pd.DataFrame, annotated: pd.DataFrame, dataset_label: str, ratio: float | None) -> pd.DataFrame:
        subset = summary[summary["query_rate"] == primary_q].copy()
        ann_subset = annotated[annotated["query_rate"] == primary_q].copy()
        if ratio is not None:
            subset = subset[subset["contamination_ratio"] == ratio].copy()
            ann_subset = ann_subset[ann_subset["contamination_ratio"] == ratio].copy()
        rows = []
        for _, row in subset.iterrows():
            ann_rows = ann_subset[ann_subset["method"] == row["method"]]
            ci_low, ci_high = bootstrap_mean_ci(ann_rows["gain_over_frozen"], seed=int(len(rows)))
            rows.append(
                {
                    "dataset": dataset_label,
                    "method": row["method"],
                    "method_label": METHOD_LABELS[row["method"]],
                    "query_rate": primary_q,
                    "contamination_ratio": ratio,
                    "id_accuracy_mean": row["id_accuracy_mean"],
                    "id_accuracy_std": row["id_accuracy_std"],
                    "gain_mean": row["gain_mean"],
                    "gain_ci_low": ci_low,
                    "gain_ci_high": ci_high,
                    "negative_transfer_rate": row["negative_transfer_rate"],
                    "ece_mean": row["ece_mean"],
                    "oem_0_9_mean": row["oem_0_9_mean"],
                    "accept_rate_mean": row["accept_rate_mean"],
                    "runtime_sec_mean": row["runtime_sec_mean"],
                    "peak_memory_mb_mean": row["peak_memory_mb_mean"],
                    "method_rank": METHOD_RANK[row["method"]],
                }
            )
        frame = pd.DataFrame(rows).sort_values(["dataset", "method_rank"]).reset_index(drop=True)
        return frame

    frontier_table = pd.concat(
        [
            build_table(frontier_c10, frontier_c10_ann, "CIFAR-10-C", None),
            build_table(frontier_c100, frontier_c100_ann, "CIFAR-100-C", None),
        ],
        ignore_index=True,
    )
    frontier_pretty = frontier_table[
        [
            "dataset",
            "method_label",
            "id_accuracy_mean",
            "id_accuracy_std",
            "gain_mean",
            "gain_ci_low",
            "gain_ci_high",
            "negative_transfer_rate",
            "ece_mean",
        ]
    ].copy()
    frontier_pretty["ID acc"] = frontier_pretty.apply(lambda row: format_mean_std(row["id_accuracy_mean"], row["id_accuracy_std"]), axis=1)
    frontier_pretty["Gain vs frozen"] = frontier_pretty.apply(
        lambda row: format_ci(row["gain_mean"], row["gain_ci_low"], row["gain_ci_high"]), axis=1
    )
    frontier_pretty["NTR"] = frontier_pretty["negative_transfer_rate"].map(lambda value: f"{value:.3f}")
    frontier_pretty["ECE"] = frontier_pretty["ece_mean"].map(lambda value: f"{value:.3f}")
    frontier_pretty = frontier_pretty[["dataset", "method_label", "ID acc", "Gain vs frozen", "NTR", "ECE"]]
    frontier_pretty.columns = ["Dataset", "Method", "ID acc", "Gain vs frozen", "NTR", "ECE"]
    save_table("frontier_primary_table", frontier_table, frontier_pretty, table_root)

    contamination_table = pd.concat(
        [
            build_table(contamination_c10, contamination_c10_ann, "CIFAR-10-C + SVHN", primary_ratio),
            build_table(contamination_c100, contamination_c100_ann, "CIFAR-100-C + SVHN", primary_ratio),
        ],
        ignore_index=True,
    )
    contamination_pretty = contamination_table[
        [
            "dataset",
            "method_label",
            "id_accuracy_mean",
            "id_accuracy_std",
            "gain_mean",
            "gain_ci_low",
            "gain_ci_high",
            "negative_transfer_rate",
            "accept_rate_mean",
        ]
    ].copy()
    contamination_pretty["ID acc"] = contamination_pretty.apply(
        lambda row: format_mean_std(row["id_accuracy_mean"], row["id_accuracy_std"]), axis=1
    )
    contamination_pretty["Gain vs frozen"] = contamination_pretty.apply(
        lambda row: format_ci(row["gain_mean"], row["gain_ci_low"], row["gain_ci_high"]), axis=1
    )
    contamination_pretty["NTR"] = contamination_pretty["negative_transfer_rate"].map(lambda value: f"{value:.3f}")
    contamination_pretty["Accept"] = contamination_pretty["accept_rate_mean"].map(lambda value: f"{value:.3f}")
    contamination_pretty = contamination_pretty[["dataset", "method_label", "ID acc", "Gain vs frozen", "NTR", "Accept"]]
    contamination_pretty.columns = ["Dataset", "Method", "ID acc", "Gain vs frozen", "NTR", "Accept"]
    save_table("contamination_primary_table", contamination_table, contamination_pretty, table_root)

    generalization = frontier_table[["dataset", "method", "id_accuracy_mean", "gain_mean", "negative_transfer_rate"]].copy()
    generalization["regime"] = "frontier"
    contamination_generalization = contamination_table[
        ["dataset", "method", "id_accuracy_mean", "gain_mean", "negative_transfer_rate"]
    ].copy()
    contamination_generalization["regime"] = "contam_r0.5"
    wide = pd.concat([generalization, contamination_generalization], ignore_index=True)
    wide["dataset_regime"] = wide["dataset"] + " / " + wide["regime"]
    generalization_numeric = (
        wide.pivot_table(
            index="method",
            columns="dataset_regime",
            values=["id_accuracy_mean", "gain_mean", "negative_transfer_rate"],
        )
        .sort_index(axis=1)
        .reset_index()
    )
    generalization_numeric.columns = [" ".join([str(part) for part in column if part]).strip() for column in generalization_numeric.columns]
    generalization_numeric.insert(1, "method_label", generalization_numeric["method"].map(METHOD_LABELS))
    generalization_numeric["method_rank"] = generalization_numeric["method"].map(METHOD_RANK)
    generalization_numeric = generalization_numeric.sort_values("method_rank").reset_index(drop=True)
    generalization_pretty = pd.DataFrame({"Method": generalization_numeric["method_label"]})
    generalization_pretty["C10 frontier acc"] = generalization_numeric["id_accuracy_mean CIFAR-10-C / frontier"].map(
        lambda value: f"{value:.3f}"
    )
    generalization_pretty["C10 contam gain"] = generalization_numeric["gain_mean CIFAR-10-C + SVHN / contam_r0.5"].map(
        lambda value: f"{value:.3f}"
    )
    generalization_pretty["C10 contam NTR"] = generalization_numeric[
        "negative_transfer_rate CIFAR-10-C + SVHN / contam_r0.5"
    ].map(lambda value: f"{value:.3f}")
    generalization_pretty["C100 frontier acc"] = generalization_numeric["id_accuracy_mean CIFAR-100-C / frontier"].map(
        lambda value: f"{value:.3f}"
    )
    generalization_pretty["C100 contam gain"] = generalization_numeric[
        "gain_mean CIFAR-100-C + SVHN / contam_r0.5"
    ].map(lambda value: f"{value:.3f}")
    generalization_pretty["C100 contam NTR"] = generalization_numeric[
        "negative_transfer_rate CIFAR-100-C + SVHN / contam_r0.5"
    ].map(lambda value: f"{value:.3f}")
    save_table("generalization_table", generalization_numeric, generalization_pretty, table_root)

    return {
        "frontier_primary_table": table_root / "frontier_primary_table.tex",
        "contamination_primary_table": table_root / "contamination_primary_table.tex",
        "generalization_table": table_root / "generalization_table.tex",
    }


def build_robustness_tables(processed_root: Path, table_root: Path) -> dict[str, Path]:
    noise_c10 = maybe_read_csv(processed_root / "robustness_noise_stage2_cifar10_summary.csv")
    delay_c10 = maybe_read_csv(processed_root / "robustness_delay_stage2_cifar10_summary.csv")
    noise_c100 = maybe_read_csv(processed_root / "robustness_noise_stage2_cifar100_summary.csv")
    delay_c100 = maybe_read_csv(processed_root / "robustness_delay_stage2_cifar100_summary.csv")

    def select_primary(frame: pd.DataFrame, dataset_label: str, axis_column: str) -> pd.DataFrame:
        subset = frame[
            (frame["method"] == "fb_gate_tent")
            & (frame["query_rate"] == 0.01)
            & (frame["contamination_ratio"] == 0.5)
        ].copy()
        subset["dataset"] = dataset_label
        subset["axis_column"] = axis_column
        return subset

    robustness_numeric = pd.concat(
        [
            select_primary(noise_c10, "CIFAR-10-C + SVHN", "feedback_noise"),
            select_primary(delay_c10, "CIFAR-10-C + SVHN", "feedback_delay_windows"),
            select_primary(noise_c100, "CIFAR-100-C + SVHN", "feedback_noise"),
            select_primary(delay_c100, "CIFAR-100-C + SVHN", "feedback_delay_windows"),
        ],
        ignore_index=True,
    )
    robustness_pretty = robustness_numeric[
        [
            "dataset",
            "axis_column",
            "robustness_value",
            "id_accuracy_mean",
            "gain_mean",
            "negative_transfer_rate",
            "accept_rate_mean",
        ]
    ].copy()
    robustness_pretty["Axis"] = robustness_pretty["axis_column"].map(
        {"feedback_noise": "Noise eta", "feedback_delay_windows": "Delay d"}
    )
    robustness_pretty["Value"] = robustness_pretty["robustness_value"].map(lambda value: f"{value:g}")
    robustness_pretty["ID acc"] = robustness_pretty["id_accuracy_mean"].map(lambda value: f"{value:.3f}")
    robustness_pretty["Gain"] = robustness_pretty["gain_mean"].map(lambda value: f"{value:.3f}")
    robustness_pretty["NTR"] = robustness_pretty["negative_transfer_rate"].map(lambda value: f"{value:.3f}")
    robustness_pretty["Accept"] = robustness_pretty["accept_rate_mean"].map(lambda value: f"{value:.3f}")
    robustness_pretty = robustness_pretty[["dataset", "Axis", "Value", "ID acc", "Gain", "NTR", "Accept"]]
    robustness_pretty.columns = ["Dataset", "Axis", "Value", "ID acc", "Gain", "NTR", "Accept"]
    save_table("robustness_primary_table", robustness_numeric, robustness_pretty, table_root)
    return {"robustness_primary_table": table_root / "robustness_primary_table.tex"}


def build_compute_table(raw_root: Path, table_root: Path) -> dict[str, Path]:
    study_specs = [
        ("Synthetic witness", raw_root / "synthetic_main" / "synthetic_main_runs.csv"),
        ("C10 frontier final", raw_root / "frontier_stage2_final_cifar10" / "frontier_runs.csv"),
        ("C10 contamination final", raw_root / "contamination_stage2_final_cifar10" / "contamination_runs.csv"),
        ("C100 frontier final", raw_root / "frontier_stage2_final_cifar100" / "frontier_runs.csv"),
        ("C100 contamination final", raw_root / "contamination_stage2_final_cifar100" / "contamination_runs.csv"),
        ("C10 robustness noise", raw_root / "robustness_noise_stage2_cifar10" / "robustness_runs.csv"),
        ("C10 robustness delay", raw_root / "robustness_delay_stage2_cifar10" / "robustness_runs.csv"),
        ("C100 robustness noise", raw_root / "robustness_noise_stage2_cifar100" / "robustness_runs.csv"),
        ("C100 robustness delay", raw_root / "robustness_delay_stage2_cifar100" / "robustness_runs.csv"),
    ]
    rows = []
    for study_label, path in study_specs:
        frame = maybe_read_csv(path)
        if "runtime_sec" not in frame.columns or "peak_memory_mb" not in frame.columns:
            continue
        rows.append(
            {
                "study": study_label,
                "num_runs": len(frame),
                "runtime_sec_mean": float(frame["runtime_sec"].mean()),
                "peak_memory_mb_mean": float(frame["peak_memory_mb"].mean()),
                "raw_path": str(path.relative_to(ROOT)).replace("\\", "/"),
            }
        )

    for dataset_dir in [raw_root / "source_train" / "cifar100"]:
        if not dataset_dir.exists():
            continue
        summaries = sorted(dataset_dir.glob("seed*/summary.json"))
        if not summaries:
            continue
        records = [json.loads(path.read_text(encoding="utf-8")) for path in summaries]
        dataset_name = records[0].get("dataset_name", dataset_dir.name)
        rows.append(
            {
                "study": f"Source train {dataset_name}",
                "num_runs": len(records),
                "runtime_sec_mean": float(np.mean([record["runtime_sec"] for record in records])),
                "peak_memory_mb_mean": float(np.mean([record["peak_memory_mb"] for record in records])),
                "raw_path": str(dataset_dir.relative_to(ROOT)).replace("\\", "/"),
            }
        )

    compute_numeric = pd.DataFrame(rows)
    compute_pretty = compute_numeric.copy()
    compute_pretty["Runtime / stream (s)"] = compute_pretty["runtime_sec_mean"].map(lambda value: f"{value:.3f}")
    compute_pretty["Peak memory (MB)"] = compute_pretty["peak_memory_mb_mean"].map(lambda value: f"{value:.1f}")
    compute_pretty["raw_path"] = compute_pretty["raw_path"].map(lambda value: f"\\path{{{value}}}")
    compute_pretty = compute_pretty[["study", "num_runs", "Runtime / stream (s)", "Peak memory (MB)", "raw_path"]]
    compute_pretty.columns = ["Study", "Runs", "Runtime / stream (s)", "Peak memory (MB)", "Raw path"]
    save_table("compute_reproducibility_table", compute_numeric, compute_pretty, table_root)
    return {"compute_reproducibility_table": table_root / "compute_reproducibility_table.tex"}


def plot_synthetic(processed_root: Path, figures_root: Path) -> dict[str, Path]:
    frame = maybe_read_csv(processed_root / "synthetic_main_summary.csv")
    frame = frame[frame["feedback_noise"] == 0.0].copy()
    frame["method_label"] = frame["method"].map({"frozen": "Frozen", "zero_bit_update": "Tent-like 0-bit", "fb_gate": "FB-Gate"})
    env_order = ["env_plus", "env_minus"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2), sharex=True)
    for ax, env_name in zip(axes, env_order):
        subset = frame[frame["env_name"] == env_name].copy()
        for method_label in ["Frozen", "Tent-like 0-bit", "FB-Gate"]:
            part = subset[subset["method_label"] == method_label].sort_values("query_rate")
            ax.plot(part["query_rate"], part["accuracy_mean"], marker="o", linewidth=2.0, label=method_label)
            ax.fill_between(
                part["query_rate"],
                part["accuracy_mean"] - part["accuracy_std"],
                part["accuracy_mean"] + part["accuracy_std"],
                alpha=0.12,
            )
        ax.set_title("Helpful env" if env_name == "env_plus" else "Harmful env")
        ax.set_xlabel("Query rate")
        ax.set_ylabel("Accuracy")
        ax.grid(alpha=0.25)
    axes[1].legend(loc="lower right", frameon=True)
    fig.suptitle("Synthetic 0-bit / 1-bit separation", y=1.02)
    fig.tight_layout()
    save_figure(fig, figures_root, "synthetic_separation_final")
    return {"synthetic_separation_final": figures_root / "synthetic_separation_final.pdf"}


def plot_frontier(processed_root: Path, figures_root: Path) -> dict[str, Path]:
    frame = maybe_read_csv(processed_root / "frontier_stage2_final_cifar10_summary.csv").copy()
    frame = frame[frame["method"].isin(PLOT_METHODS)].copy()
    frame["method_label"] = frame["method"].map(METHOD_LABELS)
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for method in PLOT_METHODS:
        part = frame[frame["method"] == method].sort_values("query_rate")
        label = METHOD_LABELS[method]
        ax.plot(
            part["query_rate"],
            part["id_accuracy_mean"],
            marker="o",
            linewidth=2.4 if method in {"tent", "fb_gate_tent", "self_gate_tent_zero_bit"} else 1.7,
            alpha=1.0 if method in {"tent", "fb_gate_tent", "self_gate_tent_zero_bit"} else 0.8,
            color=COLORS[label],
            label=label,
        )
    ax.set_title("Held-out CIFAR-10-C frontier")
    ax.set_xlabel("Query rate")
    ax.set_ylabel("Mean ID accuracy")
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=9, frameon=True)
    fig.tight_layout()
    save_figure(fig, figures_root, "cifar10_frontier_final")
    return {"cifar10_frontier_final": figures_root / "cifar10_frontier_final.pdf"}


def plot_phase_heatmap(processed_root: Path, figures_root: Path) -> dict[str, Path]:
    c10 = maybe_read_csv(processed_root / "contamination_stage2_final_cifar10_summary.csv")
    c100 = maybe_read_csv(processed_root / "contamination_stage2_final_cifar100_summary.csv")
    datasets = [("CIFAR-10-C + SVHN", c10), ("CIFAR-100-C + SVHN", c100)]
    methods = [("tent", "Tent"), ("fb_gate_tent", "FB-Gate-Tent")]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.8), sharex=True, sharey=True)
    for row_idx, (dataset_label, frame) in enumerate(datasets):
        for col_idx, (method, method_label) in enumerate(methods):
            ax = axes[row_idx, col_idx]
            heat = frame[frame["method"] == method].pivot(
                index="contamination_ratio", columns="query_rate", values="negative_transfer_rate"
            )
            sns.heatmap(heat, annot=True, fmt=".2f", cmap="RdYlGn_r", vmin=0.0, vmax=0.2, cbar=col_idx == 1, ax=ax)
            ax.set_title(f"{dataset_label} | {method_label}")
            ax.set_xlabel("Query rate")
            ax.set_ylabel("Contamination ratio")
    fig.suptitle("Contamination phase heatmap: negative-transfer rate", y=1.01)
    fig.tight_layout()
    save_figure(fig, figures_root, "contamination_phase_heatmap_final")
    return {"contamination_phase_heatmap_final": figures_root / "contamination_phase_heatmap_final.pdf"}


def plot_matched_conservatism(processed_root: Path, figures_root: Path) -> dict[str, Path]:
    c10 = maybe_read_csv(processed_root / "contamination_stage2_final_cifar10_summary.csv")
    c100 = maybe_read_csv(processed_root / "contamination_stage2_final_cifar100_summary.csv")
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6), sharex="col")
    for row_idx, (dataset_label, frame) in enumerate([("CIFAR-10", c10), ("CIFAR-100", c100)]):
        subset = frame[(frame["query_rate"] == 0.01) & (frame["method"].isin(MATCHED_CONTROL_METHODS))].copy()
        subset["method_label"] = subset["method"].map(METHOD_LABELS)
        ax_gain = axes[row_idx, 0]
        ax_ntr = axes[row_idx, 1]
        for method in MATCHED_CONTROL_METHODS:
            part = subset[subset["method"] == method].sort_values("contamination_ratio")
            label = METHOD_LABELS[method]
            ax_gain.plot(
                part["contamination_ratio"],
                part["gain_mean"],
                marker="o",
                linewidth=2.2 if method == "fb_gate_tent" else 1.8,
                color=COLORS[label],
                label=label,
            )
            ax_ntr.plot(
                part["contamination_ratio"],
                part["negative_transfer_rate"],
                marker="o",
                linewidth=2.2 if method == "fb_gate_tent" else 1.8,
                color=COLORS[label],
                label=label,
            )
        ax_gain.set_title(f"{dataset_label}: gain at q = 0.01")
        ax_gain.set_ylabel("Gain over frozen")
        ax_gain.grid(alpha=0.25)
        ax_ntr.set_title(f"{dataset_label}: negative transfer at q = 0.01")
        ax_ntr.set_ylabel("Negative-transfer rate")
        ax_ntr.grid(alpha=0.25)
    axes[1, 0].set_xlabel("Contamination ratio")
    axes[1, 1].set_xlabel("Contamination ratio")
    axes[0, 1].legend(loc="upper left", fontsize=9, frameon=True)
    fig.tight_layout()
    save_figure(fig, figures_root, "matched_conservatism_final")
    return {"matched_conservatism_final": figures_root / "matched_conservatism_final.pdf"}


def plot_robustness(processed_root: Path, figures_root: Path) -> dict[str, Path]:
    noise_c10 = maybe_read_csv(processed_root / "robustness_noise_stage2_cifar10_summary.csv")
    delay_c10 = maybe_read_csv(processed_root / "robustness_delay_stage2_cifar10_summary.csv")
    noise_c100 = maybe_read_csv(processed_root / "robustness_noise_stage2_cifar100_summary.csv")
    delay_c100 = maybe_read_csv(processed_root / "robustness_delay_stage2_cifar100_summary.csv")

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6), sharex=True)
    panels = [
        ("CIFAR-10 noise", noise_c10, "feedback_noise", axes[0, 0]),
        ("CIFAR-100 noise", noise_c100, "feedback_noise", axes[0, 1]),
        ("CIFAR-10 delay", delay_c10, "feedback_delay_windows", axes[1, 0]),
        ("CIFAR-100 delay", delay_c100, "feedback_delay_windows", axes[1, 1]),
    ]
    for title, frame, value_column, ax in panels:
        subset = frame[(frame["method"] == "fb_gate_tent") & (frame["query_rate"] == 0.01)].copy()
        for value in sorted(subset[value_column].unique()):
            part = subset[subset[value_column] == value].sort_values("contamination_ratio")
            label = f"{'eta' if value_column == 'feedback_noise' else 'd'} = {value:g}"
            ax.plot(part["contamination_ratio"], part["gain_mean"], marker="o", linewidth=2.0, label=label)
        ax.set_title(title)
        ax.set_ylabel("Gain over frozen")
        ax.grid(alpha=0.25)
    axes[1, 0].set_xlabel("Contamination ratio")
    axes[1, 1].set_xlabel("Contamination ratio")
    axes[0, 1].legend(loc="lower left", fontsize=9, frameon=True)
    fig.tight_layout()
    save_figure(fig, figures_root, "robustness_noise_delay_final")
    return {"robustness_noise_delay_final": figures_root / "robustness_noise_delay_final.pdf"}


def main() -> None:
    args = parse_args()
    processed_root = ROOT / args.processed_root
    raw_root = ROOT / args.raw_root
    figures_root = ensure_dir(ROOT / args.figures_root)
    table_root = ensure_dir(ROOT / args.tables_root)

    plt.rcParams["font.family"] = "DejaVu Serif"
    sns.set_theme(style="whitegrid", context="paper")

    table_paths = {}
    table_paths.update(build_primary_tables(processed_root, table_root))
    table_paths.update(build_robustness_tables(processed_root, table_root))
    table_paths.update(build_compute_table(raw_root, table_root))

    figure_paths = {}
    figure_paths.update(plot_synthetic(processed_root, figures_root))
    figure_paths.update(plot_frontier(processed_root, figures_root))
    figure_paths.update(plot_phase_heatmap(processed_root, figures_root))
    figure_paths.update(plot_matched_conservatism(processed_root, figures_root))
    figure_paths.update(plot_robustness(processed_root, figures_root))

    manifest = {
        "tables": {name: str(path.relative_to(ROOT)) for name, path in table_paths.items()},
        "figures": {name: str(path.relative_to(ROOT)) for name, path in figure_paths.items()},
    }
    (table_root / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
