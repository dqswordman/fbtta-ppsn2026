from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate publication figures from processed outputs.")
    parser.add_argument("--processed-root", default="results/processed")
    parser.add_argument("--figures-root", default="figures")
    return parser.parse_args()


def choose_existing(root: Path, candidates: list[str]) -> Path | None:
    for candidate in candidates:
        path = root / candidate
        if path.exists():
            return path
    return None


def plot_conceptual_phase_diagram(figures_root: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.set_title("Conceptual Phase Diagram")
    ax.set_xlabel("Query Rate")
    ax.set_ylabel("Contamination Ratio")

    ax.axvspan(0.0, 0.01, color="#f4d35e", alpha=0.45)
    ax.axvspan(0.01, 0.1, color="#9cc5a1", alpha=0.4)
    ax.fill_between([0.0, 0.1], [0.5, 0.25], [0.5, 0.5], color="#ee6c4d", alpha=0.25)
    ax.fill_between([0.0, 0.1], [0.25, 0.05], [0.25, 0.25], color="#f4d35e", alpha=0.25)
    ax.fill_between([0.0, 0.1], [0.05, 0.0], [0.05, 0.05], color="#9cc5a1", alpha=0.3)

    ax.text(0.004, 0.4, "0-bit fragile", fontsize=11)
    ax.text(0.02, 0.18, "tiny 1-bit\nsafe regime", fontsize=11)
    ax.text(0.055, 0.42, "high-feedback\nupper regime", fontsize=11)
    ax.set_xlim(0.0, 0.1)
    ax.set_ylim(0.0, 0.5)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures_root / "conceptual_phase_diagram.png", dpi=300)
    fig.savefig(figures_root / "conceptual_phase_diagram.pdf")
    plt.close(fig)


def plot_synthetic(processed_root: Path, figures_root: Path) -> None:
    path = choose_existing(processed_root, ["synthetic_main_summary.csv", "synthetic_smoke_summary.csv"])
    if path is None:
        return
    frame = pd.read_csv(path)
    if "feedback_noise" in frame.columns:
        frame = frame[frame["feedback_noise"] == 0.0]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    sns.lineplot(data=frame, x="query_rate", y="accuracy_mean", hue="method", style="env_name", marker="o", ax=axes[0])
    axes[0].set_title("Synthetic Separation")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_xlabel("Query Rate")
    axes[0].grid(alpha=0.2)

    sns.lineplot(
        data=frame,
        x="query_rate",
        y="negative_transfer_rate",
        hue="method",
        style="env_name",
        marker="o",
        ax=axes[1],
    )
    axes[1].set_title("Negative Transfer Rate")
    axes[1].set_ylabel("Rate")
    axes[1].set_xlabel("Query Rate")
    axes[1].grid(alpha=0.2)

    handles, labels = axes[0].get_legend_handles_labels()
    axes[1].get_legend().remove()
    axes[0].legend(handles=handles, labels=labels, fontsize=8)
    fig.tight_layout()
    fig.savefig(figures_root / "synthetic_separation.png", dpi=300)
    fig.savefig(figures_root / "synthetic_separation.pdf")
    plt.close(fig)


def plot_frontier(processed_root: Path, figures_root: Path) -> None:
    path = choose_existing(processed_root, ["frontier_final_summary.csv", "frontier_dev_summary.csv", "frontier_smoke_summary.csv"])
    if path is None:
        return
    frame = pd.read_csv(path)
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    sns.lineplot(data=frame, x="query_rate", y="id_accuracy_mean", hue="method", marker="o", ax=ax)
    ax.set_title("CIFAR-10-C Frontier")
    ax.set_xlabel("Query Rate")
    ax.set_ylabel("Mean ID Accuracy")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figures_root / "cifar10c_frontier.png", dpi=300)
    fig.savefig(figures_root / "cifar10c_frontier.pdf")
    plt.close(fig)


def plot_contamination(processed_root: Path, figures_root: Path) -> None:
    path = choose_existing(processed_root, ["contamination_final_summary.csv", "contamination_smoke_summary.csv"])
    if path is None:
        return
    frame = pd.read_csv(path)
    heatmap_frame = frame[frame["method"] == "fb_gate_tent"].pivot(
        index="contamination_ratio",
        columns="query_rate",
        values="gain_mean",
    )
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    sns.heatmap(heatmap_frame, annot=True, fmt=".3f", cmap="RdYlGn", center=0.0, ax=ax)
    ax.set_title("FB-Gate-Tent Gain Over Frozen")
    ax.set_xlabel("Query Rate")
    ax.set_ylabel("Contamination Ratio")
    fig.tight_layout()
    fig.savefig(figures_root / "contamination_heatmap.png", dpi=300)
    fig.savefig(figures_root / "contamination_heatmap.pdf")
    plt.close(fig)


def plot_acceptance_timeline(raw_root: Path, figures_root: Path) -> None:
    candidate_dirs = [
        raw_root / "contamination_final" / "windows",
        raw_root / "frontier_final" / "windows",
        raw_root / "contamination_smoke" / "windows",
        raw_root / "frontier_smoke" / "windows",
    ]
    window_files = []
    for directory in candidate_dirs:
        if directory.exists():
            window_files.extend(sorted(directory.glob("*fb_gate_tent*.csv")))
    if not window_files:
        return

    frame = pd.read_csv(window_files[0])
    if "accepted" not in frame.columns:
        return

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ax.step(frame["window_index"], frame["accepted"], where="mid", label="accepted")
    if "gate_delta" in frame.columns:
        ax.plot(frame["window_index"], frame["gate_delta"], label="gate delta", color="#bc4b51")
    ax.set_title("Representative Acceptance Timeline")
    ax.set_xlabel("Window Index")
    ax.set_ylabel("Decision / Delta")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_root / "acceptance_timeline.png", dpi=300)
    fig.savefig(figures_root / "acceptance_timeline.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    processed_root = ROOT / args.processed_root
    figures_root = ensure_dir(ROOT / args.figures_root)
    raw_root = ROOT / "results" / "raw"

    sns.set_theme(style="whitegrid", context="talk")
    plot_conceptual_phase_diagram(figures_root)
    plot_synthetic(processed_root, figures_root)
    plot_frontier(processed_root, figures_root)
    plot_contamination(processed_root, figures_root)
    plot_acceptance_timeline(raw_root, figures_root)


if __name__ == "__main__":
    main()
