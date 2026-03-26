from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.io import ensure_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build machine-readable tables from saved outputs.")
    parser.add_argument("--processed-root", default="results/processed")
    parser.add_argument("--raw-root", default="results/raw")
    return parser.parse_args()


def maybe_read_csv(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def save_table(frame: pd.DataFrame, csv_path: Path, tex_path: Path) -> None:
    frame.to_csv(csv_path, index=False)
    tex_path.write_text(
        frame.to_latex(index=False, float_format=lambda value: f"{value:.4f}", escape=True),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    processed_root = ROOT / args.processed_root
    raw_root = ROOT / args.raw_root
    table_root = ensure_dir(processed_root / "tables")

    frontier = maybe_read_csv(processed_root / "frontier_final_summary.csv")
    if frontier is None:
        frontier = maybe_read_csv(processed_root / "frontier_smoke_summary.csv")

    contamination = maybe_read_csv(processed_root / "contamination_final_summary.csv")
    if contamination is None:
        contamination = maybe_read_csv(processed_root / "contamination_smoke_summary.csv")

    synthetic = maybe_read_csv(processed_root / "synthetic_main_summary.csv")
    if synthetic is None:
        synthetic = maybe_read_csv(processed_root / "synthetic_smoke_summary.csv")

    if frontier is not None:
        main_table = frontier[frontier["query_rate"].isin([0.0, 0.01, 0.02])].copy()
        save_table(main_table, table_root / "main_comparison_table.csv", table_root / "main_comparison_table.tex")

        low_feedback = frontier[frontier["query_rate"].isin([0.001, 0.005, 0.01])].copy()
        save_table(low_feedback, table_root / "low_feedback_table.csv", table_root / "low_feedback_table.tex")

    if contamination is not None and synthetic is not None:
        benchmark_summary = pd.DataFrame(
            [
                {
                    "study": "synthetic",
                    "num_rows": len(synthetic),
                    "query_rates": synthetic["query_rate"].nunique(),
                    "methods": synthetic["method"].nunique(),
                },
                {
                    "study": "contamination",
                    "num_rows": len(contamination),
                    "query_rates": contamination["query_rate"].nunique(),
                    "methods": contamination["method"].nunique(),
                },
            ]
        )
        if frontier is not None:
            benchmark_summary = pd.concat(
                [
                    benchmark_summary,
                    pd.DataFrame(
                        [
                            {
                                "study": "frontier",
                                "num_rows": len(frontier),
                                "query_rates": frontier["query_rate"].nunique(),
                                "methods": frontier["method"].nunique(),
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
        save_table(benchmark_summary, table_root / "benchmark_summary.csv", table_root / "benchmark_summary.tex")

    compute_rows = []
    for name in ["source_train", "frontier_final", "contamination_final", "ablations", "frontier_smoke", "contamination_smoke"]:
        summary_path = raw_root / name
        if not summary_path.exists():
            continue
        for csv_path in summary_path.rglob("*.csv"):
            if csv_path.name.endswith("runs.csv"):
                frame = pd.read_csv(csv_path)
                if "runtime_sec" in frame.columns:
                    compute_rows.append(
                        {
                            "study": name,
                            "num_runs": len(frame),
                            "runtime_sec_mean": float(frame["runtime_sec"].mean()),
                            "peak_memory_mb_mean": float(frame["peak_memory_mb"].mean()) if "peak_memory_mb" in frame.columns else 0.0,
                        }
                    )
    if compute_rows:
        compute_frame = pd.DataFrame(compute_rows)
        save_table(compute_frame, table_root / "compute_reproducibility_table.csv", table_root / "compute_reproducibility_table.tex")


if __name__ == "__main__":
    main()
