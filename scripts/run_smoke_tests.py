from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the required smoke tests.")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def run_command(python_exec: str, script: str, config: str) -> None:
    command = [python_exec, str(ROOT / script), "--config", str(ROOT / config)]
    subprocess.run(command, check=True, cwd=ROOT)


def main() -> None:
    args = parse_args()
    checkpoint = ROOT / "artifacts" / "source_model" / "seed0" / "best.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"Smoke tests require a trained source checkpoint at {checkpoint}")

    run_command(args.python, "scripts/run_synthetic.py", "configs/synthetic_smoke.yaml")
    run_command(args.python, "scripts/run_cifar10c_frontier.py", "configs/frontier_smoke.yaml")
    run_command(args.python, "scripts/run_contamination.py", "configs/contamination_smoke.yaml")
    subprocess.run([args.python, str(ROOT / "scripts" / "plot_figures.py")], check=True, cwd=ROOT)
    subprocess.run([args.python, str(ROOT / "scripts" / "build_tables.py")], check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
