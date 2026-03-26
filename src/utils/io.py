from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import yaml


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def write_json(path: str | Path, payload: Dict[str, Any]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def write_yaml(path: str | Path, payload: Dict[str, Any]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def append_jsonl(path: str | Path, payload: Dict[str, Any]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def create_run_dir(root: str | Path, study: str, run_name: str) -> Path:
    run_dir = ensure_dir(Path(root) / study / run_name)
    return run_dir


def save_resolved_config(run_dir: str | Path, config: Dict[str, Any]) -> None:
    write_yaml(Path(run_dir) / "resolved_config.yaml", config)

