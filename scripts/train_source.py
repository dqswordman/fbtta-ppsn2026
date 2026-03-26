from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.cifar import build_cifar_test, build_cifar_train_val
from src.models.cifar_resnet import build_cifar_resnet18
from src.utils.config import load_yaml
from src.utils.io import ensure_dir, save_resolved_config, write_json
from src.utils.repro import default_device, set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CIFAR source model.")
    parser.add_argument("--config", default="configs/source_train.yaml")
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += float(loss.item()) * len(labels)
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_examples += len(labels)
    return total_loss / total_examples, total_correct / total_examples


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    if args.seed is not None:
        config["seed"] = args.seed

    seed = int(config.get("seed", 0))
    dataset_name = str(config.get("dataset", {}).get("name", "cifar10"))
    num_classes = int(config.get("dataset", {}).get("num_classes", 10 if dataset_name == "cifar10" else 100))
    set_global_seed(seed)
    device = default_device()

    training_cfg = config["training"]
    paths_cfg = config["paths"]
    output_root = ensure_dir(Path(paths_cfg["output_root"]) / f"seed{seed}")
    raw_root = ensure_dir(ROOT / "results" / "raw" / "source_train" / dataset_name / f"seed{seed}")

    train_dataset, val_dataset = build_cifar_train_val(
        ROOT / paths_cfg["data_root"],
        dataset_name=dataset_name,
        val_size=int(training_cfg["val_size"]),
        seed=seed,
    )
    test_dataset = build_cifar_test(ROOT / paths_cfg["data_root"], dataset_name=dataset_name)

    num_workers = int(training_cfg.get("num_workers", 4))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(training_cfg["batch_size"]),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(training_cfg["eval_batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(training_cfg["eval_batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    model = build_cifar_resnet18(num_classes=num_classes).to(device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=float(training_cfg["lr"]),
        momentum=float(training_cfg["momentum"]),
        weight_decay=float(training_cfg["weight_decay"]),
        nesterov=True,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(training_cfg["epochs"]))
    criterion = nn.CrossEntropyLoss()
    amp_enabled = bool(training_cfg.get("amp", True) and device.type == "cuda")

    best_val_acc = -1.0
    history_rows = []
    start = time.time()

    for epoch in range(1, int(training_cfg["epochs"]) + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_examples = 0
        progress = tqdm(train_loader, desc=f"train seed={seed} epoch={epoch}", leave=False, disable=True)
        for images, labels in progress:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item()) * len(labels)
            epoch_correct += int((logits.argmax(dim=1) == labels).sum().item())
            epoch_examples += len(labels)
            progress.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        train_loss = epoch_loss / epoch_examples
        train_acc = epoch_correct / epoch_examples
        val_loss, val_acc = evaluate(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "lr": scheduler.get_last_lr()[0],
        }
        history_rows.append(row)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "seed": seed,
                    "dataset_name": dataset_name,
                    "num_classes": num_classes,
                    "model_state": model.state_dict(),
                    "val_accuracy": val_acc,
                    "config": config,
                },
                output_root / "best.pt",
            )

    checkpoint = torch.load(output_root / "best.pt", map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    test_loss, test_acc = evaluate(model.to(device), test_loader, device)
    runtime = time.time() - start

    history_df = pd.DataFrame(history_rows)
    history_df.to_csv(raw_root / "history.csv", index=False)
    save_resolved_config(raw_root, config)
    write_json(
        raw_root / "summary.json",
        {
            "seed": seed,
            "dataset_name": dataset_name,
            "num_classes": num_classes,
            "best_val_accuracy": float(best_val_acc),
            "clean_test_accuracy": float(test_acc),
            "clean_test_error": float(1.0 - test_acc),
            "clean_test_loss": float(test_loss),
            "runtime_sec": runtime,
            "checkpoint_path": str(output_root / "best.pt"),
            "peak_memory_mb": float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else 0.0,
        },
    )
    save_resolved_config(output_root, config)

    print(
        {
            "seed": seed,
            "best_val_accuracy": round(best_val_acc, 4),
            "clean_test_accuracy": round(test_acc, 4),
            "checkpoint": str(output_root / "best.pt"),
        }
    )


if __name__ == "__main__":
    main()
