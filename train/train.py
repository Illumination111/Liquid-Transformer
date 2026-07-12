#!/usr/bin/env python3
"""Unified CIFAR-100 trainer for DeiT-Tiny and Swin-Tiny."""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
import sys
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "model-solid"
for path in (PROJECT_ROOT, MODEL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from deit_tiny import deit_tiny  # noqa: E402
from swin_tiny import swin_tiny  # noqa: E402

from dataset.cifar100 import build_cifar100_loaders  # noqa: E402

MODEL_FACTORIES: dict[str, Callable[..., nn.Module]] = {
    "deit-tiny": deit_tiny,
    "swin-tiny": swin_tiny,
}
MAX_EPOCHS = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODEL_FACTORIES, required=True)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "dataset")
    parser.add_argument("--log-dir", type=Path, default=PROJECT_ROOT / "train-log")
    parser.add_argument(
        "--epochs",
        type=int,
        default=MAX_EPOCHS,
        help=f"training epochs, 1-{MAX_EPOCHS} (default: {MAX_EPOCHS})",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--drop-path", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_true", help="disable CUDA mixed precision")
    parser.add_argument("--no-download", action="store_true", help="do not download CIFAR-100")
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_logger(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("liquid-transformer")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    warmup_epochs: int,
    min_lr: float,
    base_lr: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    min_factor = min_lr / base_lr

    def factor(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return max((epoch + 1) / warmup_epochs, min_factor)
        progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs - 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return min_factor + (1.0 - min_factor) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def autocast_context(device: torch.device, enabled: bool):
    if enabled:
        return torch.autocast(device_type=device.type, dtype=torch.float16)
    return nullcontext()


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    amp: bool = False,
    clip_grad: float = 0.0,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), autocast_context(device, amp):
            logits = model(images)
            loss = criterion(logits, targets)
        if training:
            assert scaler is not None
            scaler.scale(loss).backward()
            if clip_grad > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            scaler.step(optimizer)
            scaler.update()
        batch_size = targets.shape[0]
        total_loss += loss.detach().item() * batch_size
        total_correct += (logits.detach().argmax(dim=1) == targets).sum().item()
        total_samples += batch_size
    return total_loss / total_samples, 100.0 * total_correct / total_samples


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_accuracy: float,
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_accuracy": best_accuracy,
            "args": vars(args),
        },
        path,
    )


def main() -> None:
    args = parse_args()
    if not 1 <= args.epochs <= MAX_EPOCHS:
        raise ValueError(f"epochs must be between 1 and {MAX_EPOCHS}")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    seed_everything(args.seed)
    device = torch.device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp

    args.log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{timestamp}-{args.model}-solid"
    logger = build_logger(args.log_dir / f"{run_name}.log")
    logger.info("configuration: %s", json.dumps(vars(args), default=str, sort_keys=True))
    logger.info("device=%s amp=%s", device, use_amp)

    train_loader, validation_loader = build_cifar100_loaders(
        args.data_dir,
        args.batch_size,
        args.workers,
        download=not args.no_download,
    )
    model_kwargs = {}
    if args.drop_path is not None:
        model_kwargs["drop_path_rate"] = args.drop_path
    model = MODEL_FACTORIES[args.model](num_classes=100, **model_kwargs).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    logger.info("model=%s parameters=%.2fM", args.model, parameter_count / 1e6)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = build_scheduler(
        optimizer, args.epochs, args.warmup_epochs, args.min_lr, args.lr
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start_epoch = 0
    best_accuracy = 0.0

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_accuracy = float(checkpoint["best_accuracy"])
        logger.info("resumed=%s start_epoch=%d", args.resume, start_epoch + 1)

    last_path = args.log_dir / f"{run_name}-last.pt"
    best_path = args.log_dir / f"{run_name}-best.pt"
    started = time.monotonic()
    for epoch in range(start_epoch, args.epochs):
        train_loss, train_accuracy = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            scaler,
            use_amp,
            args.clip_grad,
        )
        validation_loss, validation_accuracy = run_epoch(
            model, validation_loader, criterion, device, amp=use_amp
        )
        scheduler.step()
        improved = validation_accuracy > best_accuracy
        best_accuracy = max(best_accuracy, validation_accuracy)
        save_checkpoint(
            last_path, model, optimizer, scheduler, epoch, best_accuracy, args
        )
        if improved:
            save_checkpoint(
                best_path, model, optimizer, scheduler, epoch, best_accuracy, args
            )
        logger.info(
            "epoch=%03d/%03d lr=%.3e train_loss=%.4f train_acc=%.2f "
            "val_loss=%.4f val_acc=%.2f best_acc=%.2f elapsed=%.1fmin",
            epoch + 1,
            args.epochs,
            optimizer.param_groups[0]["lr"],
            train_loss,
            train_accuracy,
            validation_loss,
            validation_accuracy,
            best_accuracy,
            (time.monotonic() - started) / 60,
        )
    logger.info("training complete: best_val_accuracy=%.2f checkpoint=%s", best_accuracy, best_path)


if __name__ == "__main__":
    main()
