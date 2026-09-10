#!/usr/bin/env python3
"""Legacy NSGA-II entry point. New PPO/LIF work uses train/search.py."""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import random
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from queue import Empty
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LIQUID_DIR = PROJECT_ROOT / "model-liquid"
SOLID_DIR = PROJECT_ROOT / "model-solid"
EVOLUTION_DIR = Path(__file__).resolve().parent / "evolution"
for path in (PROJECT_ROOT, LIQUID_DIR, SOLID_DIR, EVOLUTION_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from deit_tiny_liquid import deit_tiny_liquid, deit_topology_nodes  # noqa: E402
from metrics import measure_model_topologies  # noqa: E402
from nsga2 import (  # noqa: E402
    Individual,
    create_population,
    make_offspring,
    select_survivors,
)
from small_world import SmallWorldTopology  # noqa: E402
from swin_tiny_liquid import swin_tiny_liquid, swin_topology_nodes  # noqa: E402
from visualize import EvolutionRecorder  # noqa: E402

from dataset.cifar100 import DEFAULT_DATA_DIR, build_cifar100_evolution_loaders  # noqa: E402

MAX_EPOCHS = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("deit-tiny", "swin-tiny"), required=True)
    parser.add_argument("--devices", default="0,1", help="comma-separated CUDA device IDs")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--log-dir", type=Path, default=PROJECT_ROOT / "train-log")
    parser.add_argument("--solid-checkpoint", type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--validation-size", type=int, default=5000)
    parser.add_argument("--coarse-population", type=int, default=16)
    parser.add_argument("--coarse-generations", type=int, default=10)
    parser.add_argument(
        "--coarse-epochs",
        type=int,
        default=2,
        help=f"epochs per coarse candidate, max {MAX_EPOCHS}",
    )
    parser.add_argument("--coarse-train-fraction", type=float, default=0.5)
    parser.add_argument("--refine-population", type=int, default=8)
    parser.add_argument("--refine-generations", type=int, default=10)
    parser.add_argument(
        "--refine-epochs",
        type=int,
        default=5,
        help=f"epochs per refined candidate, max {MAX_EPOCHS}",
    )
    parser.add_argument("--refine-train-fraction", type=float, default=1.0)
    parser.add_argument("--finalists", type=int, default=2)
    parser.add_argument(
        "--final-epochs",
        type=int,
        default=MAX_EPOCHS,
        help=f"full training epochs for Pareto finalists, 0-{MAX_EPOCHS}",
    )
    parser.add_argument("--graph-degree", type=int, default=8)
    parser.add_argument("--message-steps", type=int, default=2)
    parser.add_argument("--mutation-rate", type=float, default=0.01)
    parser.add_argument("--crossover-probability", type=float, default=0.5)
    parser.add_argument("--min-rewire", type=float, default=0.01)
    parser.add_argument("--max-rewire", type=float, default=0.3)
    parser.add_argument("--path-samples", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--clip-grad", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--cpu", action="store_true", help="debug with CPU workers")
    parser.add_argument("--run-dir", type=Path)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        "batch_size",
        "coarse_population",
        "coarse_generations",
        "coarse_epochs",
        "refine_population",
        "refine_generations",
        "refine_epochs",
        "graph_degree",
        "message_steps",
    )
    for name in positive:
        if getattr(args, name) < 1:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.refine_population > args.coarse_population:
        raise ValueError("refine-population cannot exceed coarse-population")
    if args.finalists > args.refine_population:
        raise ValueError("finalists cannot exceed refine-population")
    for name in ("coarse_epochs", "refine_epochs", "final_epochs"):
        epochs = getattr(args, name)
        minimum = 0 if name == "final_epochs" else 1
        if not minimum <= epochs <= MAX_EPOCHS:
            raise ValueError(
                f"{name.replace('_', '-')} must be between {minimum} and {MAX_EPOCHS}"
            )
    if args.coarse_generations > 1 and args.coarse_population < 2:
        raise ValueError("coarse-population must be >= 2 when evolving offspring")
    if args.refine_generations > 1 and args.refine_population < 2:
        raise ValueError("refine-population must be >= 2 when evolving offspring")
    if not 0.0 <= args.mutation_rate <= 1.0:
        raise ValueError("mutation-rate must be in [0, 1]")
    if args.graph_degree % 2 or args.graph_degree < 2:
        raise ValueError("graph-degree must be an even integer >= 2")
    for name in ("coarse_train_fraction", "refine_train_fraction"):
        if not 0.0 < getattr(args, name) <= 1.0:
            raise ValueError(f"{name.replace('_', '-')} must be in (0, 1]")
    if not 0.0 <= args.min_rewire <= args.max_rewire <= 1.0:
        raise ValueError("rewire bounds must satisfy 0 <= min <= max <= 1")


def build_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("liquid-evolution")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    for handler in (
        logging.FileHandler(run_dir / "evolution.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_liquid_model(
    model_name: str,
    topologies: list[SmallWorldTopology],
    message_steps: int,
) -> nn.Module:
    kwargs = {"topologies": topologies, "message_steps": message_steps, "num_classes": 100}
    if model_name == "deit-tiny":
        return deit_tiny_liquid(**kwargs)
    return swin_tiny_liquid(**kwargs)


def load_solid_state(path: str | None) -> dict[str, torch.Tensor] | None:
    if path is None:
        return None
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model", checkpoint)
    return {str(key): value for key, value in state.items()}


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_dtype: torch.dtype | None,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype or torch.float32,
            enabled=amp_dtype is not None,
        ):
            logits = model(images)
            loss = criterion(logits, targets)
        total_loss += loss.item() * targets.shape[0]
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_samples += targets.shape[0]
    return total_loss / total_samples, 100.0 * total_correct / total_samples


def train_one_candidate(
    task: dict[str, Any],
    config: dict[str, Any],
    device: torch.device,
    loaders: tuple[torch.utils.data.DataLoader, ...],
    solid_state: dict[str, torch.Tensor] | None,
) -> dict[str, Any]:
    started = time.monotonic()
    # Common random numbers make topology comparisons less noisy.
    seed = int(config["seed"])
    seed_everything(seed)
    topologies = [SmallWorldTopology.from_dict(item) for item in task["topologies"]]
    model = build_liquid_model(config["model"], topologies, config["message_steps"])
    if solid_state is not None:
        incompatible = model.load_state_dict(solid_state, strict=False)
        unexpected = [key for key in incompatible.unexpected_keys if "mlp" not in key]
        if unexpected:
            raise RuntimeError(f"unexpected solid checkpoint keys: {unexpected[:5]}")
    model.to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=config["label_smoothing"])
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    epochs = int(task["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs), eta_min=config["min_lr"]
    )
    amp_dtype = torch.bfloat16 if device.type == "cuda" else None
    train_loader, validation_loader, test_loader = loaders
    history = []
    run_label = f"{task['evaluation_tag']}-{task['identifier']}-e{epochs}"
    candidate_dir = Path(config["run_dir"]) / "candidates" / run_label
    candidate_dir.mkdir(parents=True, exist_ok=True)
    epoch_log = candidate_dir / "epochs.jsonl"
    candidate_writer = SummaryWriter(candidate_dir / "tensorboard")
    for epoch in range(epochs):
        epoch_started = time.monotonic()
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype or torch.float32,
                enabled=amp_dtype is not None,
            ):
                logits = model(images)
                loss = criterion(logits, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad"])
            optimizer.step()
            total_loss += loss.detach().item() * targets.shape[0]
            total_correct += (logits.detach().argmax(dim=1) == targets).sum().item()
            total_samples += targets.shape[0]
        validation_loss, validation_accuracy = evaluate_model(
            model, validation_loader, criterion, device, amp_dtype
        )
        epoch_record = {
            "train_loss": total_loss / total_samples,
            "train_accuracy": 100.0 * total_correct / total_samples,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_seconds": time.monotonic() - epoch_started,
            "peak_memory_gb": (
                torch.cuda.max_memory_allocated(device) / 1024**3
                if device.type == "cuda"
                else 0.0
            ),
        }
        history.append(epoch_record)
        with epoch_log.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"epoch": epoch + 1, **epoch_record}) + "\n")
        for key, value in epoch_record.items():
            candidate_writer.add_scalar(key, value, epoch + 1)
        candidate_writer.flush()
        scheduler.step()
    candidate_writer.close()

    checkpoint_path = None
    test_loss = None
    test_accuracy = None
    if task.get("final", False):
        test_loss, test_accuracy = evaluate_model(
            model, test_loader, criterion, device, amp_dtype
        )
        checkpoint_dir = Path(config["run_dir"]) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"{task['identifier']}-final.pt"
        torch.save(
            {
                "model": model.state_dict(),
                "topologies": [topology.to_dict() for topology in topologies],
                "history": history,
                "validation_accuracy": history[-1]["validation_accuracy"],
                "test_accuracy": test_accuracy,
                "config": config,
            },
            checkpoint_path,
        )
    result = {
        "identifier": task["identifier"],
        "loss": history[-1]["validation_loss"],
        "accuracy": history[-1]["validation_accuracy"],
        "training_history": history,
        "checkpoint": None if checkpoint_path is None else str(checkpoint_path),
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "device": str(device),
        "runtime_seconds": time.monotonic() - started,
        "peak_memory_gb": (
            torch.cuda.max_memory_allocated(device) / 1024**3
            if device.type == "cuda"
            else 0.0
        ),
    }
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def evaluation_worker(
    tasks: list[dict[str, Any]],
    config: dict[str, Any],
    device_name: str,
    output_queue: mp.Queue,
) -> None:
    try:
        device = torch.device(device_name)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        loaders = build_cifar100_evolution_loaders(
            config["data_dir"],
            config["batch_size"],
            config["workers"],
            config["validation_size"],
            config["train_fraction"],
            config["seed"],
            config["download"],
        )
        solid_state = load_solid_state(config.get("solid_checkpoint"))
        results = [
            train_one_candidate(task, config, device, loaders, solid_state) for task in tasks
        ]
        output_queue.put({"ok": True, "device": device_name, "results": results})
    except Exception:
        output_queue.put(
            {"ok": False, "device": device_name, "error": traceback.format_exc()}
        )


def evaluate_population(
    population: list[Individual],
    epochs: int,
    train_fraction: float,
    devices: list[str],
    config: dict[str, Any],
    logger: logging.Logger,
    evaluation_tag: str,
    final: bool = False,
) -> dict[str, dict[str, Any]]:
    for individual in population:
        if not individual.topology_metrics:
            individual.topology_metrics = measure_model_topologies(
                individual.topologies, config["path_samples"]
            )
    tasks = [
        {
            "identifier": item.identifier,
            "generation": item.generation,
            "topologies": [topology.to_dict() for topology in item.topologies],
            "epochs": epochs,
            "final": final,
            "evaluation_tag": evaluation_tag,
        }
        for item in population
    ]
    active_devices = devices[: min(len(devices), len(tasks))]
    partitions = [tasks[index:: len(active_devices)] for index in range(len(active_devices))]
    worker_config = {
        **config,
        "train_fraction": train_fraction,
    }
    logger.info(
        "evaluating candidates=%d epochs=%d train_fraction=%.2f devices=%s final=%s",
        len(population),
        epochs,
        train_fraction,
        active_devices,
        final,
    )
    context = mp.get_context("spawn")
    output_queue = context.Queue()
    processes = [
        context.Process(
            target=evaluation_worker,
            args=(partition, worker_config, device, output_queue),
        )
        for partition, device in zip(partitions, active_devices, strict=True)
    ]
    for process in processes:
        process.start()
    messages = []
    deadline = time.monotonic() + 7 * 24 * 60 * 60
    while len(messages) < len(processes):
        try:
            messages.append(output_queue.get(timeout=5))
        except Empty:
            failed = [process for process in processes if process.exitcode not in (None, 0)]
            if failed:
                raise RuntimeError(f"evaluation worker exited with code {failed[0].exitcode}")
            if time.monotonic() > deadline:
                raise TimeoutError("candidate evaluation exceeded seven days")
    for process in processes:
        process.join()
    errors = [message for message in messages if not message["ok"]]
    if errors:
        raise RuntimeError(errors[0]["error"])
    results = {
        result["identifier"]: result
        for message in messages
        for result in message["results"]
    }
    for individual in population:
        result = results[individual.identifier]
        individual.accuracy = float(result["accuracy"])
        individual.loss = float(result["loss"])
        individual.training_history = result["training_history"]
        individual.checkpoint = result["checkpoint"]
        individual.runtime_seconds = float(result["runtime_seconds"])
        individual.peak_memory_gb = float(result["peak_memory_gb"])
        logger.info(
            "candidate=%s device=%s val_acc=%.2f val_loss=%.4f small_world=%.4f "
            "runtime=%.1fs peak_memory=%.2fGB",
            individual.identifier,
            result["device"],
            individual.accuracy,
            individual.loss,
            individual.small_world_score,
            individual.runtime_seconds,
            individual.peak_memory_gb,
        )
    return results


def save_state(
    run_dir: Path,
    generation: int,
    phase: str,
    population: list[Individual],
) -> None:
    state = {
        "generation": generation,
        "phase": phase,
        "population": [individual.record_dict() for individual in population],
    }
    temporary = run_dir / "state.json.tmp"
    temporary.write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, run_dir / "state.json")


def run_phase(
    population: list[Individual],
    phase: str,
    start_generation: int,
    generations: int,
    epochs: int,
    train_fraction: float,
    population_size: int,
    devices: list[str],
    config: dict[str, Any],
    args: argparse.Namespace,
    recorder: EvolutionRecorder,
    logger: logging.Logger,
    reevaluate_initial: bool,
) -> tuple[list[Individual], int]:
    generation = start_generation
    if reevaluate_initial:
        evaluate_population(
            population,
            epochs,
            train_fraction,
            devices,
            config,
            logger,
            f"{phase}-g{generation:03d}",
        )
        population = select_survivors(population, population_size)
        recorder.record_generation(generation, phase, population)
        save_state(Path(config["run_dir"]), generation, phase, population)
        generation += 1
    for _ in range(1 if reevaluate_initial else 0, generations):
        offspring = make_offspring(
            population,
            population_size,
            generation,
            args.mutation_rate,
            args.crossover_probability,
            args.seed,
        )
        evaluate_population(
            offspring,
            epochs,
            train_fraction,
            devices,
            config,
            logger,
            f"{phase}-g{generation:03d}",
        )
        population = select_survivors(population + offspring, population_size)
        recorder.record_generation(generation, phase, population)
        save_state(Path(config["run_dir"]), generation, phase, population)
        generation += 1
    return population, generation


def main() -> None:
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{timestamp}-{args.model}-liquid-nsga"
    run_dir = args.run_dir or args.log_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    logger = build_logger(run_dir)
    devices = ["cpu"] if args.cpu else [f"cuda:{item.strip()}" for item in args.devices.split(",")]
    if not args.cpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --cpu only for a small smoke run")
    if not args.cpu:
        largest_device = max(int(item.split(":")[1]) for item in devices)
        if torch.cuda.device_count() <= largest_device:
            raise RuntimeError(
                f"requested devices {devices}, found {torch.cuda.device_count()} GPUs"
            )
    if args.solid_checkpoint is not None and not args.solid_checkpoint.exists():
        raise FileNotFoundError(args.solid_checkpoint)
    if args.solid_checkpoint is None:
        logger.warning(
            "no --solid-checkpoint supplied; short candidate training starts from scratch "
            "and accuracy will be a noisy topology objective"
        )

    config = {
        "model": args.model,
        "devices": devices,
        "data_dir": str(args.data_dir.resolve()),
        "run_dir": str(run_dir.resolve()),
        "solid_checkpoint": (
            None if args.solid_checkpoint is None else str(args.solid_checkpoint.resolve())
        ),
        "batch_size": args.batch_size,
        "workers": args.workers,
        "validation_size": args.validation_size,
        "message_steps": args.message_steps,
        "path_samples": args.path_samples,
        "lr": args.lr,
        "min_lr": args.min_lr,
        "weight_decay": args.weight_decay,
        "label_smoothing": args.label_smoothing,
        "clip_grad": args.clip_grad,
        "seed": args.seed,
        "download": args.download,
        "arguments": vars(args),
    }
    (run_dir / "config.json").write_text(
        json.dumps(config, default=str, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("configuration=%s", json.dumps(config, default=str, ensure_ascii=False))
    node_counts = (
        deit_topology_nodes() if args.model == "deit-tiny" else swin_topology_nodes()
    )
    population = create_population(
        node_counts,
        args.coarse_population,
        args.graph_degree,
        args.seed,
        args.min_rewire,
        args.max_rewire,
    )
    recorder = EvolutionRecorder(run_dir, args.model)
    started = time.monotonic()
    try:
        population, generation = run_phase(
            population,
            "coarse",
            0,
            args.coarse_generations,
            args.coarse_epochs,
            args.coarse_train_fraction,
            args.coarse_population,
            devices,
            config,
            args,
            recorder,
            logger,
            reevaluate_initial=True,
        )
        population = select_survivors(population, args.refine_population)
        population, generation = run_phase(
            population,
            "refine",
            generation,
            args.refine_generations,
            args.refine_epochs,
            args.refine_train_fraction,
            args.refine_population,
            devices,
            config,
            args,
            recorder,
            logger,
            reevaluate_initial=True,
        )

        finalists = select_survivors(population, args.finalists)
        final_results = {}
        if args.final_epochs > 0 and finalists:
            final_results = evaluate_population(
                finalists,
                args.final_epochs,
                args.refine_train_fraction,
                devices,
                config,
                logger,
                f"final-g{generation:03d}",
                final=True,
            )
            recorder.record_generation(generation, "final", finalists)
        final_payload = []
        for individual in finalists:
            record = individual.record_dict()
            if individual.identifier in final_results:
                record["test_loss"] = final_results[individual.identifier]["test_loss"]
                record["test_accuracy"] = final_results[individual.identifier]["test_accuracy"]
            final_payload.append(record)
        (run_dir / "finalists.json").write_text(
            json.dumps(final_payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(
            "evolution complete elapsed=%.2fh finalists=%s",
            (time.monotonic() - started) / 3600,
            [item.identifier for item in finalists],
        )
    finally:
        recorder.close()


if __name__ == "__main__":
    main()
