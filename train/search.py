#!/usr/bin/env python3
"""PPO topology search for hybrid DeiT with LIF FFNs (analog attention)."""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "model-liquid", ROOT / "model-solid", ROOT / "train/evolution"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

# Direct execution puts train/ first: force the project package ahead of train.py.
sys.path.insert(0, str(ROOT))

from metrics import measure_model_topologies  # noqa: E402
from small_world import watts_strogatz_topology  # noqa: E402

from dataset.cifar100 import (  # noqa: E402
    DEFAULT_DATA_DIR,
    build_cifar100_search_loaders,
    build_cifar100_test_loader,
)
from train.rl.evaluation import CandidateEvaluator  # noqa: E402
from train.rl.ppo import PPO, ActorCritic  # noqa: E402
from train.rl.topology import ACTION_RATES, apply_actions, observation, quality  # noqa: E402
from train.train import seed_everything  # noqa: E402


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--rollout-steps", type=int, default=4)
    parser.add_argument("--candidate-epochs", type=int, default=2)
    parser.add_argument("--final-epochs", type=int, default=0)
    parser.add_argument("--train-fraction", type=float, default=1.0)
    parser.add_argument("--validation-size", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--time-steps", type=int, default=4)
    parser.add_argument("--lif-beta", type=float, default=0.5, help="membrane decay in (0, 1)")
    parser.add_argument("--lif-threshold", type=float, default=1.0)
    parser.add_argument(
        "--surrogate-slope", type=float, default=5.0, help="SpikingJelly ATan surrogate alpha"
    )
    parser.add_argument("--message-steps", type=int, default=2)
    parser.add_argument("--graph-degree", type=int, default=8)
    parser.add_argument("--path-samples", type=int, default=64)
    parser.add_argument("--topology-weight", type=float, default=0.01)
    parser.add_argument("--policy-lr", type=float, default=3e-4)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download", action="store_true")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="bounded synthetic-data integration check, not an experiment",
    )
    return parser.parse_args(argv)


def validate_args(args):
    for name in (
        "rollout_steps",
        "candidate_epochs",
        "batch_size",
        "time_steps",
        "message_steps",
        "path_samples",
        "ppo_epochs",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.iterations < 0 or args.workers < 0 or not 0 <= args.final_epochs <= 100:
        raise ValueError("invalid iterations, workers, or final_epochs (0-100)")
    if args.candidate_epochs > 100 or not 1 <= args.validation_size < 50000:
        raise ValueError("invalid candidate epochs (1-100) or validation size")
    for name in ("lif_threshold", "surrogate_slope", "lr", "policy_lr"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            raise ValueError(f"{name} must be finite and positive")
    for name in ("topology_weight", "entropy_coef", "weight_decay"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not 0 < args.lif_beta < 1 or not 0 < args.train_fraction <= 1:
        raise ValueError("invalid lif_beta or train_fraction")
    if not 0 < args.clip_epsilon < 1 or not 0 <= args.label_smoothing < 1:
        raise ValueError("invalid clip_epsilon or label_smoothing")
    if not 0 <= args.gamma <= 1 or not 0 <= args.gae_lambda <= 1:
        raise ValueError("gamma and gae_lambda must be in [0, 1]")
    if args.graph_degree < 2 or args.graph_degree % 2 or args.graph_degree >= 768:
        raise ValueError("graph_degree must be even and in [2, 766]")


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def smoke_loaders():
    from torchvision.datasets import FakeData
    from torchvision.transforms import ToTensor

    return tuple(
        DataLoader(FakeData(size, (3, 32, 32), 100, ToTensor(), random_offset=offset), batch_size=4)
        for size, offset in ((8, 0), (4, 100))
    )


def run(args):
    validate_args(args)
    if args.smoke_test:
        # Explicit bounded configuration, independent of experiment defaults.
        args.iterations, args.rollout_steps, args.candidate_epochs, args.final_epochs = 1, 2, 1, 1
        args.workers, args.batch_size = 0, 4
        args.validation_size, args.train_fraction = 4, 1.0
        args.time_steps, args.message_steps, args.graph_degree, args.path_samples = 2, 1, 4, 4
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; select --device cpu")
        torch.cuda.set_device(device)
    elif device.type != "cpu":
        raise ValueError("supported devices are cpu and cuda:<index>")
    run_dir = args.run_dir or ROOT / "train-log" / (
        datetime.now().strftime("%Y%m%d-%H%M%S-%f") + "-deit-snn-ppo"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    args.run_dir = run_dir
    model_config = dict(
        image_size=32,
        patch_size=4,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        time_steps=args.time_steps,
        lif_beta=args.lif_beta,
        lif_threshold=args.lif_threshold,
        surrogate_slope=args.surrogate_slope,
        message_steps=args.message_steps,
        num_classes=100,
    )
    if args.smoke_test:
        model_config.update(embed_dim=24, depth=2, num_heads=3, drop_path_rate=0.0)
    depth, width = model_config.get("depth", 12), model_config.get("embed_dim", 192)
    graphs = [
        watts_strogatz_topology(4 * width, args.graph_degree, 0.1, args.seed + index)
        for index in range(depth)
    ]
    train_config = {
        key: getattr(args, key) for key in ("seed", "lr", "weight_decay", "label_smoothing")
    }
    train_config.update(
        data_dir=str(args.data_dir),
        validation_size=args.validation_size,
        train_fraction=args.train_fraction,
        synthetic=args.smoke_test,
        spikingjelly_version=version("spikingjelly"),
        surrogate="ATan",
    )

    def loader_factory():
        if args.smoke_test:
            return smoke_loaders()
        return build_cifar100_search_loaders(
            args.data_dir,
            args.batch_size,
            args.workers,
            args.validation_size,
            args.train_fraction,
            args.seed,
            args.download,
        )

    evaluator = CandidateEvaluator(model_config, train_config, loader_factory, device, run_dir)
    policy = ActorCritic(2 + depth * 3, depth, len(ACTION_RATES))
    ppo = PPO(
        policy,
        args.policy_lr,
        args.clip_epsilon,
        args.entropy_coef,
        update_epochs=args.ppo_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
    )
    policy_rng = torch.Generator().manual_seed(args.seed + 1000)
    topology_rng = random.Random(args.seed + 2000)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()
    config = {
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "model_config": model_config,
        "train_config": train_config,
        "code_revision": revision,
        "code_dirty": bool(dirty),
        "synthetic": args.smoke_test,
        "torch_version": str(torch.__version__),
        "action_rates": ACTION_RATES,
    }
    write_json(run_dir / "config.json", config)

    def evaluate(graphs, identifier):
        result = evaluator(graphs, args.candidate_epochs, identifier)
        result["topology_metrics"] = measure_model_topologies(graphs, args.path_samples)
        result["quality"] = quality(
            result["accuracy"], result["topology_metrics"], args.topology_weight
        )
        with (run_dir / "candidates.jsonl").open("a") as stream:
            stream.write(json.dumps(result, allow_nan=False) + "\n")
        print(
            f"{identifier}: val_acc={result['accuracy']:.2f}% quality={result['quality']:.5f}",
            flush=True,
        )
        return result

    initial = evaluate(graphs, "initial")
    best, best_graphs = initial, graphs
    for iteration in range(args.iterations):
        current, current_graphs = initial, graphs
        rollout = []
        for step in range(args.rollout_steps):
            state = observation(
                current["topology_metrics"], current["accuracy"], 1 - step / args.rollout_steps
            )
            actions, log_prob, value = policy.act(state, policy_rng)
            next_graphs = apply_actions(current_graphs, actions, topology_rng)
            result = evaluate(next_graphs, f"iteration-{iteration:03d}-step-{step:03d}")
            reward = result["quality"] - current["quality"]
            rollout.append(
                dict(
                    state=state,
                    actions=actions,
                    log_prob=log_prob,
                    value=value,
                    reward=reward,
                    done=step == args.rollout_steps - 1,
                )
            )
            with (run_dir / "transitions.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        dict(
                            iteration=iteration,
                            step=step,
                            state=state.tolist(),
                            actions=actions.tolist(),
                            reward=reward,
                            identifier=result["identifier"],
                        )
                    )
                    + "\n"
                )
            if result["quality"] > best["quality"]:
                best, best_graphs = result, next_graphs
            current, current_graphs = result, next_graphs
        stats = ppo.update(rollout)
        with (run_dir / "ppo.jsonl").open("a") as stream:
            stream.write(json.dumps(dict(iteration=iteration, **stats), allow_nan=False) + "\n")
        torch.save(
            {
                "policy": policy.state_dict(),
                "optimizer": ppo.optimizer.state_dict(),
                "iteration": iteration,
                "config": config,
                "policy_rng": policy_rng.get_state(),
            },
            run_dir / "policy.pt",
        )
    selected = {
        "synthetic": args.smoke_test,
        "candidate": best,
        "topologies": [graph.to_dict() for graph in best_graphs],
    }
    write_json(run_dir / "selected.json", selected)
    if args.final_epochs:
        result = evaluator(best_graphs, args.final_epochs, "final", final=True)
        # Only now, after topology and checkpoint selection, open the test partition.
        if args.smoke_test:
            from torchvision.datasets import FakeData
            from torchvision.transforms import ToTensor

            test_loader = DataLoader(
                FakeData(4, (3, 32, 32), 100, ToTensor(), random_offset=200), batch_size=4
            )
        else:
            test_loader = build_cifar100_test_loader(
                args.data_dir, args.batch_size, args.workers, args.download
            )
        result.update(evaluator.test(Path(result["checkpoint"]), test_loader))
        write_json(run_dir / "final.json", {"synthetic": args.smoke_test, **result})
    print(f"{'SYNTHETIC CHECK' if args.smoke_test else 'Search'} complete: {run_dir}", flush=True)
    return run_dir


if __name__ == "__main__":
    run(parse_args())
