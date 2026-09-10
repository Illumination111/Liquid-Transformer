from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for directory in (ROOT, ROOT / "model-liquid"):
    sys.path.insert(0, str(directory))

from small_world import is_connected, watts_strogatz_topology  # noqa: E402

from train.rl.ppo import PPO, ActorCritic, compute_gae  # noqa: E402
from train.rl.topology import apply_actions, rewire  # noqa: E402
from train.search import parse_args, run, validate_args  # noqa: E402


def test_gae_terminal_boundaries_and_bootstrap():
    advantages, returns = compute_gae(
        torch.tensor([1.0, 2.0, 3.0]),
        torch.zeros(3),
        torch.tensor([0.0, 1.0, 0.0]),
        bootstrap=4.0,
        gamma=1.0,
        gae_lambda=1.0,
    )
    torch.testing.assert_close(returns, torch.tensor([3.0, 2.0, 7.0]))
    torch.testing.assert_close(advantages, returns)


def test_ppo_increases_probability_of_a_rewarded_action():
    torch.manual_seed(3)
    policy = ActorCritic(2, 1, 2)
    for parameter in policy.critic.parameters():
        torch.nn.init.zeros_(parameter)
    ppo = PPO(policy, lr=0.01, entropy_coef=0, update_epochs=2)
    state, action = torch.zeros(2), torch.tensor([1])
    with torch.no_grad():
        before = policy(state)[0].probs[0, 1].item()
        log_prob, _, value = policy.evaluate(state, action)
    stats = ppo.update(
        [dict(state=state, actions=action, log_prob=log_prob, value=value, reward=1.0, done=True)]
    )
    assert policy(state)[0].probs[0, 1].item() > before
    assert all(torch.isfinite(torch.tensor(value)) for value in stats.values())


def test_policy_sampling_uses_private_rng_and_log_probs_match():
    policy = ActorCritic(2, 2, 4)
    state = torch.zeros(2)
    first_rng = torch.Generator().manual_seed(7)
    second_rng = torch.Generator().manual_seed(7)
    first = [policy.act(state, first_rng)[0] for _ in range(5)]
    second = []
    for _ in range(5):
        torch.manual_seed(42)  # candidate initialization must not reset the policy stream
        action, log_prob, _ = policy.act(state, second_rng)
        second.append(action)
        torch.testing.assert_close(log_prob, policy.evaluate(state, action)[0])
    assert all(torch.equal(a, b) for a, b in zip(first, second, strict=True))


def test_topology_actions_preserve_invariants_and_reproduce():
    graph = watts_strogatz_topology(32, 4, seed=42)
    assert rewire(graph, 0, random.Random(1)) is graph
    left = rewire(graph, 0.2, random.Random(1))
    right = rewire(graph, 0.2, random.Random(1))
    assert left == right and left.edges != graph.edges
    assert is_connected(left.num_nodes, left.edges)
    assert len(left.edges) == len(graph.edges) and left.seed == graph.seed
    with pytest.raises(ValueError):
        apply_actions([graph], torch.tensor([4]), random.Random(1))


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--lif-beta", "1"),
        ("--candidate-epochs", "0"),
        ("--topology-weight", "nan"),
        ("--graph-degree", "3"),
    ],
)
def test_search_rejects_invalid_arguments(flag, value):
    with pytest.raises(ValueError):
        validate_args(parse_args([flag, value]))


def test_synthetic_pipeline_updates_policy_and_roundtrips_checkpoint(tmp_path, monkeypatch):
    import train.search as search
    from train.rl.evaluation import load_snn_checkpoint

    def forbid_real_data(*args, **kwargs):
        raise AssertionError("synthetic smoke check must not access CIFAR-100")

    monkeypatch.setattr(search, "build_cifar100_search_loaders", forbid_real_data)
    monkeypatch.setattr(search, "build_cifar100_test_loader", forbid_real_data)
    run_dir = run(
        parse_args(["--smoke-test", "--device", "cpu", "--run-dir", str(tmp_path / "smoke")])
    )
    final = json.loads((run_dir / "final.json").read_text())
    config = json.loads((run_dir / "config.json").read_text())
    selected = json.loads((run_dir / "selected.json").read_text())
    assert final["synthetic"] and config["synthetic"] and selected["synthetic"]
    assert config["arguments"]["validation_size"] == 4
    checkpoint = torch.load(final["checkpoint"], weights_only=True)
    assert checkpoint["train_config"]["synthetic"]
    assert checkpoint["model_config"]["image_size"] == 32
    assert len((run_dir / "candidates.jsonl").read_text().splitlines()) == 3
    assert len((run_dir / "transitions.jsonl").read_text().splitlines()) == 2
    assert len((run_dir / "ppo.jsonl").read_text().splitlines()) == 1
    policy = torch.load(run_dir / "policy.pt", weights_only=True)
    assert policy["optimizer"]["state"]  # a real PPO optimization step ran
    model = load_snn_checkpoint(Path(final["checkpoint"])).eval()
    with torch.no_grad():
        assert torch.isfinite(model(torch.randn(1, 3, 32, 32))).all()
    candidates = [
        json.loads(line) for line in (run_dir / "candidates.jsonl").read_text().splitlines()
    ]
    assert selected["candidate"]["quality"] == max(item["quality"] for item in candidates)


def test_final_checkpoint_saves_best_validation_epoch_not_last(tmp_path, monkeypatch):
    from train.rl import evaluation

    epoch = 0

    def fake_epoch(model, loader, criterion, device, optimizer=None, *args, **kwargs):
        nonlocal epoch
        if optimizer is not None:
            optimizer.step()
            epoch += 1
            with torch.no_grad():
                model.head.bias.fill_(epoch)
            return 1.0, 10.0
        return (0.3, 90.0) if epoch == 1 else (0.9, 20.0)

    monkeypatch.setattr(evaluation, "run_epoch", fake_epoch)
    config = dict(embed_dim=24, depth=1, num_heads=3, time_steps=2, message_steps=1)
    trainer = evaluation.CandidateEvaluator(
        config,
        dict(seed=42, lr=1e-3, weight_decay=0.05, label_smoothing=0.1),
        lambda: ([None], [None]),
        "cpu",
        tmp_path,
    )
    result = trainer([watts_strogatz_topology(96, 4)], 2, "final", final=True)
    restored = evaluation.load_snn_checkpoint(Path(result["checkpoint"]))
    assert result["best_epoch"] == 1 and result["accuracy"] == 90.0
    assert torch.all(restored.head.bias == 1)


def test_search_without_final_training_never_loads_test_data(tmp_path, monkeypatch):
    import train.search as search

    class StubEvaluator:
        def __init__(self, *args):
            pass

        def __call__(self, graphs, epochs, identifier):
            return {"identifier": identifier, "accuracy": 1.0, "loss": 4.0}

    def forbidden(*args, **kwargs):
        raise AssertionError("test data must remain unopened during search")

    monkeypatch.setattr(search, "CandidateEvaluator", StubEvaluator)
    monkeypatch.setattr(
        search, "measure_model_topologies", lambda *args: {"small_world_score": 1.0}
    )
    monkeypatch.setattr(search, "build_cifar100_test_loader", forbidden)
    run_dir = run(
        parse_args(
            [
                "--device",
                "cpu",
                "--iterations",
                "0",
                "--final-epochs",
                "0",
                "--run-dir",
                str(tmp_path / "search-only"),
            ]
        )
    )
    assert (run_dir / "selected.json").exists()
    assert not (run_dir / "final.json").exists()


def test_direct_cli_entry_point_imports_from_outside_repository(tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "train/search.py"), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--smoke-test" in result.stdout and "--ppo-epochs" in result.stdout
