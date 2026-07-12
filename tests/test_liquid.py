from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
for directory in ("model-liquid", "model-solid", "train/evolution"):
    sys.path.insert(0, str(ROOT / directory))

from deit_tiny import deit_tiny  # noqa: E402
from deit_tiny_liquid import deit_tiny_liquid  # noqa: E402
from metrics import measure_topology  # noqa: E402
from nsga2 import Individual, mutate_topology, select_survivors  # noqa: E402
from small_world import SmallWorldFFN, is_connected, watts_strogatz_topology  # noqa: E402
from swin_tiny_liquid import swin_tiny_liquid  # noqa: E402


@pytest.mark.parametrize("factory", [deit_tiny_liquid, swin_tiny_liquid])
def test_liquid_model_output_shape(factory):
    model = factory(graph_degree=4, message_steps=1).eval()
    with torch.inference_mode():
        logits = model(torch.randn(1, 3, 32, 32))
    assert logits.shape == (1, 100)


def test_small_world_ffn_backpropagates_to_edges():
    topology = watts_strogatz_topology(32, degree=4, rewire_probability=0.1, seed=7)
    layer = SmallWorldFFN(8, 32, topology, message_steps=2)
    output = layer(torch.randn(2, 5, 8))
    output.square().mean().backward()
    assert output.shape == (2, 5, 8)
    assert layer.edge_weight.grad is not None
    assert torch.isfinite(layer.edge_weight.grad).all()


def test_solid_checkpoint_initializes_matching_liquid_weights():
    solid = deit_tiny()
    liquid = deit_tiny_liquid(graph_degree=4, message_steps=1)
    incompatible = liquid.load_state_dict(solid.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert torch.equal(liquid.blocks[0].mlp.fc1.weight, solid.blocks[0].mlp.fc1.weight)
    assert any("edge_weight" in key for key in incompatible.missing_keys)


def test_topology_metrics_and_mutation_remain_connected():
    topology = watts_strogatz_topology(64, degree=8, rewire_probability=0.1, seed=11)
    import random

    mutated = mutate_topology(topology, mutation_rate=0.1, rng=random.Random(12))
    unchanged = mutate_topology(topology, mutation_rate=0.0, rng=random.Random(12))
    metrics, communities = measure_topology(mutated, sample_sources=16)
    assert is_connected(mutated.num_nodes, mutated.edges)
    assert len(mutated.edges) == len(topology.edges)
    assert unchanged is topology
    assert metrics.clustering > 0
    assert metrics.sigma > 1
    assert len(communities) == metrics.communities


def test_nsga_selection_preserves_accuracy_and_topology_extremes():
    topology = watts_strogatz_topology(32, degree=4, rewire_probability=0.1, seed=1)
    values = [(80.0, 2.0), (75.0, 3.0), (70.0, 1.0)]
    population = []
    for index, (accuracy, score) in enumerate(values):
        individual = Individual(f"i{index}", [topology], accuracy=accuracy, loss=1.0)
        individual.topology_metrics = {"small_world_score": score}
        population.append(individual)
    selected = select_survivors(population, 2)
    assert {item.identifier for item in selected} == {"i0", "i1"}
