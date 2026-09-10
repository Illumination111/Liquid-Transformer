"""Connected, fixed-edge-count topology actions and PPO observations."""

from __future__ import annotations

import math
import random

import torch
from small_world import SmallWorldTopology, canonical_edge, is_connected

ACTION_RATES = (0.0, 0.01, 0.05, 0.1)


def rewire(topology: SmallWorldTopology, rate: float, rng: random.Random) -> SmallWorldTopology:
    if not 0 <= rate <= 1:
        raise ValueError("rewire rate must be in [0, 1]")
    if rate == 0:
        return topology
    edges = set(topology.edges)
    target = max(1, round(len(edges) * rate))
    changed = 0
    for _ in range(target * 4):
        if changed >= target:
            break
        old_edge = rng.choice(sorted(edges))
        source = old_edge[rng.randrange(2)]
        destination = rng.randrange(topology.num_nodes)
        if destination == source:
            continue
        new_edge = canonical_edge(source, destination)
        if new_edge in edges:
            continue
        edges.remove(old_edge)
        edges.add(new_edge)
        if is_connected(topology.num_nodes, edges):
            changed += 1
        else:
            edges.remove(new_edge)
            edges.add(old_edge)
    # Keep the metric sampling seed fixed so unchanged graphs retain identical scores.
    return SmallWorldTopology(
        topology.num_nodes,
        tuple(sorted(edges)),
        topology.degree,
        topology.rewire_probability,
        topology.seed,
    )


def apply_actions(topologies, actions: torch.Tensor, rng: random.Random):
    if len(actions) != len(topologies):
        raise ValueError("one action is required per block")
    if any(int(action) < 0 or int(action) >= len(ACTION_RATES) for action in actions):
        raise ValueError("invalid topology action")
    return [
        rewire(graph, ACTION_RATES[int(action)], rng)
        for graph, action in zip(topologies, actions, strict=True)
    ]


def observation(metrics: dict, accuracy: float, remaining: float) -> torch.Tensor:
    features = [accuracy / 100.0, remaining]
    for block in metrics["blocks"]:
        features.extend(
            (
                block["clustering"],
                block["average_path_length"] / max(1.0, math.log(block["nodes"])),
                block["log_sigma"] / 10.0,
            )
        )
    return torch.tensor(features, dtype=torch.float32)


def quality(accuracy: float, metrics: dict, topology_weight: float) -> float:
    return accuracy / 100.0 + topology_weight * float(metrics["small_world_score"])
