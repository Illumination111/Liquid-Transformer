"""Small, dependency-free NSGA-II implementation for graph genomes."""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field

from small_world import (
    SmallWorldTopology,
    canonical_edge,
    is_connected,
    watts_strogatz_topology,
)


@dataclass
class Individual:
    identifier: str
    topologies: list[SmallWorldTopology]
    generation: int = 0
    parent_ids: tuple[str, ...] = ()
    accuracy: float | None = None
    loss: float | None = None
    topology_metrics: dict[str, object] = field(default_factory=dict)
    training_history: list[dict[str, float]] = field(default_factory=list)
    rank: int = 0
    crowding: float = 0.0
    checkpoint: str | None = None
    runtime_seconds: float = 0.0
    peak_memory_gb: float = 0.0

    @property
    def small_world_score(self) -> float:
        return float(self.topology_metrics.get("small_world_score", -math.inf))

    def evaluated(self) -> bool:
        return self.accuracy is not None and self.loss is not None

    def genome_dict(self) -> dict[str, object]:
        return {
            "identifier": self.identifier,
            "generation": self.generation,
            "parent_ids": list(self.parent_ids),
            "topologies": [topology.to_dict() for topology in self.topologies],
        }

    def record_dict(self) -> dict[str, object]:
        return {
            **self.genome_dict(),
            "accuracy": self.accuracy,
            "loss": self.loss,
            "topology_metrics": self.topology_metrics,
            "training_history": self.training_history,
            "rank": self.rank,
            "crowding": self.crowding,
            "checkpoint": self.checkpoint,
            "runtime_seconds": self.runtime_seconds,
            "peak_memory_gb": self.peak_memory_gb,
        }

    @classmethod
    def from_record(cls, data: dict[str, object]) -> Individual:
        individual = cls(
            identifier=str(data["identifier"]),
            generation=int(data.get("generation", 0)),
            parent_ids=tuple(str(value) for value in data.get("parent_ids", [])),  # type: ignore[arg-type]
            topologies=[
                SmallWorldTopology.from_dict(topology)
                for topology in data["topologies"]  # type: ignore[union-attr]
            ],
        )
        individual.accuracy = (
            None if data.get("accuracy") is None else float(data["accuracy"])
        )
        individual.loss = None if data.get("loss") is None else float(data["loss"])
        individual.topology_metrics = dict(data.get("topology_metrics", {}))  # type: ignore[arg-type]
        individual.training_history = list(data.get("training_history", []))  # type: ignore[arg-type]
        individual.rank = int(data.get("rank", 0))
        individual.crowding = float(data.get("crowding", 0.0))
        individual.checkpoint = (
            None if data.get("checkpoint") is None else str(data["checkpoint"])
        )
        individual.runtime_seconds = float(data.get("runtime_seconds", 0.0))
        individual.peak_memory_gb = float(data.get("peak_memory_gb", 0.0))
        return individual


def create_population(
    node_counts: list[int],
    size: int,
    degree: int,
    seed: int,
    minimum_rewire: float = 0.01,
    maximum_rewire: float = 0.3,
) -> list[Individual]:
    rng = random.Random(seed)
    population = []
    for index in range(size):
        probability = rng.uniform(minimum_rewire, maximum_rewire)
        topologies = [
            watts_strogatz_topology(
                nodes, degree, probability, seed + index * 1009 + block
            )
            for block, nodes in enumerate(node_counts)
        ]
        population.append(Individual(f"g000-i{index:03d}", topologies))
    return population


def dominates(left: Individual, right: Individual) -> bool:
    if not left.evaluated() or not right.evaluated():
        raise ValueError("dominance requires evaluated individuals")
    left_objectives = (float(left.accuracy), left.small_world_score)
    right_objectives = (float(right.accuracy), right.small_world_score)
    return all(a >= b for a, b in zip(left_objectives, right_objectives, strict=True)) and any(
        a > b for a, b in zip(left_objectives, right_objectives, strict=True)
    )


def rank_and_crowding(population: list[Individual]) -> list[list[Individual]]:
    dominates_set: dict[str, list[Individual]] = {item.identifier: [] for item in population}
    dominated_count = {item.identifier: 0 for item in population}
    fronts: list[list[Individual]] = [[]]
    for left in population:
        for right in population:
            if left is right:
                continue
            if dominates(left, right):
                dominates_set[left.identifier].append(right)
            elif dominates(right, left):
                dominated_count[left.identifier] += 1
        if dominated_count[left.identifier] == 0:
            left.rank = 0
            fronts[0].append(left)
    index = 0
    while index < len(fronts) and fronts[index]:
        next_front = []
        for left in fronts[index]:
            for right in dominates_set[left.identifier]:
                dominated_count[right.identifier] -= 1
                if dominated_count[right.identifier] == 0:
                    right.rank = index + 1
                    next_front.append(right)
        if next_front:
            fronts.append(next_front)
        index += 1

    for front in fronts:
        for item in front:
            item.crowding = 0.0
        if len(front) <= 2:
            for item in front:
                item.crowding = math.inf
            continue
        for objective in ("accuracy", "small_world_score"):
            ordered = sorted(
                front,
                key=lambda item: (
                    float(item.accuracy)
                    if objective == "accuracy"
                    else item.small_world_score
                ),
            )
            ordered[0].crowding = ordered[-1].crowding = math.inf
            low = (
                float(ordered[0].accuracy)
                if objective == "accuracy"
                else ordered[0].small_world_score
            )
            high = (
                float(ordered[-1].accuracy)
                if objective == "accuracy"
                else ordered[-1].small_world_score
            )
            if high == low:
                continue
            for position in range(1, len(ordered) - 1):
                previous = (
                    float(ordered[position - 1].accuracy)
                    if objective == "accuracy"
                    else ordered[position - 1].small_world_score
                )
                following = (
                    float(ordered[position + 1].accuracy)
                    if objective == "accuracy"
                    else ordered[position + 1].small_world_score
                )
                ordered[position].crowding += (following - previous) / (high - low)
    return fronts


def select_survivors(population: list[Individual], size: int) -> list[Individual]:
    fronts = rank_and_crowding(population)
    selected = []
    for front in fronts:
        if len(selected) + len(front) <= size:
            selected.extend(front)
        else:
            selected.extend(
                sorted(front, key=lambda item: item.crowding, reverse=True)[
                    : size - len(selected)
                ]
            )
            break
    return selected


def mutate_topology(
    topology: SmallWorldTopology, mutation_rate: float, rng: random.Random
) -> SmallWorldTopology:
    if mutation_rate <= 0.0:
        return topology
    edges = set(topology.edges)
    attempts = max(1, round(len(edges) * mutation_rate))
    successful = 0
    for _ in range(attempts * 4):
        if successful >= attempts:
            break
        old_edge = rng.choice(tuple(edges))
        source = old_edge[rng.randrange(2)]
        candidate = rng.randrange(topology.num_nodes)
        if candidate == source:
            continue
        new_edge = canonical_edge(source, candidate)
        if new_edge in edges:
            continue
        edges.remove(old_edge)
        edges.add(new_edge)
        if is_connected(topology.num_nodes, edges):
            successful += 1
        else:
            edges.remove(new_edge)
            edges.add(old_edge)
    return SmallWorldTopology(
        topology.num_nodes,
        tuple(sorted(edges)),
        topology.degree,
        topology.rewire_probability,
        rng.randrange(2**31),
    )


def _tournament(population: list[Individual], rng: random.Random) -> Individual:
    left, right = rng.sample(population, 2)
    if left.rank != right.rank:
        return left if left.rank < right.rank else right
    if left.crowding != right.crowding:
        return left if left.crowding > right.crowding else right
    return rng.choice((left, right))


def make_offspring(
    population: list[Individual],
    size: int,
    generation: int,
    mutation_rate: float,
    crossover_probability: float,
    seed: int,
) -> list[Individual]:
    rank_and_crowding(population)
    rng = random.Random(seed + generation * 100_003)
    children = []
    for index in range(size):
        first = _tournament(population, rng)
        second = _tournament(population, rng)
        topologies = []
        for left, right in zip(first.topologies, second.topologies, strict=True):
            chosen = right if rng.random() < crossover_probability else left
            topologies.append(mutate_topology(chosen, mutation_rate, rng))
        children.append(
            Individual(
                identifier=f"g{generation:03d}-i{index:03d}",
                topologies=copy.deepcopy(topologies),
                generation=generation,
                parent_ids=(first.identifier, second.identifier),
            )
        )
    return children
