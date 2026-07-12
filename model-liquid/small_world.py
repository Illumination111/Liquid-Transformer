"""Differentiable sparse small-world feed-forward layers and graph utilities."""

from __future__ import annotations

import math
import random
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass

import torch
from torch import nn

Edge = tuple[int, int]


def canonical_edge(source: int, target: int) -> Edge:
    if source == target:
        raise ValueError("small-world graphs do not use self edges")
    return (source, target) if source < target else (target, source)


def is_connected(num_nodes: int, edges: Iterable[Edge]) -> bool:
    if num_nodes < 1:
        return False
    adjacency = [[] for _ in range(num_nodes)]
    for source, target in edges:
        adjacency[source].append(target)
        adjacency[target].append(source)
    visited = {0}
    queue = deque([0])
    while queue:
        node = queue.popleft()
        for neighbor in adjacency[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    return len(visited) == num_nodes


@dataclass(frozen=True)
class SmallWorldTopology:
    """Serializable undirected graph used as an evolutionary genome block."""

    num_nodes: int
    edges: tuple[Edge, ...]
    degree: int
    rewire_probability: float
    seed: int

    def __post_init__(self) -> None:
        normalized = tuple(sorted({canonical_edge(*edge) for edge in self.edges}))
        if normalized != self.edges:
            object.__setattr__(self, "edges", normalized)
        if any(node < 0 or node >= self.num_nodes for edge in self.edges for node in edge):
            raise ValueError("topology contains a node outside the graph")
        if not is_connected(self.num_nodes, self.edges):
            raise ValueError("small-world topology must be connected")

    def to_dict(self) -> dict[str, object]:
        return {
            "num_nodes": self.num_nodes,
            "edges": [list(edge) for edge in self.edges],
            "degree": self.degree,
            "rewire_probability": self.rewire_probability,
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> SmallWorldTopology:
        return cls(
            num_nodes=int(data["num_nodes"]),
            edges=tuple((int(edge[0]), int(edge[1])) for edge in data["edges"]),  # type: ignore[index,union-attr]
            degree=int(data["degree"]),
            rewire_probability=float(data.get("rewire_probability", 0.0)),
            seed=int(data.get("seed", 0)),
        )


def watts_strogatz_topology(
    num_nodes: int,
    degree: int = 8,
    rewire_probability: float = 0.1,
    seed: int = 0,
) -> SmallWorldTopology:
    """Generate a connected Watts-Strogatz graph without external dependencies."""
    if degree < 2 or degree % 2:
        raise ValueError("degree must be an even integer >= 2")
    if degree >= num_nodes:
        raise ValueError("degree must be smaller than num_nodes")
    if not 0.0 <= rewire_probability <= 1.0:
        raise ValueError("rewire_probability must be between 0 and 1")

    for attempt in range(32):
        rng = random.Random(seed + attempt)
        edges = {
            canonical_edge(node, (node + offset) % num_nodes)
            for node in range(num_nodes)
            for offset in range(1, degree // 2 + 1)
        }
        original_edges = sorted(edges)
        for source, target in original_edges:
            if rng.random() >= rewire_probability or (source, target) not in edges:
                continue
            candidates = [
                node
                for node in range(num_nodes)
                if node != source and canonical_edge(source, node) not in edges
            ]
            if not candidates:
                continue
            replacement = rng.choice(candidates)
            edges.remove((source, target))
            edges.add(canonical_edge(source, replacement))
        normalized = tuple(sorted(edges))
        if is_connected(num_nodes, normalized):
            return SmallWorldTopology(
                num_nodes, normalized, degree, rewire_probability, seed + attempt
            )
    raise RuntimeError("failed to generate a connected small-world topology")


class SmallWorldFFN(nn.Module):
    """Transformer FFN with sparse hidden-neuron message passing.

    Dense input/output projections remain compatible with solid-model weights.
    The binary graph is fixed during a candidate's gradient update, while edge
    strengths and all dense parameters are differentiable. NSGA-II changes the
    graph between candidates.
    """

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        topology: SmallWorldTopology,
        message_steps: int = 2,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        if topology.num_nodes != hidden_features:
            raise ValueError(
                f"topology has {topology.num_nodes} nodes, expected {hidden_features}"
            )
        if message_steps < 1:
            raise ValueError("message_steps must be positive")
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.topology = topology
        self.message_steps = message_steps
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.activation = nn.GELU()
        self.drop1 = nn.Dropout(drop)
        self.drop2 = nn.Dropout(drop)
        self.message_norms = nn.ModuleList(
            nn.LayerNorm(hidden_features) for _ in range(message_steps)
        )
        self.message_scales = nn.Parameter(torch.full((message_steps,), 0.1))

        directed_edges = []
        for source, target in topology.edges:
            directed_edges.extend(((source, target), (target, source)))
        edge_index = torch.tensor(directed_edges, dtype=torch.long).t().contiguous()
        self.register_buffer("edge_index", edge_index, persistent=True)
        initial_weight = 1.0 / math.sqrt(max(1, topology.degree))
        self.edge_weight = nn.Parameter(torch.full((edge_index.shape[1],), initial_weight))
        nn.init.normal_(self.edge_weight, mean=initial_weight, std=0.02)

    def _propagate(self, state: torch.Tensor) -> torch.Tensor:
        original_shape = state.shape
        flat = state.reshape(-1, self.hidden_features)
        # CUDA sparse kernels have broader support in FP32 than BF16/FP16.
        # Keep graph propagation stable while surrounding dense layers use AMP.
        with torch.autocast(device_type=flat.device.type, enabled=False):
            compute = flat.float()
            adjacency = torch.sparse_coo_tensor(
                self.edge_index,
                self.edge_weight.float(),
                (self.hidden_features, self.hidden_features),
                device=flat.device,
                dtype=torch.float32,
                check_invariants=False,
            ).coalesce()
            message = torch.sparse.mm(adjacency, compute.transpose(0, 1)).transpose(0, 1)
        return message.to(dtype=state.dtype).reshape(original_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state = self.drop1(self.activation(self.fc1(x)))
        for index, norm in enumerate(self.message_norms):
            message = self.activation(norm(self._propagate(state)))
            state = state + self.message_scales[index] * message
        return self.drop2(self.fc2(state))

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, hidden_features={self.hidden_features}, "
            f"edges={len(self.topology.edges)}, steps={self.message_steps}"
        )
