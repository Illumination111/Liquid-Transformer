"""Liquid DeiT-Tiny with evolvable small-world feed-forward networks."""

from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
SOLID_DIR = THIS_DIR.parent / "model-solid"
for path in (THIS_DIR, SOLID_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from deit_tiny import DeiTTiny  # noqa: E402
from small_world import (  # noqa: E402
    SmallWorldFFN,
    SmallWorldTopology,
    watts_strogatz_topology,
)


def deit_topology_nodes(depth: int = 12, embed_dim: int = 192) -> list[int]:
    return [embed_dim * 4] * depth


class LiquidDeiTTiny(DeiTTiny):
    def __init__(
        self,
        topologies: list[SmallWorldTopology] | None = None,
        graph_degree: int = 8,
        rewire_probability: float = 0.1,
        topology_seed: int = 42,
        message_steps: int = 2,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        nodes = deit_topology_nodes(len(self.blocks), self.head.in_features)
        if topologies is None:
            topologies = [
                watts_strogatz_topology(
                    count, graph_degree, rewire_probability, topology_seed + index
                )
                for index, count in enumerate(nodes)
            ]
        if len(topologies) != len(self.blocks):
            raise ValueError(f"expected {len(self.blocks)} topologies")
        drop = self.pos_drop.p
        for block, topology in zip(self.blocks, topologies, strict=True):
            block.mlp = SmallWorldFFN(
                block.norm2.normalized_shape[0],
                topology.num_nodes,
                topology,
                message_steps,
                drop,
            )
        self.topologies = topologies


def deit_tiny_liquid(
    num_classes: int = 100, **kwargs: object
) -> LiquidDeiTTiny:
    return LiquidDeiTTiny(num_classes=num_classes, **kwargs)
