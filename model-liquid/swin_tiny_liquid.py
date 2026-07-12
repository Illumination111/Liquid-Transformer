"""Liquid Swin-Tiny with evolvable small-world feed-forward networks."""

from __future__ import annotations

import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
SOLID_DIR = THIS_DIR.parent / "model-solid"
for path in (THIS_DIR, SOLID_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from small_world import (  # noqa: E402
    SmallWorldFFN,
    SmallWorldTopology,
    watts_strogatz_topology,
)
from swin_tiny import SwinTiny  # noqa: E402


def swin_topology_nodes(
    embed_dim: int = 96, depths: tuple[int, ...] = (2, 2, 6, 2)
) -> list[int]:
    return [embed_dim * 2**stage * 4 for stage, depth in enumerate(depths) for _ in range(depth)]


class LiquidSwinTiny(SwinTiny):
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
        blocks = [block for layer in self.layers for block in layer.blocks]
        nodes = [block.norm2.normalized_shape[0] * 4 for block in blocks]
        if topologies is None:
            topologies = [
                watts_strogatz_topology(
                    count, graph_degree, rewire_probability, topology_seed + index
                )
                for index, count in enumerate(nodes)
            ]
        if len(topologies) != len(blocks):
            raise ValueError(f"expected {len(blocks)} topologies")
        drop = self.pos_drop.p
        for block, topology in zip(blocks, topologies, strict=True):
            block.mlp = SmallWorldFFN(
                block.norm2.normalized_shape[0],
                topology.num_nodes,
                topology,
                message_steps,
                drop,
            )
        self.topologies = topologies


def swin_tiny_liquid(
    num_classes: int = 100, **kwargs: object
) -> LiquidSwinTiny:
    return LiquidSwinTiny(num_classes=num_classes, **kwargs)
