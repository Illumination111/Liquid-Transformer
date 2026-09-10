"""Hybrid DeiT: analog attention and multi-step LIF small-world FFNs."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

THIS_DIR = Path(__file__).resolve().parent
for directory in (THIS_DIR, THIS_DIR.parent / "model-solid"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from deit_tiny import DeiTTiny  # noqa: E402
from lif import MultiStepLIF  # noqa: E402
from small_world import SmallWorldFFN, SmallWorldTopology, watts_strogatz_topology  # noqa: E402


class SpikingSmallWorldFFN(SmallWorldFFN):
    """LIF replaces hidden GELU; dense output and residual remain analog."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        topology: SmallWorldTopology,
        message_steps: int = 2,
        drop: float = 0.0,
        beta: float = 0.5,
        threshold: float = 1.0,
        surrogate_slope: float = 5.0,
    ) -> None:
        super().__init__(in_features, hidden_features, topology, message_steps, drop)
        self.input_norm = nn.LayerNorm(hidden_features)
        self.input_lif = MultiStepLIF(beta, threshold, surrogate_slope)
        self.message_lifs = nn.ModuleList(
            MultiStepLIF(beta, threshold, surrogate_slope) for _ in range(message_steps)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("spiking FFN expects [time, batch, tokens, channels]")
        state = self.drop1(self.input_lif(self.input_norm(self.fc1(x))))
        for index, (norm, lif) in enumerate(zip(self.message_norms, self.message_lifs)):
            message = lif(norm(self._propagate(state)))
            state = state + self.message_scales[index] * message
        return self.drop2(self.fc2(state))


class SpikingDeiTTiny(DeiTTiny):
    """Repeat static patch currents, apply analog attention at each time step,
    integrate LIF neurons inside FFNs, and average temporal class features.

    Attention/patch embedding/readout are inherited from DeiT. This is a
    hybrid ANN/SNN, not a fully spike-driven Transformer or an ANN conversion.
    """

    def __init__(
        self,
        topologies: list[SmallWorldTopology] | None = None,
        time_steps: int = 4,
        lif_beta: float = 0.5,
        lif_threshold: float = 1.0,
        surrogate_slope: float = 5.0,
        graph_degree: int = 8,
        rewire_probability: float = 0.1,
        topology_seed: int = 42,
        message_steps: int = 2,
        **kwargs: object,
    ) -> None:
        if not isinstance(time_steps, int) or time_steps < 1:
            raise ValueError("time_steps must be a positive integer")
        super().__init__(**kwargs)
        self.time_steps = time_steps
        hidden_sizes = [block.mlp.fc1.out_features for block in self.blocks]
        if topologies is None:
            topologies = [
                watts_strogatz_topology(size, graph_degree, rewire_probability, topology_seed + i)
                for i, size in enumerate(hidden_sizes)
            ]
        if len(topologies) != len(self.blocks):
            raise ValueError("one topology is required per Transformer block")
        for block, size, topology in zip(self.blocks, hidden_sizes, topologies, strict=True):
            block.mlp = SpikingSmallWorldFFN(
                block.norm2.normalized_shape[0],
                size,
                topology,
                message_steps,
                self.pos_drop.p,
                lif_beta,
                lif_threshold,
                surrogate_slope,
            )
            block.mlp.apply(self._init_weights)
        self.topologies = list(topologies)

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embed(images)
        cls = self.cls_token.expand(patches.shape[0], -1, -1)
        tokens = torch.cat((cls, patches), dim=1) + self.pos_embed
        x = self.pos_drop(tokens.unsqueeze(0).expand(self.time_steps, -1, -1, -1))
        for block in self.blocks:
            # Flatten time and batch only for the unchanged analog attention.
            attention = block.attn(block.norm1(x).flatten(0, 1))
            x = x + block.drop_path1(attention).reshape_as(x)
            spikes = block.mlp(block.norm2(x))
            x = x + block.drop_path2(spikes.flatten(0, 1)).reshape_as(x)
        return self.norm(x)[:, :, 0].mean(dim=0)


def deit_tiny_snn(num_classes: int = 100, **kwargs: object) -> SpikingDeiTTiny:
    return SpikingDeiTTiny(num_classes=num_classes, **kwargs)
