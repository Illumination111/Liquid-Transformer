"""Batch-isolated multi-step LIF neurons backed by SpikingJelly."""

from __future__ import annotations

import math

import torch
from spikingjelly.activation_based import neuron, surrogate
from torch import nn


class MultiStepLIF(nn.Module):
    """Integrate [time, ...] currents with SpikingJelly LIFNode and ATan.

    u[t] = beta * v[t-1] + current[t]
    s[t] = H(u[t] - threshold)
    v[t] = u[t] - stop_gradient(s[t]) * threshold

    tau = 1 / (1 - beta), decay_input=False, v_reset=None reproduce this
    charging/soft-reset equation. SpikingJelly requires tau > 1, hence beta > 0.
    Its built-in ATan surrogate uses slope as alpha. State is reset before and
    after each call; the returned autograd graph still spans all time steps.
    Use train() for surrogate-gradient training, eval() for inference.
    """

    def __init__(self, beta: float = 0.5, threshold: float = 1.0, slope: float = 5.0):
        super().__init__()
        if not math.isfinite(beta) or not 0 < beta < 1:
            raise ValueError("lif beta must be finite and in (0, 1)")
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("lif threshold must be finite and positive")
        if not math.isfinite(slope) or slope <= 0:
            raise ValueError("surrogate slope must be finite and positive")
        self.beta, self.threshold, self.slope = beta, threshold, slope
        tau = 1.0 / (1.0 - beta)
        if tau <= 1.0:
            raise ValueError("lif beta is too small to represent tau > 1")
        self.node = neuron.LIFNode(
            tau=tau,
            decay_input=False,
            v_threshold=float(threshold),
            v_reset=None,
            surrogate_function=surrogate.ATan(alpha=float(slope)),
            detach_reset=True,
            step_mode="m",
            backend="torch",
            store_v_seq=False,
        )

    def forward(self, currents: torch.Tensor) -> torch.Tensor:
        if currents.ndim < 2 or currents.shape[0] < 1:
            raise ValueError("LIF expects a nonempty [time, ...] tensor")
        self.node.reset()
        try:
            # Keep integration FP32 even when surrounding projections use AMP.
            with torch.autocast(device_type=currents.device.type, enabled=False):
                return self.node(currents.float()).to(currents.dtype)
        finally:
            # Release membrane/graph references, including after a failed forward.
            self.node.reset()
