"""Stateless multi-step LIF neurons with a fast-sigmoid surrogate gradient."""

from __future__ import annotations

import math

import torch
from torch import nn


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, voltage: torch.Tensor, slope: float) -> torch.Tensor:
        ctx.save_for_backward(voltage)
        ctx.slope = slope
        return (voltage >= 0).to(voltage.dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (voltage,) = ctx.saved_tensors
        return grad_output / (1 + ctx.slope * voltage.abs()).square(), None


class MultiStepLIF(nn.Module):
    """Integrate [time, ...] currents; membrane is local to each forward call.

    u[t] = beta * v[t-1] + current[t]
    s[t] = H(u[t] - threshold)
    v[t] = u[t] - stop_gradient(s[t]) * threshold

    Soft reset is detached, but membrane integration remains differentiable
    through time. No membrane is stored in the module or in checkpoints.
    """

    def __init__(self, beta: float = 0.5, threshold: float = 1.0, slope: float = 5.0):
        super().__init__()
        if not math.isfinite(beta) or not 0 <= beta < 1:
            raise ValueError("lif beta must be finite and in [0, 1)")
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("lif threshold must be finite and positive")
        if not math.isfinite(slope) or slope <= 0:
            raise ValueError("surrogate slope must be finite and positive")
        self.beta, self.threshold, self.slope = beta, threshold, slope

    def forward(self, currents: torch.Tensor) -> torch.Tensor:
        if currents.ndim < 2 or currents.shape[0] < 1:
            raise ValueError("LIF expects a nonempty [time, ...] tensor")
        # FP32 integration is stable even when surrounding projections use AMP.
        membrane = torch.zeros_like(currents[0], dtype=torch.float32)
        spikes = []
        for current in currents.unbind(0):
            voltage = self.beta * membrane + current.float()
            spike = SurrogateSpike.apply(voltage - self.threshold, self.slope)
            membrane = voltage - spike.detach() * self.threshold
            spikes.append(spike.to(currents.dtype))
        return torch.stack(spikes)
