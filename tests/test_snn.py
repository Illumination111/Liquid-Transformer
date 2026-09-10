from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model-liquid"))

from deit_tiny_snn import SpikingDeiTTiny  # noqa: E402
from lif import MultiStepLIF  # noqa: E402


def test_lif_integrates_leaks_resets_and_backpropagates_through_time():
    lif = MultiStepLIF(beta=0.5, threshold=1.0)
    currents = torch.full((4, 1), 0.6, requires_grad=True)
    spikes = lif(currents)
    assert spikes.flatten().tolist() == [0.0, 0.0, 1.0, 0.0]
    spikes[-1].sum().backward()
    assert currents.grad[0].abs().item() > 0  # credit passes across time steps
    assert torch.isfinite(currents.grad).all()
    assert torch.equal(lif(currents.detach()), spikes.detach())
    assert not lif(torch.zeros_like(currents)).any()  # no batch-to-batch membrane leakage


@pytest.mark.parametrize("kwargs", [{"beta": 1}, {"threshold": 0}, {"slope": float("nan")}])
def test_invalid_lif_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        MultiStepLIF(**kwargs)


@pytest.mark.parametrize("time_steps", [1, 4])
def test_hybrid_deit_gradients_attention_and_state_isolation(time_steps):
    torch.manual_seed(8)
    model = SpikingDeiTTiny(
        embed_dim=24,
        depth=2,
        num_heads=3,
        graph_degree=4,
        time_steps=time_steps,
        message_steps=1,
        drop_path_rate=0,
    ).eval()
    images = torch.randn(2, 3, 32, 32)
    output = model(images)
    assert output.shape == (2, 100)
    torch.nn.functional.cross_entropy(output, torch.tensor([3, 7])).backward()
    for parameter in (
        model.blocks[0].mlp.fc1.weight,
        model.blocks[0].mlp.edge_weight,
        model.blocks[0].attn.qkv.weight,
        model.patch_embed.proj.weight,
    ):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0
    from deit_tiny import Attention

    assert isinstance(model.blocks[0].attn, Attention)
    assert not any(isinstance(module, MultiStepLIF) for module in model.blocks[0].attn.modules())
    with torch.no_grad():
        model(torch.randn(1, 3, 32, 32))
        torch.testing.assert_close(model(images), output)


def test_time_steps_change_temporal_response_with_identical_weights():
    torch.manual_seed(9)
    model = SpikingDeiTTiny(
        embed_dim=24,
        depth=1,
        num_heads=3,
        graph_degree=4,
        time_steps=1,
        message_steps=1,
        drop_path_rate=0,
    ).eval()
    images = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        one_step = model(images)
        model.time_steps = 4
        four_steps = model(images)
    assert not torch.allclose(one_step, four_steps)


def test_default_deit_backbone_shape():
    model = SpikingDeiTTiny(time_steps=2).eval()
    assert len(model.blocks) == 12 and model.head.in_features == 192
    with torch.no_grad():
        logits = model(torch.randn(1, 3, 32, 32))
    assert logits.shape == (1, 100) and torch.isfinite(logits).all()
