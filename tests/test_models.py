from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model-solid"))

from deit_tiny import deit_tiny  # noqa: E402
from swin_tiny import swin_tiny  # noqa: E402


@pytest.mark.parametrize("factory", [deit_tiny, swin_tiny])
def test_cifar100_output_shape(factory):
    model = factory(num_classes=100).eval()
    images = torch.randn(2, 3, 32, 32)
    with torch.inference_mode():
        logits = model(images)
    assert logits.shape == (2, 100)


def test_tiny_parameter_scale():
    deit_parameters = sum(p.numel() for p in deit_tiny().parameters())
    swin_parameters = sum(p.numel() for p in swin_tiny().parameters())
    assert 5_000_000 < deit_parameters < 6_500_000
    assert 27_000_000 < swin_parameters < 30_000_000
