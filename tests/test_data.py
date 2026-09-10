from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset import cifar100  # noqa: E402


class FakeCIFAR:
    def __init__(self, root, train, transform, download):
        assert train, "search must not instantiate the official test partition"

    def __len__(self):
        return 40

    def __getitem__(self, index):
        return torch.tensor(index), index


def test_search_split_is_disjoint_reproducible_and_never_opens_test(tmp_path, monkeypatch):
    monkeypatch.setattr(cifar100.datasets, "CIFAR100", FakeCIFAR)
    left, validation = cifar100.build_cifar100_search_loaders(
        tmp_path,
        batch_size=4,
        num_workers=0,
        validation_size=8,
        train_fraction=0.5,
    )
    right, second_validation = cifar100.build_cifar100_search_loaders(
        tmp_path,
        batch_size=4,
        num_workers=0,
        validation_size=8,
        train_fraction=0.5,
    )
    assert len(left.dataset) == 16 and len(validation.dataset) == 8
    assert not set(left.dataset.indices) & set(validation.dataset.indices)
    assert left.dataset.indices == right.dataset.indices
    assert validation.dataset.indices == second_validation.dataset.indices
    assert torch.equal(next(iter(left))[0], next(iter(right))[0])
    with pytest.raises(ValueError, match="smaller than batch-size"):
        cifar100.build_cifar100_search_loaders(
            tmp_path,
            batch_size=64,
            num_workers=0,
            validation_size=8,
        )
