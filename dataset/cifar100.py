"""CIFAR-100 input pipeline and augmentations."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)
DEFAULT_DATA_DIR = Path(os.environ.get("LIQUID_DATA_DIR", str(Path.home() / "dataset")))


def build_cifar100_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
            transforms.RandomErasing(p=0.25),
        ]
    )
    validation_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]
    )
    return train_transform, validation_transform


def _loader_options(batch_size: int, num_workers: int) -> dict[str, object]:
    return {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": num_workers > 0,
    }


def build_cifar100_loaders(
    data_dir: str | Path,
    batch_size: int = 128,
    num_workers: int = 4,
    download: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Build train and test loaders for a final solid-model training run."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    train_transform, validation_transform = build_cifar100_transforms()

    train_dataset = datasets.CIFAR100(
        root=data_dir, train=True, transform=train_transform, download=download
    )
    validation_dataset = datasets.CIFAR100(
        root=data_dir, train=False, transform=validation_transform, download=download
    )
    common = _loader_options(batch_size, num_workers)
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=True, **common)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **common)
    return train_loader, validation_loader


def build_cifar100_search_loaders(
    data_dir: str | Path,
    batch_size: int = 128,
    num_workers: int = 4,
    validation_size: int = 5000,
    train_fraction: float = 1.0,
    seed: int = 42,
    download: bool = False,
) -> tuple[DataLoader, DataLoader]:
    """Build search train/validation loaders without opening the official test set."""
    if not 0.0 < train_fraction <= 1.0:
        raise ValueError("train_fraction must be in (0, 1]")
    if not 1 <= validation_size < 50_000:
        raise ValueError("validation_size must be between 1 and 49,999")
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    train_transform, validation_transform = build_cifar100_transforms()
    augmented = datasets.CIFAR100(
        root=data_dir, train=True, transform=train_transform, download=download
    )
    clean = datasets.CIFAR100(
        root=data_dir, train=True, transform=validation_transform, download=download
    )
    if validation_size >= len(augmented):
        raise ValueError("validation_size must be smaller than the training dataset")
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(augmented), generator=generator).tolist()
    validation_indices = permutation[:validation_size]
    available_train = permutation[validation_size:]
    train_count = max(1, round(len(available_train) * train_fraction))
    train_indices = available_train[:train_count]
    if train_count < batch_size:
        raise ValueError("training subset is smaller than batch-size with drop_last=True")
    options = _loader_options(batch_size, num_workers)
    train_loader = DataLoader(
        Subset(augmented, train_indices),
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed),
        **options,
    )
    validation_loader = DataLoader(Subset(clean, validation_indices), shuffle=False, **options)
    return train_loader, validation_loader


def build_cifar100_test_loader(data_dir, batch_size=128, num_workers=4, download=False):
    """Load the official test partition only for an explicitly requested final evaluation."""
    _, transform = build_cifar100_transforms()
    dataset = datasets.CIFAR100(root=data_dir, train=False, transform=transform, download=download)
    return DataLoader(dataset, shuffle=False, **_loader_options(batch_size, num_workers))


def build_cifar100_evolution_loaders(
    data_dir,
    batch_size=128,
    num_workers=4,
    validation_size=5000,
    train_fraction=1.0,
    seed=42,
    download=False,
):
    """Compatibility wrapper for the legacy NSGA-II entry point."""
    train, validation = build_cifar100_search_loaders(
        data_dir, batch_size, num_workers, validation_size, train_fraction, seed, download
    )
    test = build_cifar100_test_loader(data_dir, batch_size, num_workers, download)
    return train, validation, test
