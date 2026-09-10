"""Download and verify CIFAR-100 without starting training."""

import argparse
from pathlib import Path

from torchvision.datasets import CIFAR100

from dataset.cifar100 import DEFAULT_DATA_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()
    for training in (True, False):
        data = CIFAR100(root=args.data_dir, train=training, download=True)
        print(f"{'train' if training else 'test'}: {len(data)} images; {args.data_dir}")


if __name__ == "__main__":
    main()
