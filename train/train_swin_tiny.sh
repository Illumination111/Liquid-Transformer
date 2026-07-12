#!/usr/bin/env bash
set -euo pipefail

python train/train.py --model swin-tiny "$@"
