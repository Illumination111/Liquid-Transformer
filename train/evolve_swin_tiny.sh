#!/usr/bin/env bash
set -euo pipefail

python train/evolve.py --model swin-tiny --devices 0,1 "$@"
