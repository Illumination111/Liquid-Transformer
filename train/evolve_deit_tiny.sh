#!/usr/bin/env bash
set -euo pipefail

python train/evolve.py --model deit-tiny --devices 0,1 "$@"
