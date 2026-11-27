#!/bin/zsh
# E12: v1 + known gap length as input (given for 70% of training games). Same data, steps, size, seed as runs/v1.
cd "$(dirname "$0")/.."
COMMON=(--steps 30000 --eval-every 2000 --log-every 500 --workers 6 --val-games 1000)
uv run recon train --out runs/v5_gap "${COMMON[@]}" --gap-input > runs/v5_gap.out 2>&1
