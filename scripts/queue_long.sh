#!/bin/zsh
# E13: v1 trained 100k steps instead of 30k (same model, data, seed, val). Starts after the E11 full eval.
cd "$(dirname "$0")/.."
while pgrep -f "scripts/assisted_eval.py" > /dev/null; do sleep 30; done
COMMON=(--steps 100000 --eval-every 5000 --log-every 500 --workers 6 --val-games 1000)
uv run recon train --out runs/v6_long "${COMMON[@]}" > runs/v6_long.out 2>&1
