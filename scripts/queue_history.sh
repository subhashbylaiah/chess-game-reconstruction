#!/bin/zsh
# E10: v1 + move number + last 8 moves. Same data, steps, size and val set as runs/v1.
cd "$(dirname "$0")/.."
COMMON=(--steps 30000 --eval-every 2000 --log-every 500 --workers 6 --val-games 1000)
uv run recon train --out runs/v4_hist "${COMMON[@]}" --move-number --history 8 > runs/v4_hist.out 2>&1
