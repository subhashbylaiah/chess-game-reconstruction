#!/bin/zsh
# E08 / E09: structured-notation ablations. Same data, steps, size and val set as runs/v1.
cd "$(dirname "$0")/.."
COMMON=(--steps 30000 --eval-every 2000 --log-every 500 --workers 6 --val-games 1000)
uv run recon train --out runs/v2_struct "${COMMON[@]}" --struct-notes > runs/v2_struct.out 2>&1
uv run recon train --out runs/v3_struct_gap "${COMMON[@]}" --struct-notes --gap-input > runs/v3_struct_gap.out 2>&1
