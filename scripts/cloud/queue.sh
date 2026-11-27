#!/bin/bash
# Training queue on the cloud GPU. Safe to re-run after an interruption: each run resumes from its state.pt
# and a finished run is skipped. Start inside tmux so it survives SSH disconnects:
#   tmux new -d -s train 'bash scripts/cloud/queue.sh 12 c13 c16'   (12 data workers; runs in order)
# With two instances sharing the filesystem, give each instance different runs.
cd "$(dirname "$0")/../.."; source ~/.local/bin/env 2>/dev/null
export UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:-$HOME/.venv-recon}
mkdir -p runs
W=${1:-12}; shift
COMMON=(--eval-every 5000 --log-every 500 --workers $W --val-games 1000 --amp --compile --steps 100000)
for r in "${@:-c13 c16}"; do
  case $r in
    # E13-cuda: exact E13 setup on this hardware, the baseline for every later cloud run.
    c13) uv run recon train --out runs/c13_long "${COMMON[@]}" >> runs/c13_long.out 2>&1 ;;
    # E16: E13 + every later written entry as tokens (exact ply offsets) + auxiliary future-board prediction.
    c16) uv run recon train --out runs/c16_future "${COMMON[@]}" --future 48 --future-boards 2 >> runs/c16_future.out 2>&1 ;;
    # E16-noboard: ablation, E16 without the auxiliary future-board loss.
    c16nb) uv run recon train --out runs/c16_noboard "${COMMON[@]}" --future 48 --future-boards 0 >> runs/c16_noboard.out 2>&1 ;;
    # E16-f16: ablation, E16 with only the next 16 future entries (c13's window size).
    c16f16) uv run recon train --out runs/c16_f16 "${COMMON[@]}" --future 16 --future-boards 2 >> runs/c16_f16.out 2>&1 ;;
    # wait until the 2M-game fetch has finished (used before c17b)
    waitfetch) until grep -q "^wrote" runs/fetch_2m.out 2>/dev/null; do sleep 60; done ;;
    # E17a: E16 design, bigger model (d=384, 8 layers, 12 heads; about 15M params), same 500k games.
    c17a) uv run recon train --out runs/c17a_big "${COMMON[@]}" --future 48 --future-boards 2 --d-model 384 --layers 8 --heads 12 >> runs/c17a_big.out 2>&1 ;;
    # E17b: same bigger model on 2M August games (first 500k identical to games_500k, so val is unchanged).
    c17b) uv run recon train --out runs/c17b_big2m "${COMMON[@]}" --future 48 --future-boards 2 --d-model 384 --layers 8 --heads 12 --games data/games_2m.parquet >> runs/c17b_big2m.out 2>&1 ;;
    # E19b: hybrid. Our tower starts from E16 (runs/c16_future/best.pt), plus pretrained Maia-3 5M and gated
    # cross-attention from Maia's squares to our scoresheet tokens. Starts equal to E19a (E16 + 1.5 x Maia).
    # New parts lr 5e-4, pretrained parts 0.1x. 50k steps.
    c19b) uv run recon train --out runs/c19b_hybrid "${COMMON[@]}" --steps 50000 --future 48 --future-boards 2 \
            --maia 5m --init-from runs/c16_future/best.pt --pre-lr-mult 0.1 --lam-init 1.5 >> runs/c19b_hybrid.out 2>&1 ;;
    # E19c: E19b with the Maia-3 23M tower (46M params; 18M new fusion params). E19a: E16 + 23M sum 62.7 / 79.4.
    c19c) uv run recon train --out runs/c19c_hybrid23m "${COMMON[@]}" --steps 50000 --future 48 --future-boards 2 \
            --maia 23m --init-from runs/c16_future/best.pt --pre-lr-mult 0.1 --lam-init 1.5 >> runs/c19c_hybrid23m.out 2>&1 ;;
    c20) uv run recon train --out runs/c20_hist "${COMMON[@]}" --steps 50000 --future 48 --future-boards 2 --history 8 \
            --maia 23m --init-from runs/c16_future/best.pt --pre-lr-mult 0.1 --lam-init 1.5 >> runs/c20_hist.out 2>&1 ;;
    # E31: E19c fine-tuned on real OTB tournament games (Lichess broadcasts 2020-01..2026-08) + 400k Lichess games
    c31) uv run recon train --out runs/c31_otb "${COMMON[@]}" --games data/games_otb_mix.parquet --steps 20000 --lr 2e-4 \
            --future 48 --future-boards 2 --maia 23m --init-full runs/c19c_hybrid23m/best.pt --pre-lr-mult 0.1 --lam-init 1.5 \
            >> runs/c31_otb.out 2>&1 ;;
    # E32: ablation of E16 (c16): written entries as whole (hashed) tokens instead of characters
    c16w) uv run recon train --out runs/c16_whole "${COMMON[@]}" --future 48 --future-boards 2 --whole-notes >> runs/c16_whole.out 2>&1 ;;
    # wait for E19b's training to finish (lets c19c queue behind a running c19b)
    wait19b) until grep -q "^done:" runs/c19b_hybrid.out 2>/dev/null; do sleep 60; done ;;
    *) echo "unknown run $r" ;;
  esac
done
echo "queue done: $*"
