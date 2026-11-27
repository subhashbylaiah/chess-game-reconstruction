#!/bin/bash
# Wait for a cloud training run to finish, then run the assisted test-set evals on this GPU.
# Usage (in tmux): bash scripts/cloud/eval_after.sh c13_long E13c   (NOSEARCH=1 to skip the search eval)
cd "$(dirname "$0")/../.."; source ~/.local/bin/env 2>/dev/null
export UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:-$HOME/.venv-recon}
RUN=$1; ID=$2; C=runs/$RUN/best.pt; R=experiments/results/$ID
until grep -q "^done:" runs/$RUN.out 2>/dev/null; do sleep 60; done
mkdir -p $R
for p in 0 6; do
  uv run python scripts/assisted_eval.py --ckpt $C --penalty $p --out $R/model_pen$p.json --save $R/model_pen${p}_ranks.jsonl 2>/dev/null | head -1 > $R/model_pen$p.txt
done
if [ -n "$NOSEARCH" ]; then touch $R/DONE; exit 0; fi  # the search does not support the hybrid model yet
uv run python scripts/assisted_eval.py --ckpt $C --search --max-dist 3 --penalty 6 --first-moves 10 --width 8 \
  --out $R/search_d3.json --save $R/search_d3_ranks.jsonl 2>/dev/null > $R/search_d3.txt
uv run python scripts/topk_table.py $R/search_d3_ranks.jsonl > $R/search_d3_topk.txt
touch $R/DONE
