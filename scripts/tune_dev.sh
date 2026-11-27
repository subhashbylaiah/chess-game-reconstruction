#!/bin/zsh
# E21: re-tune the search settings on the dev set (bench/dev.jsonl, 200 unseen val games), for E19c.
# Model(+filter) once, then search within 8 plies for 5 / 10 / 16 candidates; the cutoff (1..8) and the
# candidate count are then chosen from the saved ranks with scripts/combine_cutoff.py (BENCH=bench/dev.jsonl).
cd "$(dirname $0)/.."
C=${CKPT:-runs/c19c_hybrid23m/best.pt}; R=experiments/results/E21; mkdir -p $R
ev() { name=$1; shift; uv run python scripts/assisted_eval.py --bench bench/dev.jsonl --ckpt $C --out $R/$name.json --save $R/${name}_ranks.jsonl "$@" 2>$R/$name.err | head -1 > $R/$name.txt; echo "$name: $(cat $R/$name.txt)"; }
ev model_pen0 --penalty 0
ev model_pen6 --penalty 6
for c in 10 5 16; do ev search_c${c}_d8 --search --max-dist 8 --penalty 6 --first-moves $c --width 8; done
touch $R/DONE
