#!/bin/zsh
# E11 sweep on the 20-game pilot: feasibility filter x search width / candidates / tree.
# Search runs at every distance (--max-dist 99) so the per-distance table shows where each helps.
cd "$(dirname $0)/.."
R=experiments/results/E11; mkdir -p $R
run() { name=$1; shift; echo "== $name $*"; uv run python scripts/assisted_eval.py --limit 20 --out $R/pilot_$name.json --save $R/pilot_${name}_ranks.jsonl "$@" 2>/dev/null; }
run model            --penalty 0
run model_filter     --penalty 6
run e06_repro        --search --max-dist 99 --penalty 0 --first-moves 16 --width 8
run c16_w8           --search --max-dist 99 --penalty 6 --first-moves 16 --width 8
run c10_w8           --search --max-dist 99 --penalty 6 --first-moves 10 --width 8
run c5_w8            --search --max-dist 99 --penalty 6 --first-moves 5  --width 8
run c10_w32          --search --max-dist 99 --penalty 6 --first-moves 10 --width 32
run c10_w128         --search --max-dist 99 --penalty 6 --first-moves 10 --width 128
run c10_tree2        --search --max-dist 99 --penalty 6 --first-moves 10 --width 512 --blank-moves 2
run c10_tree3        --search --max-dist 99 --penalty 6 --first-moves 10 --width 512 --blank-moves 3
echo done
