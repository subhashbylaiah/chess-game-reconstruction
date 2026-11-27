"""Chance level for the assisted eval: rank the true move uniformly among the legal moves."""
import statistics, chess
from recon.bench.build import read_jsonl
ns = []
for inst in read_jsonl("bench/test.jsonl"):
    if inst["preset"] not in ("missing", "mixed", "mixed_ocr") or not inst["truth"]["hidden_plies"]:
        continue
    hidden = set(inst["truth"]["hidden_plies"])
    b = chess.Board(inst["input"]["start_fen"])
    for i, u in enumerate(inst["truth"]["moves_uci"]):
        if i in hidden:
            ns.append(b.legal_moves.count())
        b.push_uci(u)
q = sorted(ns)
print(f"missing plies {len(ns)}; legal moves: median {statistics.median(ns)}, mean {statistics.mean(ns):.1f}, p10 {q[len(q)//10]}, p90 {q[9*len(q)//10]}")
for k in (1, 3, 5):
    print(f"chance top-{k}: {statistics.mean(min(k, n) / n for n in ns):.1%}")
