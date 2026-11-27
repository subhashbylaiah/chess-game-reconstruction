"""Top-k table from saved per-ply ranks, with chance level on the same plies."""
import json, sys, statistics, chess
from recon.bench.build import read_jsonl
ranks = [json.loads(l) for l in open(sys.argv[1])]
insts = {i["id"]: i for i in read_jsonl("bench/test.jsonl")}
legal = {}
for iid in {r["id"] for r in ranks}:
    inst = insts[iid]; b = chess.Board(inst["input"]["start_fen"])
    for i, u in enumerate(inst["truth"]["moves_uci"]):
        legal[(iid, i)] = b.legal_moves.count(); b.push_uci(u)
rs = [r["rank"] for r in ranks]; ns = [legal[(r["id"], r["ply"])] for r in ranks]
print(f"n={len(rs)}  mean rank {statistics.mean(rs):.2f} (chance {statistics.mean((n + 1) / 2 for n in ns):.1f})  median rank {statistics.median(rs)}")
for k in (1, 2, 3, 4, 5, 8, 10):
    print(f"top-{k:<2} model {sum(r <= k for r in rs) / len(rs):6.1%}   chance {statistics.mean(min(k, n) / n for n in ns):6.1%}")
