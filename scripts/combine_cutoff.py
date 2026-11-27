"""Exact ranks for a search cutoff, from saved per-ply ranks: search ranks where the next written move is
within `cutoff` plies, model(+filter) ranks beyond (that is exactly what AssistConfig.max_dist does)."""
import json, os, sys, collections
from recon.bench.build import read_jsonl
search, model, cutoff = sys.argv[1], sys.argv[2], int(sys.argv[3])
insts = {i["id"]: i for i in read_jsonl(os.environ.get("BENCH", "bench/test.jsonl"))}
load = lambda p: {(r["id"], r["ply"]): r["rank"] for r in map(json.loads, open(p))}
S, M = load(search), load(model)
def dist(i, ply):
    tp = insts[i]["truth"]["sources"][0]["token_ply"]; w = [p for p in tp if p is not None and p > ply]
    return min(w) - ply if w else 0
out = {k: (S[k] if 0 < dist(*k) <= cutoff else M[k]) for k in M}
rs = list(out.values()); n = len(rs)
games = collections.defaultdict(list)
for (i, _), r in out.items(): games[i].append(r)
G = len(games)
print(f"cutoff {cutoff}: top1 {sum(r==1 for r in rs)/n:.1%} top2 {sum(r<=2 for r in rs)/n:.1%} top3 {sum(r<=3 for r in rs)/n:.1%} "
      f"top5 {sum(r<=5 for r in rs)/n:.1%} top10 {sum(r<=10 for r in rs)/n:.1%} mean rank {sum(rs)/n:.2f} | per game tap {sum(r<=2 for r in rs)/G:.2f} "
      f"expand {sum(2<r<=5 for r in rs)/G:.2f} typed {sum(r>5 for r in rs)/G:.2f} no-typing {sum(all(r<=5 for r in v) for v in games.values())/G:.0%}")
if len(sys.argv) > 4:
    with open(sys.argv[4], "w") as f:
        for (i, p), r in out.items(): f.write(json.dumps({"id": i, "ply": p, "rank": int(r)}) + "\n")
