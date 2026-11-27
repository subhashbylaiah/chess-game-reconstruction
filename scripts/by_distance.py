import json, sys, collections
from recon.bench.build import read_jsonl
insts = {i["id"]: i for i in read_jsonl("bench/test.jsonl")}
def load(p): return {(r["id"], r["ply"]): r["rank"] for r in map(json.loads, open(p))}
runs = {n: load(p) for n, p in zip(sys.argv[1::2], sys.argv[2::2])}
first = next(iter(runs.values()))
def dist(i, ply):
    tp = insts[i]["truth"]["sources"][0]["token_ply"]; w = {p for p in tp if p is not None}
    d = 1
    while ply + d not in w and ply + d < len(insts[i]["truth"]["moves_uci"]): d += 1
    return d
b = collections.defaultdict(lambda: collections.defaultdict(list))
for (i, ply) in first:
    d = dist(i, ply); key = str(d) if d <= 3 else ("4-8" if d <= 8 else "9+")
    for n, r in runs.items(): b[key][n].append(r[(i, ply)])
for key in ["1", "2", "3", "4-8", "9+"]:
    if key not in b: continue
    row = " | ".join(f"{n}: top1 {sum(x==1 for x in v)/len(v):.2f} top5 {sum(x<=5 for x in v)/len(v):.2f}" for n, v in b[key].items())
    print(f"dist {key:>4} n={len(next(iter(b[key].values()))):4d}  {row}")
