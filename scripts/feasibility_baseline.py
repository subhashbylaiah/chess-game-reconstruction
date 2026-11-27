"""Legality-only baseline (no learned model): at each missing ply, the moves that violate no later written
entry (scripts' feasibility filter) are ranked first in random order, the rest after them in random order.
Reports the EXPECTED top-k (exact, no sampling). Same plies as assisted_eval."""
import argparse, json
import chess
from recon.bench.build import read_jsonl
from recon.solvers.feasible import Entries, violations

ap = argparse.ArgumentParser()
ap.add_argument("--bench", default="bench/test.jsonl")
ap.add_argument("--presets", default="missing,mixed,mixed_ocr")
a = ap.parse_args()
insts = [i for i in read_jsonl(a.bench) if i["preset"] in a.presets.split(",") and i["truth"]["hidden_plies"]]
K = (1, 2, 5)
tot = {k: 0.0 for k in K}; chance = {k: 0.0 for k in K}; n = 0; feas_sizes = []; true_in = 0
for inst in insts:
    tp = inst["truth"]["sources"][0]["token_ply"]
    layout = {p: t for t, p in enumerate(tp) if p is not None}
    ents = Entries(inst["input"]["sources"][0]["tokens"], layout)
    hidden = set(inst["truth"]["hidden_plies"])
    b = chess.Board(inst["input"]["start_fen"])
    for i, u in enumerate(inst["truth"]["moves_uci"]):
        m = chess.Move.from_uci(u)
        if i in hidden:
            legal = list(b.legal_moves); L = len(legal)
            F = [x for x in legal if violations(b, x, i, ents) == 0]; f = len(F)
            feas_sizes.append(f); n += 1
            if m in F:
                true_in += 1
                for k in K: tot[k] += min(k, f) / f
            else:
                r = L - f
                for k in K: tot[k] += max(0, min(k - f, r)) / r
            for k in K: chance[k] += min(k, L) / L
        b.push(m)
feas_sizes.sort()
print(f"plies {n}  true move feasible {true_in/n:.1%}  median feasible set {feas_sizes[n//2]}")
print("chance          " + "  ".join(f"top{k} {100*chance[k]/n:.1f}" for k in K))
print("legality+filter " + "  ".join(f"top{k} {100*tot[k]/n:.1f}" for k in K))
