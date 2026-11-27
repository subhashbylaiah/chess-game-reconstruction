"""E19a: Maia-3 as a baseline and as a zero-training prior combined with our model.

Same setting as assisted_eval (gap length known, earlier moves confirmed). For each missing ply:
    ours      E16 model + feasibility filter (penalty 6)
    maia      Maia-3 log-probability (board, last 8 positions, both ratings); no scoresheet
    maia+F    Maia-3 + feasibility filter
    combo     ours + lambda * maia. lambda is tuned by 2-fold cross-fitting over games (tune on
              even-indexed games, report on odd, and vice versa), so no test game picks its own lambda.
"""
import argparse, collections, json, math, sys
import chess, numpy as np, torch
from recon.bench.build import read_jsonl
from recon.model import features as F
from recon.model.maia import load_maia, log_probs
from recon.model.train import load, pick_device
from recon.solvers.feasible import Entries, violations

ap = argparse.ArgumentParser()
ap.add_argument("--bench", default="bench/test.jsonl")
ap.add_argument("--ckpt", default="runs/c16_future/best.pt")
ap.add_argument("--maia", default="5m")
ap.add_argument("--presets", default="missing,mixed,mixed_ocr")
ap.add_argument("--penalty", type=float, default=6.0)
ap.add_argument("--device", default="auto")
ap.add_argument("--limit", type=int, default=None)
ap.add_argument("--out", default=None)
a = ap.parse_args()
dev = pick_device(a.device)
insts = [i for i in read_jsonl(a.bench) if i["preset"] in a.presets.split(",") and i["truth"]["hidden_plies"]][: a.limit]


def layout_of(inst):
    tp = inst["truth"]["sources"][0]["token_ply"]
    return {p: t for t, p in enumerate(tp) if p is not None}


rows, owner, hists, elos = [], [], [], []
for ii, inst in enumerate(insts):
    inp, truth = inst["input"], inst["truth"]
    tokens = inp["sources"][0]["tokens"]
    pos_of = layout_of(inst)
    hidden = set(truth["hidden_plies"])
    ctx = F.Context.from_input(inp)
    cache = F.TokenCache.build(tokens)
    pl = inp.get("players") or {}
    we, be = (pl.get("white") or {}).get("rating"), (pl.get("black") or {}).get("rating")
    board, last, start = chess.Board(inp["start_fen"]), None, 0
    history = [board.copy(stack=False)]
    for i, u in enumerate(truth["moves_uci"]):
        m = chess.Move.from_uci(u)
        if i in hidden:
            rows.append(F.encode(board, last, tokens, start, i, ctx, None, pos_of, cache))
            nxt = next((p for p in sorted(pos_of) if p > i), None)
            owner.append((ii, i, board.copy(stack=False), m, (nxt - i) if nxt is not None else 0))
            hists.append(history[-8:])
            elos.append((we, be) if board.turn == chess.WHITE else (be, we))
        if i in pos_of:
            start = pos_of[i] + 1
        board.push(m); last = m
        history.append(board.copy(stack=False))

# our model + filter
model = load(a.ckpt, a.device)
ours, viol, ents = [], [], {}
with torch.no_grad():
    for lo in range(0, len(rows), 256):
        xs = rows[lo:lo + 256]
        batch = {k: torch.from_numpy(np.stack([x[k] for x in xs])).to(dev) for k in xs[0]}
        out = model(batch)
        mv = torch.log_softmax(out["move"].float(), -1).cpu().numpy()
        pr = torch.log_softmax(out["promo"].float(), -1).cpu().numpy()
        for ci, (ii, ply, board, true, d) in enumerate(owner[lo:lo + 256]):
            if ii not in ents:
                ents[ii] = Entries(insts[ii]["input"]["sources"][0]["tokens"], layout_of(insts[ii]))
            sc, vi = {}, {}
            for m in board.legal_moves:
                sc[m] = float(mv[ci, F.encode_move(m, board.turn)] + (pr[ci, F.PROMOS[m.promotion]] if m.promotion else 0.0))
                vi[m] = violations(board, m, ply, ents[ii])
            ours.append(sc); viol.append(vi)
print("scored ours", file=sys.stderr, flush=True)

mmodel, mcfg = load_maia(a.maia, str(dev))
maia = log_probs(mmodel, mcfg, hists, elos, str(dev))
print("scored maia", file=sys.stderr, flush=True)


def rank(i, w_ours, w_maia, use_filter=True):
    true = owner[i][3]
    s = {m: w_ours * ours[i][m] + w_maia * maia[i][m] - (a.penalty * viol[i][m] if use_filter else 0) for m in ours[i]}
    st = s[true]
    return 1 + sum(v > st for m, v in s.items() if m != true)


def stats(rs):
    n = len(rs)
    return {"n": n, "top1": sum(r == 1 for r in rs) / n, "top2": sum(r <= 2 for r in rs) / n,
            "top3": sum(r <= 3 for r in rs) / n, "top5": sum(r <= 5 for r in rs) / n, "mean_rank": float(np.mean(rs))}


def bucket(d):
    return "end" if d == 0 else str(d) if d <= 3 else "4-8" if d <= 8 else "9+"


N = len(owner)
fold = [owner[i][0] % 2 for i in range(N)]
grid = [0.0, 0.2, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0]


def objective(idx, lam):  # top-2 then top-1 on a fold
    rs = [rank(i, 1.0, lam) for i in idx]
    return (sum(r <= 2 for r in rs), sum(r == 1 for r in rs))


combo_ranks, chosen = [0] * N, {}
for f in (0, 1):
    tune = [i for i in range(N) if fold[i] == f]
    lam = max(grid, key=lambda l: objective(tune, l))
    chosen[f"tuned_on_fold{f}"] = lam
    for i in range(N):
        if fold[i] != f:
            combo_ranks[i] = rank(i, 1.0, lam)

res = {
    "ours (E16 + filter)": [rank(i, 1.0, 0.0) for i in range(N)],
    "maia alone": [rank(i, 0.0, 1.0, use_filter=False) for i in range(N)],
    "maia + filter": [rank(i, 0.0, 1.0) for i in range(N)],
    "ours + lambda*maia (cross-fitted)": combo_ranks,
}
summary = {"lambda": chosen, "n": N}
print(f"missing plies {N}; lambda chosen per fold: {chosen}")
for name, rs in res.items():
    s = stats(rs)
    by = collections.defaultdict(list)
    for i, r in enumerate(rs):
        by[bucket(owner[i][4])].append(r)
    bd = {k: stats(by[k]) for k in ["1", "2", "3", "4-8", "9+", "end"] if k in by}
    summary[name] = {**s, "by_dist": bd}
    print(f"{name:36s} top1 {s['top1']:.1%}  top2 {s['top2']:.1%}  top3 {s['top3']:.1%}  top5 {s['top5']:.1%}  mean rank {s['mean_rank']:.2f}")
    print("    top1 by dist: " + "  ".join(f"{k} {100 * v['top1']:.1f}" for k, v in bd.items()))
print("full-grid sweep (all games, for reference only):")
for lam in grid:
    rs = [rank(i, 1.0, lam) for i in range(N)]
    s = stats(rs)
    print(f"  lambda {lam:4.1f}  top1 {s['top1']:.1%}  top2 {s['top2']:.1%}  top5 {s['top5']:.1%}")
if a.out:
    json.dump(summary, open(a.out, "w"), indent=1)
