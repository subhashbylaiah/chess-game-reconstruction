"""Simulated user-assisted reconstruction (primary setting: gap length known).

At every missing ply the player has confirmed all earlier moves, so the model sees the
true board. We record the rank of the true move in the suggestion list.

Effort tiers (the app shows 2 suggestions up front, 3 more on "expand", else the player types):
    tier1  rank 1-2   one tap
    tier2  rank 3-5   expand + tap
    typed  rank 6+    type the move
"""
import argparse, collections, json, math, sys, time
import chess, numpy as np, torch
from recon.bench.build import read_jsonl, gap_bucket
from recon.model import features as F
from recon.model.train import load, pick_device
from recon.solvers.feasible import Entries, violations
from recon.model.maia import elo_pair, hist_codes

ap = argparse.ArgumentParser()
ap.add_argument("--bench", default="bench/test.jsonl")
ap.add_argument("--ckpt", default="runs/v1/best.pt")
ap.add_argument("--presets", default="missing,mixed,mixed_ocr")
ap.add_argument("--show", type=int, default=5, help="suggestions shown after expanding")
ap.add_argument("--search", action="store_true", help="rank with gap-aware search instead of the model alone")
ap.add_argument("--limit", type=int, default=None, help="first N games with gaps")
ap.add_argument("--horizon", type=int, default=6)
ap.add_argument("--first-moves", type=int, default=16)
ap.add_argument("--width", type=int, default=8)
ap.add_argument("--blank-moves", type=int, default=6)
ap.add_argument("--written-extra", type=int, default=4)
ap.add_argument("--max-dist", type=int, default=8, help="search only this close to the next written move")
ap.add_argument("--penalty", type=float, default=0.0, help="feasibility filter penalty per violation (inf = remove)")
ap.add_argument("--save", default=None, help="write per-ply ranks (jsonl)")
ap.add_argument("--out", default=None, help="write the summary (json)")
ap.add_argument("--device", default="auto")
ap.add_argument("--gap-known", action="store_true", help="give the model plies-to-next-written (needs a gap_input model)")
a = ap.parse_args()

dev = pick_device(a.device)
insts = [i for i in read_jsonl(a.bench) if i["preset"] in a.presets.split(",") and i["truth"]["hidden_plies"]][: a.limit]


def layout_of(inst):
    tp = inst["truth"]["sources"][0]["token_ply"]
    return {p: t for t, p in enumerate(tp) if p is not None}


rows, owner = [], []  # encoded positions; (instance idx, ply, board, true move, dist to next written)
for ii, inst in enumerate(insts):
    inp, truth = inst["input"], inst["truth"]
    tokens = inp["sources"][0]["tokens"]
    pos_of = layout_of(inst)
    hidden = set(truth["hidden_plies"])
    ctx = F.Context.from_input(inp)
    cache = F.TokenCache.build(tokens)
    board, last, start = chess.Board(inp["start_fen"]), None, 0
    raws, turns = [], []  # history for the hybrid model's Maia tower
    for i, u in enumerate(truth["moves_uci"]):
        raws.append(F.raw_squares(board)); turns.append(board.turn)
        m = chess.Move.from_uci(u)
        if i in hidden:
            nxt = next((p for p in sorted(pos_of) if p > i), None)
            gap = (nxt - i) if nxt is not None else F.MAX_GAP
            if not a.search:
                x = F.encode(board, last, tokens, start, i, ctx, gap if a.gap_known else None, pos_of, cache)
                x["maia_hist"], x["maia_elo"] = hist_codes(raws, turns), elo_pair(ctx.white_elo, ctx.black_elo, board.turn)
                rows.append(x)
            owner.append((ii, i, board.copy(stack=False), m, nxt - i if nxt is not None else 0))
        if i in pos_of:
            start = pos_of[i] + 1
        board.push(m); last = m

ranks, filt_true = [], 0
t0 = time.time()
if a.search:
    from recon.solvers.assisted import AssistConfig, AssistedRanker
    cfg = AssistConfig(first_moves=a.first_moves, width=a.width, blank_moves=a.blank_moves,
                       written_extra=a.written_extra, horizon=a.horizon, max_dist=a.max_dist, penalty=a.penalty)
    ranker = AssistedRanker(a.ckpt, a.device, cfg=cfg)
    for n, (ii, ply, board, true, d) in enumerate(owner):
        inst = insts[ii]
        moves = inst["truth"]["moves_uci"]
        order = ranker.rank(inst["input"], moves[:ply], layout_of(inst), len(moves))
        ranks.append(order.index(true) + 1)
        if n % 50 == 0:
            print(f"  {n}/{len(owner)} plies  {time.time() - t0:.0f}s", file=sys.stderr, flush=True)
else:
    model = load(a.ckpt, a.device)
    ents = {}
    with torch.no_grad():
        for lo in range(0, len(rows), 256):
            xs = rows[lo:lo + 256]
            batch = {k: torch.from_numpy(np.stack([x[k] for x in xs])).to(dev) for k in xs[0]}
            out = model(batch)
            mv = torch.log_softmax(out["move"].float(), -1).cpu().numpy()
            pr = torch.log_softmax(out["promo"].float(), -1).cpu().numpy()
            for ci, (ii, ply, board, true, d) in enumerate(owner[lo:lo + 256]):
                turn = board.turn
                if a.penalty and ii not in ents:
                    ents[ii] = Entries(insts[ii]["input"]["sources"][0]["tokens"], layout_of(insts[ii]))

                def s(m):
                    v = mv[ci, F.encode_move(m, turn)]
                    v += pr[ci, F.PROMOS[m.promotion]] if m.promotion else 0.0
                    if a.penalty:
                        k = violations(board, m, ply, ents[ii])
                        v -= (a.penalty if a.penalty != math.inf else 1e6) * k
                    return v
                scores = {m: s(m) for m in board.legal_moves}
                st = scores[true]
                ranks.append(1 + int(sum(v > st for m, v in scores.items() if m != true)))
elapsed = time.time() - t0

if a.save:
    with open(a.save, "w") as f:
        for (ii, ply, *_), r in zip(owner, ranks):
            f.write(json.dumps({"id": insts[ii]["id"], "ply": ply, "rank": int(r)}) + "\n")

per_game = collections.defaultdict(list)
for (ii, *_), r in zip(owner, ranks):
    per_game[ii].append(r)


def stats(rs):
    n = len(rs)
    return {"n": n, "top1": sum(r == 1 for r in rs) / n, "top2": sum(r <= 2 for r in rs) / n,
            "top3": sum(r <= 3 for r in rs) / n, "top5": sum(r <= 5 for r in rs) / n,
            "top10": sum(r <= 10 for r in rs) / n, "mean_rank": float(np.mean(rs))}


def summarize(game_ids):
    rs = [r for g in game_ids for r in per_game[g]]
    if not rs:
        return None
    G = len(game_ids)
    res = {"games": G, **stats(rs)}
    res["tier1/game"] = sum(r <= 2 for r in rs) / G
    res["tier2/game"] = sum(2 < r <= a.show for r in rs) / G
    res["typed/game"] = sum(r > a.show for r in rs) / G
    res["missing/game"] = len(rs) / G
    res["no_typing"] = sum(all(r <= a.show for r in per_game[g]) for g in game_ids) / G
    return res


groups = {"ALL": list(per_game)}
for g in per_game:
    groups.setdefault("preset=" + insts[g]["preset"], []).append(g)
    groups.setdefault("gap=" + gap_bucket(insts[g]["meta"]["max_gap"]), []).append(g)
res = {k: summarize(v) for k, v in groups.items()}

by_d = collections.defaultdict(list)
for (ii, ply, board, true, d), r in zip(owner, ranks):
    key = "end" if d == 0 else str(d) if d <= 3 else "4-8" if d <= 8 else "9+"
    by_d[key].append(r)
res["by_dist"] = {k: stats(by_d[k]) for k in ["1", "2", "3", "4-8", "9+", "end"] if k in by_d}
res["config"] = vars(a)
res["sec_per_suggestion"] = elapsed / max(len(owner), 1)

A = res["ALL"]
print(f"ALL n={A['n']}  top1 {A['top1']:.1%}  top2 {A['top2']:.1%}  top3 {A['top3']:.1%}  top5 {A['top5']:.1%}  "
      f"top10 {A['top10']:.1%}  mean rank {A['mean_rank']:.2f}  | per game: tap {A['tier1/game']:.2f} "
      f"expand {A['tier2/game']:.2f} typed {A['typed/game']:.2f} of {A['missing/game']:.1f}  "
      f"no-typing {A['no_typing']:.0%}  | {res['sec_per_suggestion']:.2f}s/suggestion")
for k, v in res["by_dist"].items():
    print(f"  dist {k:>4} n={v['n']:4d}  top1 {v['top1']:.1%}  top2 {v['top2']:.1%}  top5 {v['top5']:.1%}  mean rank {v['mean_rank']:.2f}")
if a.out:
    json.dump(res, open(a.out, "w"), indent=1)
