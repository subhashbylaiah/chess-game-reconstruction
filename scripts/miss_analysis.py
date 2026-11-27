"""Why do we miss? Classify every missing ply where the true move is not ranked first (E15-lite).

For each missing ply i (all earlier moves confirmed, as in assisted_eval) we take the model's
ranking (model + feasibility filter) and ask, for the top suggestion s != true move m:

    order      s is played later in the same gap by the same side (we got the order wrong)
    equivalent replacing m by s, the rest of the TRUE game still replays legally and every later
               written entry keeps the same SAN: the clean scoresheet cannot tell s from m, even if
               every other missing move were known
    other      the sheet (with the true continuation) rules s out: a model error, or s needs a
               different continuation of the other blanks (not checked)

We also compute, per missing ply, the set of ALL legal moves that are 'equivalent' to the truth in
that sense. Its size k is an optimistic identifiability measure (it assumes the other missing
moves are known): if k > 1, no method can be sure of the move from the sheet alone.

Written entries are compared as the true SAN (a clean, correctly written sheet), so OCR noise and
writer slips are ignored here.
"""
import argparse, collections, json, math, sys
import chess, numpy as np, torch
from recon.bench.build import read_jsonl
from recon.model import features as F
from recon.model.train import load, pick_device
from recon.solvers.feasible import Entries, violations
from recon.model.maia import elo_pair, hist_codes

ap = argparse.ArgumentParser()
ap.add_argument("--bench", default="bench/test.jsonl")
ap.add_argument("--ckpt", default="runs/c16_future/best.pt")
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


def equivalent(moves, i, alt, written, start_fen):
    """Does playing alt at ply i keep the true continuation legal with the same written SANs?"""
    b = chess.Board(start_fen)
    for m in moves[:i]:
        b.push(m)
    b.push(alt)
    for j in range(i + 1, len(moves)):
        m = moves[j]
        if not b.is_legal(m):
            return False
        if j in written and b.san(m) != written[j]:
            return False
        b.push(m)
    return True


rows, owner = [], []
for ii, inst in enumerate(insts):
    inp, truth = inst["input"], inst["truth"]
    tokens = inp["sources"][0]["tokens"]
    pos_of = layout_of(inst)
    hidden = set(truth["hidden_plies"])
    ctx = F.Context.from_input(inp)
    cache = F.TokenCache.build(tokens)
    board, last, start = chess.Board(inp["start_fen"]), None, 0
    raws, turns = [], []  # history for a hybrid model's Maia tower (ignored by others)
    for i, u in enumerate(truth["moves_uci"]):
        m = chess.Move.from_uci(u)
        raws.append(F.raw_squares(board)); turns.append(board.turn)
        if i in hidden:
            x = F.encode(board, last, tokens, start, i, ctx, None, pos_of, cache)
            x["maia_hist"], x["maia_elo"] = hist_codes(raws, turns), elo_pair(ctx.white_elo, ctx.black_elo, board.turn)
            rows.append(x)
            nxt = next((p for p in sorted(pos_of) if p > i), None)
            owner.append((ii, i, board.copy(stack=False), m, (nxt - i) if nxt is not None else 0))
        if i in pos_of:
            start = pos_of[i] + 1
        board.push(m); last = m

model = load(a.ckpt, a.device)
ents, ranked = {}, []
with torch.no_grad():
    for lo in range(0, len(rows), 256):
        xs = rows[lo:lo + 256]
        batch = {k: torch.from_numpy(np.stack([x[k] for x in xs])).to(dev) for k in xs[0]}
        out = model(batch)
        mv = torch.log_softmax(out["move"].float(), -1).cpu().numpy()
        pr = torch.log_softmax(out["promo"].float(), -1).cpu().numpy()
        for ci, (ii, ply, board, true, d) in enumerate(owner[lo:lo + 256]):
            turn = board.turn
            if ii not in ents:
                ents[ii] = Entries(insts[ii]["input"]["sources"][0]["tokens"], layout_of(insts[ii]))

            def s(m):
                v = mv[ci, F.encode_move(m, turn)] + (pr[ci, F.PROMOS[m.promotion]] if m.promotion else 0.0)
                return v - a.penalty * violations(board, m, ply, ents[ii])
            ranked.append(sorted(board.legal_moves, key=s, reverse=True))
        print(f"  scored {min(lo + 256, len(rows))}/{len(rows)}", file=sys.stderr, flush=True)

# per-game truth helpers
game_moves, game_written = {}, {}
for ii, inst in enumerate(insts):
    ms = [chess.Move.from_uci(u) for u in inst["truth"]["moves_uci"]]
    b, sans = chess.Board(inst["input"]["start_fen"]), []
    for m in ms:
        sans.append(b.san(m)); b.push(m)
    game_moves[ii] = ms
    game_written[ii] = {p: sans[p] for p in layout_of(inst)}

recs = []
for n, ((ii, ply, board, true, d), order) in enumerate(zip(owner, ranked)):
    ms, wr = game_moves[ii], game_written[ii]
    fen = insts[ii]["input"]["start_fen"]
    nxt = ply + d if d else len(ms)
    later_same_side = {ms[j] for j in range(ply + 2, nxt, 2)}
    eq = [m for m in board.legal_moves if m == true or equivalent(ms, ply, m, wr, fen)]
    rank = order.index(true) + 1

    def kind(s):
        if s in later_same_side:
            return "order"
        if s in eq:
            return "equivalent"
        return "other"
    recs.append({"id": insts[ii]["id"], "ply": ply, "dist": d, "rank": rank, "k_equiv": len(eq),
                 "top1": kind(order[0]) if rank > 1 else "hit",
                 "top2": None if rank <= 2 else [kind(order[0]), kind(order[1])]})
    if n % 500 == 0:
        print(f"  classified {n}/{len(owner)}", file=sys.stderr, flush=True)

N = len(recs)
pct = lambda x: f"{100 * x / N:5.1f}%"
print(f"\nmissing plies: {N}")
print("\n== Identifiability (optimistic: other missing moves known, clean sheet) ==")
k = collections.Counter(min(r["k_equiv"], 5) for r in recs)
for v in sorted(k):
    print(f"  {('k=' + str(v)) if v < 5 else 'k>=5'}: {pct(k[v])}")
print(f"  uniquely determined (k=1): {pct(k[1])}")
print(f"  best possible top-1 if we pick uniformly within the equivalent set: {100 * sum(1 / r['k_equiv'] for r in recs) / N:.1f}%")
print(f"  equivalent set fits in top-2 (k<=2): {pct(sum(r['k_equiv'] <= 2 for r in recs))}")

print("\n== Top-1 misses: what was our first suggestion? ==")
c = collections.Counter(r["top1"] for r in recs)
for key in ["hit", "order", "equivalent", "other"]:
    print(f"  {key:10s} {pct(c[key])}")

print("\n== Top-2 misses (true move ranked 3+): what were our 2 suggestions? ==")
m2 = [r for r in recs if r["top2"]]
c2 = collections.Counter("order" if "order" in r["top2"] else "equivalent" if "equivalent" in r["top2"] else "other" for r in m2)
print(f"  top-2 misses: {pct(len(m2))}")
for key in ["order", "equivalent", "other"]:
    print(f"    at least one {key:10s} {pct(c2[key])}")

print("\n== Top-1 by distance: hit / order / equivalent / other ==")
for name, f in [("1", lambda d: d == 1), ("2", lambda d: d == 2), ("3", lambda d: d == 3),
                ("4-8", lambda d: 4 <= d <= 8), ("9+", lambda d: d >= 9), ("end", lambda d: d == 0)]:
    rs = [r for r in recs if f(r["dist"])]
    if rs:
        cc = collections.Counter(r["top1"] for r in rs)
        u = sum(r["k_equiv"] == 1 for r in rs) / len(rs)
        print(f"  {name:>4} n={len(rs):4d}  " + "  ".join(f"{key} {100 * cc[key] / len(rs):4.1f}" for key in ["hit", "order", "equivalent", "other"])
              + f"   unique {100 * u:4.1f}")

if a.out:
    with open(a.out, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
