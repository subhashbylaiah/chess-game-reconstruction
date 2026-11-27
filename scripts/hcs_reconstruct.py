"""Recover the most likely true game behind each HCS sheet (labels = what the player wrote, slips included).

Beam search over legal games; each written entry costs the edit distance between it and the SAN of the move
played (check/mate marks ignored; "0-0" = "O-O"); a blank box costs BLANK. A move that matches exactly costs 0.
Reports per sheet the edits needed, and saves data/hcs/recon.jsonl. Only substitutions are modelled (no
skipped/extra rows), so sheets with a row shift come out expensive and are flagged rather than trusted.
"""
import json, sys
import chess

BEAM, BLANK = int(sys.argv[1]) if len(sys.argv) > 1 else 64, 1.0


def norm(s):
    return s.replace("0", "O").replace("+", "").replace("#", "").replace("!", "").replace("?", "").strip()


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def reconstruct(written):
    beam = [(0.0, chess.Board(), [])]
    for w in written:
        w = norm(w)
        nxt = []
        for cost, b, moves in beam:
            for m in b.legal_moves:
                s = norm(b.san(m))
                c = BLANK if not w else (0.0 if s == w else float(lev(s, w)))
                nxt.append((cost + c, b, moves, m, s))
        nxt.sort(key=lambda t: t[0])
        beam, seen = [], set()
        for cost, b, moves, m, s in nxt:
            nb = b.copy(stack=False); nb.push(m)
            key = nb.fen()
            if key in seen:
                continue
            seen.add(key)
            beam.append((cost, nb, moves + [s]))
            if len(beam) >= BEAM:
                break
        if not beam:
            return None
    return beam[0][0], beam[0][2]


rows = [json.loads(l) for l in open("data/hcs/games.jsonl")]
out = []
for r in rows:
    res = reconstruct(r["written"])
    if res is None:
        continue
    cost, moves = res
    diffs = [(i, w, s) for i, (w, s) in enumerate(zip(r["written"], moves)) if norm(w) != s]
    out.append({**r, "truth": moves, "cost": cost, "n_diff": len(diffs), "diffs": diffs})
with open("data/hcs/recon.jsonl", "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
import statistics
nd = [r["n_diff"] for r in out]
plies = sum(r["n"] for r in out)
print(f"sheets {len(out)}  plies {plies}  differing entries {sum(nd)} ({sum(nd)/plies:.1%})  "
      f"median per sheet {statistics.median(nd)}  sheets with 0 diffs {sum(x==0 for x in nd)}")
print("cost-per-ply quantiles", [round(q, 3) for q in statistics.quantiles([r['cost']/r['n'] for r in out], n=10)])
