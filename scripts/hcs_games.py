"""Build per-game written move lists from HCS ground-truth labels and replay them with python-chess.

HCS labels are what the player WROTE in each box (so real slips remain). Training tags:
  [game]_[page]_[move]_[white|black].png SAN ; testing tags: [game]_[move]_[white|black] SAN.
Output: data/hcs/games.jsonl, one row per sheet (a game's two pages are the two players' copies): id, split, written (list of SAN or "" per ply, in order),
and how far the written record replays legally.
"""
import collections, json, re, sys
import chess

ROOT = sys.argv[1] if len(sys.argv) > 1 else "data/hcs/x/HCS Dataset December 2021/data"
games = collections.defaultdict(dict)  # (split, game, page) -> {(move, side): san}; each page is one player's copy
for split, fn in (("train", "training_tags.txt"), ("test", "testing_tags.txt")):
    for line in open(f"{ROOT}/{fn}"):
        parts = line.strip().split(maxsplit=1)
        if not parts:
            continue
        key, san = parts[0], (parts[1] if len(parts) > 1 else "")
        f = re.sub(r"\.png$", "", key).split("_")
        g, pg, mv, side = (f[0], f[1], f[2], f[3]) if len(f) == 4 else (f[0], "t", f[1], f[2])
        games[(split, g, pg)][(int(mv), side)] = san.strip()

out, stats = [], collections.Counter()
for (split, g, pg), boxes in sorted(games.items()):
    last = max(m for m, _ in boxes)
    written = []
    for m in range(1, last + 1):
        for side in ("white", "black"):
            if (m, side) in boxes:
                written.append(boxes[(m, side)])
            elif m < last or side == "white":
                written.append("")
    while written and not written[-1]:
        written.pop()
    b, legal_plies, first_bad = chess.Board(), 0, None
    for i, s in enumerate(written):
        try:
            b.push_san(s.replace("0", "O"))
            legal_plies += 1
        except Exception:
            first_bad = i
            break
    stats["games"] += 1
    stats["fully_legal"] += first_bad is None
    stats["plies"] += len(written)
    stats["blank_boxes"] += sum(1 for s in written if not s)
    out.append({"id": f"hcs/{split}/{g}_{pg}", "game": g, "page": pg, "split": split, "written": written, "n": len(written),
                "legal_prefix": legal_plies, "first_bad": first_bad,
                "bad_san": written[first_bad] if first_bad is not None else None})
with open("data/hcs/games.jsonl", "w") as f:
    for r in out:
        f.write(json.dumps(r) + "\n")
print(dict(stats))
print("split counts", collections.Counter(r["split"] for r in out))
print("legal-prefix share of plies", round(sum(r["legal_prefix"] for r in out) / stats["plies"], 3))
bad = [r for r in out if r["first_bad"] is not None]
print("first-bad examples:", [(r["id"], r["first_bad"], r["bad_san"]) for r in bad[:12]])
