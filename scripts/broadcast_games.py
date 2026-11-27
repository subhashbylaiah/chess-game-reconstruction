"""Training games from the Lichess broadcast database (real OTB tournament games; CC BY-SA 4.0).

Reads data/broadcast/lichess_db_broadcast_YYYY-MM.pgn.zst for months up to --until, keeps standard-start games
with >= 30 plies, deduplicates by move sequence (broadcasts repeat games), drops every game whose moves also occur
in the --exclude months (the E30 evaluation month), and writes a games parquet (store schema, id-hash splits).
Optionally appends N Lichess training games so the fine-tune keeps online play (--mix path:N).
"""
import argparse, collections, glob, io, re
import chess.pgn, zstandard
from recon.data.store import read_games, write_games
from recon.game import Game, TimeControl

ap = argparse.ArgumentParser()
ap.add_argument("--until", default="2026-08")
ap.add_argument("--exclude", default="2026-09")
ap.add_argument("--out", default="data/games_otb.parquet")
ap.add_argument("--mix", default="")
a = ap.parse_args()


def elo(t):
    try:
        return int(t) if t and t.strip("?") else None
    except ValueError:
        return None


def tc(t):
    nums = [int(x) for x in re.findall(r"\d+", t or "")]
    if not nums:
        return TimeControl(None, None)
    base = nums[0] * 60 if nums[0] <= 180 else nums[0]
    if re.match(r"^\s*\d+\s*\+\s*\d+\s*$", t or ""):
        inc = nums[1]
    else:
        m = re.search(r"(\d+)\s*(?:sec|second)", t or "")
        inc = int(m.group(1)) if m else 0
    return TimeControl(base, inc)


def games_in(path):
    with open(path, "rb") as fh:
        text = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8", errors="replace")
        while True:
            try:
                g = chess.pgn.read_game(text)
            except Exception:
                continue
            if g is None:
                return
            h = g.headers
            if h.get("Variant", "Standard") != "Standard" or "FEN" in h or g.errors:
                continue
            moves = [n.move.uci() for n in g.mainline()]
            if len(moves) >= 30:
                yield h, moves


import logging
logging.getLogger("chess.pgn").setLevel(logging.CRITICAL)
excl = set()
for m in a.exclude.split(","):
    for _, mv in games_in(f"data/broadcast/lichess_db_broadcast_{m}.pgn.zst"):
        excl.add(hash(tuple(mv)))
seen, out, speeds = set(), [], collections.Counter()
for path in sorted(glob.glob("data/broadcast/lichess_db_broadcast_*.pgn.zst")):
    month = path[-15:-8]
    if month > a.until:
        continue
    for h, mv in games_in(path):
        k = hash(tuple(mv))
        if k in seen or k in excl:
            continue
        seen.add(k)
        t = tc(h.get("TimeControl"))
        gid = h.get("GameURL", "").rsplit("/", 1)[-1] or f"{month}-{len(out)}"
        out.append(Game(id=f"bc{gid}", moves=mv, white_elo=elo(h.get("WhiteElo")), black_elo=elo(h.get("BlackElo")),
                        time_control=t, result=h.get("Result", "*"), date=h.get("UTCDate", h.get("Date", ""))))
        speeds[t.speed] += 1
    print(month, len(out), flush=True)
print("broadcast games", len(out), dict(speeds), "excluded-month duplicates dropped", len(excl))
if a.mix:
    p, n = a.mix.split(":")
    extra = [g for _, g in zip(range(int(n)), read_games(p, split="train"))]
    out += extra
    print("+ lichess", len(extra))
print("wrote", write_games(out, a.out))
