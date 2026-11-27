"""E30: simulated benchmark on real over-the-board games (Lichess broadcast database, CC BY-SA 4.0).

Parses a broadcast month, keeps standard-start games with >= 30 plies, and builds the primary presets
(missing, mixed, mixed_ocr) for 150 games each of classical / rapid / blitz, with the same simulator and seed as
bench/test.jsonl. Ratings are the broadcast's (usually FIDE) when present, else missing.
"""
import collections, io, sys
import chess.pgn, zstandard
from recon.bench.build import build_instances, write_jsonl
from recon.game import Game, TimeControl

src, out = sys.argv[1], sys.argv[2]
per = int(sys.argv[3]) if len(sys.argv) > 3 else 150


def elo(t):
    try:
        return int(t) if t and t.strip("?") else None
    except ValueError:
        return None


def tc(t):
    """Broadcast time controls are free text, mostly minutes ("90+30" = 90 min + 30 s); bases <= 180 are minutes."""
    import re
    nums = [int(x) for x in re.findall(r"\d+", t or "")]
    if not nums:
        return TimeControl(None, None)
    base = nums[0] * 60 if nums[0] <= 180 else nums[0]
    m = re.search(r"(\d+)\s*(?:sec|s\b|second|''|\+\s*\d)", t or "")
    inc = nums[1] if len(nums) > 1 and "+" in t and re.match(r"^\s*\d+\s*\+\s*\d+\s*$", t) else 0
    if not inc:
        m = re.search(r"(\d+)\s*(?:sec|second)", t or "")
        inc = int(m.group(1)) if m else 0
    return TimeControl(base, inc)


games, seen = collections.defaultdict(list), set()
with open(src, "rb") as fh:
    text = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8", errors="replace")
    while (g := chess.pgn.read_game(text)) is not None:
        h = g.headers
        if h.get("Variant", "Standard") != "Standard" or "FEN" in h or g.errors:
            continue
        moves = [n.move.uci() for n in g.mainline()]
        if len(moves) < 30 or tuple(moves) in seen:  # broadcasts sometimes repeat a game
            continue
        seen.add(tuple(moves))
        t = tc(h.get("TimeControl"))
        gid = h.get("GameURL", "").rsplit("/", 1)[-1] or f"g{len(seen)}"
        game = Game(id=f"bc{gid}", moves=moves, white_elo=elo(h.get("WhiteElo")), black_elo=elo(h.get("BlackElo")),
                    time_control=t, result=h.get("Result", "*"))
        games[t.speed].append(game)
print({k: len(v) for k, v in games.items()})
rows = []
for speed in ("classical", "rapid", "blitz"):
    for x in build_instances(games[speed], ["missing", "mixed", "mixed_ocr"], per, seed=0, split=f"bc_{speed}"):
        rows.append(x)
print("wrote", write_jsonl(rows, out))
rated = [g for v in games.values() for g in v if g.white_elo and g.black_elo]
print("rated share", round(len(rated) / sum(map(len, games.values())), 2),
      "median mean-elo", sorted((g.white_elo + g.black_elo) / 2 for g in rated)[len(rated) // 2] if rated else None)
