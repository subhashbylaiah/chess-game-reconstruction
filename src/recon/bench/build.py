"""Build reproducible benchmark instances from stored complete games.

Each instance is one JSON object::

    {"id", "game_id", "split", "preset", "meta": {...},
     "input": {...what a solver may see...},
     "truth": {...ground truth, never passed to a solver...}}

The same games are used for every preset so presets can be compared pairwise.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable, Iterator
from pathlib import Path

from recon.game import Game, rating_band
from recon.observe.corrupt import PRESETS, CorruptionConfig, corrupt


def _h(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big")


def longest_run(plies: list[int]) -> int:
    best = cur = 0
    prev = None
    for p in plies:
        cur = cur + 1 if prev is not None and p == prev + 1 else 1
        best = max(best, cur)
        prev = p
    return best


def gap_bucket(n: int) -> str:
    if n == 0:
        return "0"
    if n < 5:
        return "1-4"
    if n < 10:
        return "5-9"
    if n < 20:
        return "10-19"
    return "20+"


def make_instance(game: Game, cfg: CorruptionConfig, seed: int, split: str) -> dict:
    rng = random.Random(_h(seed, game.id, cfg.name))
    sources, truth = corrupt(game, cfg, rng)
    hidden = sorted(set.intersection(*(set(s["omitted_plies"]) for s in truth["sources"])))
    truth["hidden_plies"] = hidden  # plies with no token in any source
    n_err = sum(k != "match" for s in truth["sources"] for k in s["token_kind"])
    return {
        "id": f"{cfg.name}/{game.id}",
        "game_id": game.id,
        "split": split,
        "preset": cfg.name,
        "meta": {
            "num_plies": len(truth["moves_uci"]),
            "max_gap": longest_run(truth["sources"][0]["omitted_plies"]),
            "num_hidden": len(hidden),
            "num_token_errors": n_err,
            "rating_band": rating_band(game.mean_elo),
            "speed": game.time_control.speed,
        },
        "input": {
            "variant": "standard",
            "start_fen": game.start_fen,
            "players": {
                "white": {"rating": game.white_elo, "rating_system": "lichess"},
                "black": {"rating": game.black_elo, "rating_system": "lichess"},
            },
            "time_control": game.time_control.to_dict(),
            "sources": sources,
            "clock_observations": [],
        },
        "truth": truth,
    }


def build_instances(
    games: Iterable[Game],
    presets: list[str],
    num_games: int,
    seed: int = 0,
    split: str = "test",
) -> Iterator[dict]:
    pool = sorted(games, key=lambda g: _h(seed, g.id))[:num_games]
    for name in presets:
        cfg = PRESETS[name]
        for game in pool:
            yield make_instance(game, cfg, seed, split)


def write_jsonl(rows: Iterable[dict], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, separators=(",", ":")) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)
