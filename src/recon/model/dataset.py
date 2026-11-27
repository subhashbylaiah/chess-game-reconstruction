"""Training positions generated on the fly from complete games.

Each pass re-corrupts every game with a fresh random scoresheet, so the model never
sees the same noisy notation twice. For each corrupted game we emit:
  * every ply that has no token (the hard, interesting case), and
  * a random fraction of written plies (so the model also learns to align/copy).
"""

from __future__ import annotations

import random
from collections.abc import Iterator
from dataclasses import dataclass, field, replace

import chess
import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from recon.data import store
from recon.game import Game
from recon.model import features as F
from recon.model.maia import elo_pair, hist_codes
from recon.observe.corrupt import PRESETS, CorruptionConfig, corrupt

DEFAULT_MIX = {"missing": 0.35, "mixed": 0.30, "mixed_ocr": 0.15, "substitution": 0.10, "clean": 0.10}


@dataclass
class DataConfig:
    games_path: str = "data/games.parquet"
    split: str = "train"
    mix: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_MIX))
    max_plies: int | None = 120
    p_keep_written: float = 0.25
    p_drop_rating: float = 0.1
    p_drop_tc: float = 0.1
    p_gap_known: float = 0.0  # chance the scoresheet layout (plies to next written move) is given
    maia_inputs: bool = False  # add Maia-3 inputs (last 8 boards, ratings) for the hybrid model
    shuffle_buffer: int = 20000
    seed: int = 0


def game_positions(
    game: Game, cfg: CorruptionConfig, rng: random.Random, p_keep_written: float, ctx: F.Context,
    gap_known: bool = False, maia: bool = False,
) -> Iterator[dict[str, np.ndarray]]:
    sources, truth = corrupt(game, cfg, rng)
    tokens = sources[0]["tokens"]
    token_ply = truth["sources"][0]["token_ply"]
    pos_of_ply = {p: t for t, p in enumerate(token_ply) if p is not None}
    written = sorted(pos_of_ply)
    moves = truth["moves_uci"]
    cache = F.TokenCache.build(tokens)
    raws, turns = [], []  # white-frame board (and side to move) before each ply
    b = chess.Board(game.start_fen)
    for uci in moves:
        raws.append(F.raw_squares(b))
        turns.append(b.turn)
        b.push_uci(uci)

    board = chess.Board(game.start_fen)
    last: chess.Move | None = None
    start = 0  # first unconsumed token
    nxt = 0  # index into `written` of the next written ply >= i
    for i, uci in enumerate(moves):
        move = chess.Move.from_uci(uci)
        while nxt < len(written) and written[nxt] < i:
            nxt += 1
        is_written = i in pos_of_ply
        if not is_written or rng.random() < p_keep_written:
            gap = (written[nxt] - i) if nxt < len(written) else F.MAX_GAP
            x = F.encode(board, last, tokens, start, i, ctx, gap if gap_known else None, pos_of_ply, cache)
            fb = np.zeros((F.NB, 64), np.int64)
            fbm = np.zeros(F.NB, np.bool_)
            for j, p in enumerate(written[nxt + (1 if is_written else 0):][: F.NB]):
                fb[j] = F.orient_raw(raws[p], board.turn)
                fbm[j] = True
            x["fut_board"], x["fut_board_mask"] = fb, fbm
            if maia:
                x["maia_hist"] = hist_codes(raws[: i + 1], turns[: i + 1])
                x["maia_elo"] = elo_pair(ctx.white_elo, ctx.black_elo, board.turn)
            if is_written:
                k = pos_of_ply[i] - start
                align = k if k < F.K else F.K
            else:
                align = F.K
            x["move"] = np.int64(F.encode_move(move, board.turn))
            x["promo"] = np.int64(F.PROMOS[move.promotion])
            x["align"] = np.int64(align)
            x["gap"] = np.int64(min(gap, F.MAX_GAP))
            x["hidden"] = np.bool_(not is_written)
            yield x
        if is_written:
            start = pos_of_ply[i] + 1
        board.push(move)
        last = move


def _context(game: Game, dc: DataConfig, rng: random.Random) -> F.Context:
    drop_r = rng.random() < dc.p_drop_rating
    drop_t = rng.random() < dc.p_drop_tc
    return F.Context(
        None if drop_r else game.white_elo,
        None if drop_r else game.black_elo,
        None if drop_t else game.time_control.initial_seconds,
        None if drop_t else game.time_control.increment_seconds,
    )


class PositionStream(IterableDataset):
    def __init__(self, dc: DataConfig):
        self.dc = dc
        self.games: list[Game] | None = None

    def __iter__(self):
        dc = self.dc
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        if self.games is None:
            self.games = list(store.read_games(dc.games_path, split=dc.split, shard=(wid, nw)))
        rng = random.Random(dc.seed * 1000 + wid)
        gap_rng = random.Random(dc.seed * 1000 + wid + 777_777)
        names, weights = zip(*dc.mix.items())
        cfgs = {n: replace(PRESETS[n], max_plies=dc.max_plies) for n in names}
        mine = self.games
        buf: list[dict] = []
        while True:  # infinite stream; trainer decides when to stop
            rng.shuffle(mine)
            for game in mine:
                cfg = cfgs[rng.choices(names, weights)[0]]
                known = gap_rng.random() < dc.p_gap_known  # own stream: positions match runs without it
                for x in game_positions(game, cfg, rng, dc.p_keep_written, _context(game, dc, rng), known, dc.maia_inputs):
                    if len(buf) < dc.shuffle_buffer:
                        buf.append(x)
                    else:
                        j = rng.randrange(len(buf))
                        yield buf[j]
                        buf[j] = x


def fixed_positions(dc: DataConfig, limit_games: int | None = None) -> dict[str, torch.Tensor]:
    """A deterministic evaluation set (e.g. the val split) as stacked tensors."""
    games = list(store.read_games(dc.games_path, split=dc.split))[:limit_games]
    rng = random.Random(dc.seed)
    gap_rng = random.Random(dc.seed + 777_777)  # separate, so val positions are identical across runs
    names, weights = zip(*dc.mix.items())
    cfgs = {n: replace(PRESETS[n], max_plies=dc.max_plies) for n in names}
    rows = []
    for game in games:
        cfg = cfgs[rng.choices(names, weights)[0]]
        known = gap_rng.random() < dc.p_gap_known
        rows.extend(game_positions(game, cfg, rng, dc.p_keep_written, _context(game, dc, rng), known, dc.maia_inputs))
    return {k: torch.from_numpy(np.stack([r[k] for r in rows])) for k in rows[0]}
