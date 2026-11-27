"""Suggestions for user-assisted reconstruction with known gap structure.

Setting: the player has confirmed every move before ply ``i`` and the scoresheet layout
says which plies are blank and which token each written ply belongs to (gap length known).
To rank candidates for ply ``i`` we run, for each candidate first move, a small beam
search through the rest of the gap and the next ``horizon`` written plies, scoring:

    blank ply    log P_model(move)
    written ply  log P_model(move) + log P(token text | move)   (tolerates OCR/slips)

A candidate's score is its prior plus the log-sum-exp over its kept continuations.
(Averaging the evidence over continuations instead was tried and was worse at every distance.)

Search only helps when the next written move is close: deep inside a long gap the few kept
continuations are mostly wrong and add noise (measured: top-5 78% -> 63% at 9+ plies away).
So beyond ``max_dist`` plies from the next written move we rank by the model prior alone. Moves outside the candidate set
are ranked after it by the model prior.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import chess
import numpy as np

from recon.model import features as F
from recon.observe import notation
from recon.solvers.beam import BeamConfig, BeamSolver, _TokenView
from recon.solvers.feasible import Entries, violations


@dataclass
class State:  # duck-types beam.State for BeamSolver._score
    board: chess.Board
    j: int
    moves: list[str]
    blank: float  # sum of log P_model over blank plies after the first move
    evid: float  # sum of log P_model + log P(text|move) over written plies
    last: chess.Move | None

    @property
    def score(self) -> float:
        return self.blank + self.evid


@dataclass
class AssistConfig:
    first_moves: int = 16  # candidate first moves searched (all legal if the gap has 1 ply left)
    width: int = 8  # beam width per candidate first move
    blank_moves: int = 6  # model top-N expanded at blank plies
    written_extra: int = 4  # model top-N tried at written plies besides what the token denotes
    horizon: int = 6  # written plies played out after the current gap
    max_dist: int = 8  # search only if the next written move is at most this many plies away
    penalty: float = 0.0  # feasibility filter: subtract this per violated later entry (inf = remove; 0 = off)


class AssistedRanker:
    def __init__(self, ckpt: str = "runs/v1/best.pt", device: str = "auto", cfg: AssistConfig | None = None):
        self.bs = BeamSolver(ckpt, device)
        self.cfg = cfg or AssistConfig()
        self.obs_cfg = BeamConfig()

    def _lps(self, states: list[State], tokens, ctx):
        out = []
        for lo in range(0, len(states), 256):
            mv, _, pr = self.bs._score(states[lo : lo + 256], tokens, ctx, self._layout, self._cache)
            out.extend(zip(mv, pr))
        return out

    @staticmethod
    def _move_lp(mv, pr, m: chess.Move, turn: bool) -> float:
        v = float(mv[F.encode_move(m, turn)])
        return v + float(pr[F.PROMOS[m.promotion]]) if m.promotion else v

    def rank(
        self,
        inp: dict,
        prefix: list[str],
        ply_token: dict[int, int],
        num_plies: int,
    ) -> list[chess.Move]:
        """Legal moves at ply len(prefix), best first.

        ``ply_token`` maps written plies to their token index in source 0 (the known layout);
        plies absent from it are blank. ``num_plies`` bounds the game length.
        """
        cfg = self.cfg
        tokens = inp["sources"][0]["tokens"]
        ctx = F.Context.from_input(inp)
        # the known layout feeds the model's future-entry tokens (ignored by models without them)
        self._layout, self._cache = (ply_token, F.TokenCache.build(tokens)) if self.bs.model.cfg.future else (None, None)
        board = chess.Board(inp["start_fen"])
        last = None
        start = 0
        for p, u in enumerate(prefix):
            last = chess.Move.from_uci(u)
            board.push(last)
            if p in ply_token:
                start = ply_token[p] + 1
        i = len(prefix)
        root = State(board, start, list(prefix), 0.0, 0.0, last)
        (mv0, pr0), = self._lps([root], tokens, ctx)
        legal = list(board.legal_moves)
        prior = {m: self._move_lp(mv0, pr0, m, board.turn) for m in legal}
        if cfg.penalty:
            ent = Entries(tokens, ply_token)
            for m in legal:
                v = violations(board, m, i, ent)
                if v:
                    prior[m] -= cfg.penalty * v if cfg.penalty != math.inf else 1e6 * v
        by_prior = sorted(legal, key=prior.__getitem__, reverse=True)

        # plies to play out: rest of this gap, then until `horizon` written plies are consumed
        end, seen_written = i, 0
        while end < num_plies and seen_written < cfg.horizon:
            if end in ply_token and end > i:
                seen_written += 1
            end += 1
        gap_left = 0
        while i + gap_left < num_plies and (i + gap_left) not in ply_token:
            gap_left += 1
        if gap_left > cfg.max_dist:
            return by_prior
        firsts = by_prior if gap_left <= 1 else by_prior[: cfg.first_moves]

        # groups: first move -> beam of states (after that first move)
        groups: dict[chess.Move, list[State]] = {}
        for m in firsts:
            nb = board.copy(stack=F.HIST)
            nb.push(m)
            groups[m] = [State(nb, start, prefix + [m.uci()], 0.0, 0.0, m)]
        for p in range(i + 1, end):
            flat = [(g, s) for g, beam in groups.items() for s in beam]
            if not flat:
                break
            lps = self._lps([s for _, s in flat], tokens, ctx)
            children: dict[chess.Move, list[tuple[float, float, State, chess.Move, int]]] = {g: [] for g in groups}
            written = p in ply_token
            for (g, s), (mv, pr) in zip(flat, lps):
                turn = s.board.turn
                if written:
                    index = notation.MoveIndex(s.board)
                    if not index.moves:
                        continue
                    view = _TokenView(index, tokens[ply_token[p]])
                    lp_of = {m: self._move_lp(mv, pr, m, turn) for m in index.moves}
                    cands = set(sorted(index.moves, key=lp_of.__getitem__, reverse=True)[: cfg.written_extra])
                    for _, resolved, _ in view.readings:
                        cands |= resolved
                    for m in cands:
                        e = lp_of[m] + view.logp(m, index.key[m], self.obs_cfg)
                        children[g].append((s.blank, s.evid + e, s, m, ply_token[p] + 1))
                else:
                    moves = list(s.board.legal_moves)
                    if not moves:
                        continue
                    lp_of = {m: self._move_lp(mv, pr, m, turn) for m in moves}
                    for m in sorted(moves, key=lp_of.__getitem__, reverse=True)[: cfg.blank_moves]:
                        children[g].append((s.blank + lp_of[m], s.evid, s, m, s.j))
            new_groups = {}
            for g, ch in children.items():
                ch.sort(key=lambda t: t[0] + t[1], reverse=True)
                beam, seen = [], set()
                for bl, ev, s, m, nj in ch:
                    if len(beam) >= cfg.width:
                        break
                    nb = s.board.copy(stack=F.HIST)
                    nb.push(m)
                    key = nb._transposition_key()
                    if key in seen:
                        continue
                    seen.add(key)
                    beam.append(State(nb, nj, s.moves + [m.uci()], bl, ev, m))
                if beam:
                    new_groups[g] = beam
                else:  # game ended inside the horizon (mate/stalemate): keep as is
                    new_groups[g] = groups[g] if not ch and groups[g][0].board.is_game_over() else []
            groups = {g: b for g, b in new_groups.items() if b}

        def lse(xs):
            mx = max(xs)
            return mx + math.log(sum(math.exp(x - mx) for x in xs))

        def value(g, b):  # prior + log-sum-exp over the kept continuations
            return prior[g] + lse([s.blank + s.evid for s in b])

        scored = sorted(((value(g, b), g) for g, b in groups.items()), key=lambda t: t[0], reverse=True)
        ranked = [g for _, g in scored]
        got = set(ranked)
        return ranked + [m for m in by_prior if m not in got]
