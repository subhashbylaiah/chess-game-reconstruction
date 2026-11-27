"""Observation-aware beam search driven by the notation-aware move model.

Search state: (board, next unconsumed token j, moves so far, alignment, score).
At each ply, every live state is scored by the model in one batch. Children are
formed by pairing a candidate move with an alignment choice:

    none   this ply was not written down              log P_align(none)
    k      token j+k records this ply (tokens          log P_align(k) + log P(text | move)
           j..j+k-1 are extra/duplicated entries)       + k * log P(insert)

Candidate moves are the model's top-M moves plus every move that any of the next
few tokens could denote, so written moves are never pruned for having low prior.
Children are scored first and only the survivors are materialized (board copy),
deduplicated by (position, j). States reaching the end of the notation become
complete hypotheses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import chess
import numpy as np
import torch

from recon.model import features as F
from recon.observe import notation
from recon.solvers.base import register


@dataclass
class BeamConfig:
    beam: int = 32
    top_moves: int = 8
    max_skip: int = 2  # extra tokens that may be skipped before an aligned one
    p_match: float = 0.95
    p_insert: float = 0.01
    dist_penalty: float = 1.0  # per character edit for a mismatched token
    max_extra_plies: int = 60  # beyond the number of tokens
    keep_complete: int = 50
    stop_margin: float = 20.0  # nats behind the best complete game
    progress_bonus: float = 2.0  # ranking bonus per consumed token (simple future heuristic)
    max_unwritten_run: int = 40  # longest allowed run of unrecorded plies
    uniform_prior: int = 0  # 1 = ignore the model's move probabilities (rules-only ablation)


@dataclass
class State:
    board: chess.Board
    j: int
    moves: list[str]
    align: list[int | None]  # per token: ply or None
    score: float
    last: chess.Move | None
    run: int = 0  # consecutive unwritten plies ending here


def _levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for jj, cb in enumerate(b, 1):
            cur.append(min(prev[jj] + 1, cur[-1] + 1, prev[jj - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _token_texts(tok: dict) -> list[tuple[str, float]]:
    if tok.get("alternatives"):
        return [(a["text"], max(a["p"], 1e-3)) for a in tok["alternatives"]]
    return [(tok["text"], 1.0)]


class _TokenView:
    """Per (state, token): which legal moves each written reading denotes."""

    __slots__ = ("readings",)

    def __init__(self, index: notation.MoveIndex, tok: dict):
        self.readings = [
            (notation.normalize(text), set(index.resolve(text)), math.log(p) if p < 1.0 else 0.0)
            for text, p in _token_texts(tok)
        ]

    def logp(self, move: chess.Move, key: str, cfg: BeamConfig) -> float:
        best = -math.inf
        for norm, resolved, lp_read in self.readings:
            if move in resolved:
                lp = math.log(cfg.p_match)
            else:
                lp = math.log(1 - cfg.p_match) - cfg.dist_penalty * _levenshtein(norm, key)
            best = max(best, lp + lp_read)
        return best


class BeamSolver:
    name = "beam"

    def __init__(self, ckpt: str = "runs/v1/best.pt", device: str = "auto", **kw):
        from recon.model.train import load, pick_device

        self.device = pick_device(device)
        self.model = load(ckpt, device)
        types = {f.name: type(f.default) for f in fields(BeamConfig)}
        self.cfg = BeamConfig(**{k: types[k](v) for k, v in kw.items()})

    @torch.no_grad()
    def _score(self, states: list[State], tokens: list[dict], ctx: F.Context, layout=None, cache=None):
        xs = [F.encode(s.board, s.last, tokens, s.j, len(s.moves), ctx, layout=layout, cache=cache) for s in states]
        if getattr(self.model.cfg, "maia", ""):  # hybrid model: Maia tower needs recent boards + ratings
            from recon.model.maia import board_inputs
            for x, s in zip(xs, states):
                x["maia_hist"], x["maia_elo"] = board_inputs(s.board, ctx)
        batch = {k: torch.from_numpy(np.stack([x[k] for x in xs])).to(self.device) for k in xs[0]}
        out = self.model(batch)
        return (
            torch.log_softmax(out["move"].float(), -1).cpu().numpy(),
            torch.log_softmax(out["align"].float(), -1).cpu().numpy(),
            torch.log_softmax(out["promo"].float(), -1).cpu().numpy(),
        )

    @torch.no_grad()
    def score_path(self, inp: dict, moves: list[str], align: list[int | None]) -> float:
        """Score a given (moves, token alignment) with exactly the search's scoring.

        Used for diagnosis: if the true game scores higher than the search's best answer,
        the search lost it (search error); if lower, the model prefers a wrong game (model error).
        """
        cfg = self.cfg
        tokens = inp["sources"][0]["tokens"]
        n_tok = len(tokens)
        ctx = F.Context.from_input(inp)
        lp_ins = math.log(cfg.p_insert)
        tok_of_ply = {p: t for t, p in enumerate(align) if p is not None}
        states, steps = [], []
        board, last, j = chess.Board(inp["start_fen"]), None, 0
        for i, u in enumerate(moves):
            m = chess.Move.from_uci(u)
            states.append(State(board.copy(stack=F.HIST), j, moves[:i], align, 0.0, last))
            k = tok_of_ply[i] - j if i in tok_of_ply else None
            steps.append((m, k))
            board.push(m)
            last = m
            if k is not None:
                j = tok_of_ply[i] + 1
        total = (n_tok - j) * lp_ins
        for lo in range(0, len(states), 256):
            chunk = states[lo : lo + 256]
            move_lp, align_lp, promo_lp = self._score(chunk, tokens, ctx)
            for ci, (st, (m, k)) in enumerate(zip(chunk, steps[lo : lo + 256])):
                lp = float(move_lp[ci, F.encode_move(m, st.board.turn)])
                if m.promotion:
                    lp += float(promo_lp[ci, F.PROMOS[m.promotion]])
                if cfg.uniform_prior:
                    lp = -math.log(st.board.legal_moves.count())
                if k is None:
                    total += lp + float(align_lp[ci, F.K])
                elif k >= F.K:
                    return -math.inf  # outside the model's window; search could not produce it
                else:
                    index = notation.MoveIndex(st.board)
                    o = _TokenView(index, tokens[st.j + k]).logp(m, index.key[m], cfg)
                    total += lp + float(align_lp[ci, k]) + o + k * lp_ins
        return total

    def solve(self, inp: dict) -> dict:
        cfg = self.cfg
        tokens = inp["sources"][0]["tokens"]
        n_tok = len(tokens)
        ctx = F.Context.from_input(inp)
        lp_ins = math.log(cfg.p_insert)
        beam = [State(chess.Board(inp["start_fen"]), 0, [], [None] * n_tok, 0.0, None)]
        complete: list[State] = []
        evals = expanded = 0

        for _ in range(n_tok + cfg.max_extra_plies):
            if not beam:
                break
            move_lp, align_lp, promo_lp = self._score(beam, tokens, ctx)
            evals += len(beam)

            # 1. score all children cheaply: (score, state index, move, k)
            scored: list[tuple[float, int, chess.Move, int | None]] = []
            for si, s in enumerate(beam):
                index = notation.MoveIndex(s.board)
                may_skip = s.run < cfg.max_unwritten_run
                legal = index.moves
                if not legal:
                    continue
                turn = s.board.turn
                lp_of = {}
                for m in legal:
                    lp = float(move_lp[si, F.encode_move(m, turn)])
                    if m.promotion:  # split the (from,to) mass over promotion pieces
                        lp += float(promo_lp[si, F.PROMOS[m.promotion]])
                    lp_of[m] = -math.log(len(legal)) if cfg.uniform_prior else lp
                n_k = min(cfg.max_skip + 1, n_tok - s.j, F.K)
                views = [_TokenView(index, tokens[s.j + k]) for k in range(n_k)]
                cands = set(sorted(legal, key=lp_of.__getitem__, reverse=True)[: cfg.top_moves])
                for v in views:
                    for _, resolved, _ in v.readings:
                        cands |= resolved
                for m in cands:
                    base = s.score + lp_of[m]
                    if may_skip:
                        scored.append((base + float(align_lp[si, F.K]), si, m, None))
                    key = index.key[m]
                    for k, v in enumerate(views):
                        o = v.logp(m, key, cfg)
                        scored.append((base + float(align_lp[si, k]) + o + k * lp_ins, si, m, k))
            expanded += len(scored)
            bonus = cfg.progress_bonus

            def rank(t):  # score + bonus for tokens consumed after this child
                sc, si, _, k = t
                return sc + bonus * (beam[si].j + (0 if k is None else k + 1))

            scored.sort(key=rank, reverse=True)

            # 2. materialize survivors, deduplicating transpositions
            new_beam: list[State] = []
            seen: set = set()
            for sc, si, m, k in scored:
                if len(new_beam) >= cfg.beam:
                    break
                s = beam[si]
                nj = s.j if k is None else s.j + k + 1
                nb = s.board.copy(stack=F.HIST)
                nb.push(m)
                key = (nb._transposition_key(), nj)
                if key in seen:
                    continue
                seen.add(key)
                align = s.align
                if k is not None:
                    align = list(s.align)
                    align[s.j + k] = len(s.moves)
                c = State(nb, nj, s.moves + [m.uci()], align, sc, m, s.run + 1 if k is None else 0)
                remaining = n_tok - nj
                if remaining == 0:
                    complete.append(c)
                    continue
                if remaining <= cfg.max_skip:  # trailing extra tokens: may finish here
                    complete.append(State(nb, n_tok, c.moves, align, sc + remaining * lp_ins, m))
                new_beam.append(c)
            beam = new_beam
            complete = sorted(complete, key=lambda c: c.score, reverse=True)[: cfg.keep_complete]
            # Scores only decrease along a path, so stop once live states can no longer
            # enter the kept set, or are hopelessly behind the best complete game.
            if complete and beam:
                full = len(complete) >= cfg.keep_complete and beam[0].score < complete[-1].score
                if full or beam[0].score < complete[0].score - cfg.stop_margin:
                    break

        complete.sort(key=lambda c: c.score, reverse=True)
        uniq, seen_moves = [], set()
        for c in complete:
            t = tuple(c.moves)
            if t not in seen_moves:
                seen_moves.add(t)
                uniq.append(c)

        cands = []
        if uniq:
            scores = np.array([c.score for c in uniq])
            w = np.exp(scores - scores.max())
            w /= w.sum()
            top = uniq[0].moves
            conf = [
                float(sum(wi for wi, c in zip(w, uniq) if i < len(c.moves) and c.moves[i] == mv))
                for i, mv in enumerate(top)
            ]
            for r, c in enumerate(uniq):
                align = [c.align] + [[None] * len(s["tokens"]) for s in inp["sources"][1:]]
                d = {"moves_uci": c.moves, "score": float(c.score), "alignment": align}
                if r == 0:
                    d["move_confidence"] = conf
                cands.append(d)
        return {"candidates": cands, "stats": {"model_evals": evals, "expanded": expanded}}


register("beam")(BeamSolver)
