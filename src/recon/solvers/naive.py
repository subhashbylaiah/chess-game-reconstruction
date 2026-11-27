"""Naive floor: read the first scoresheet left to right, play every token that parses,
skip every token that does not. Never invents unrecorded moves.
"""

from __future__ import annotations

import chess

from recon.observe import notation
from recon.solvers.base import register


@register("naive")
class NaiveSolver:
    name = "naive"

    def solve(self, inp: dict) -> dict:
        board = chess.Board(inp["start_fen"])
        moves: list[str] = []
        alignment: list[list[int | None]] = [[None] * len(s["tokens"]) for s in inp["sources"]]
        for t, tok in enumerate(inp["sources"][0]["tokens"]):
            texts = [tok["text"]] + [a["text"] for a in (tok.get("alternatives") or [])[1:]]
            for text in texts:
                cands = notation.resolve(board, text)
                if cands:
                    alignment[0][t] = len(moves)
                    moves.append(cands[0].uci())
                    board.push(cands[0])
                    break
        return {
            "candidates": [{"moves_uci": moves, "score": 0.0, "alignment": alignment}],
            "stats": {"model_evals": 0, "expanded": len(moves)},
        }
