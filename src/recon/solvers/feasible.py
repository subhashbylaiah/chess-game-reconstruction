"""Deterministic feasibility checks for a candidate move, given the moves written later.

Setting: user-assisted, gap length known. We are at ply ``i`` (all earlier moves confirmed) and
the scoresheet layout says which later plies are written and what was written there. For a
candidate move ``m`` we ask: can the game still reach each later written move, however the
blank plies in between were played? We use only conditions that hold for *every* path, so no
search is needed:

    castling   a castle written later needs that castling right; rights never come back
    reach      the piece making a later written move must exist now (captured pieces never
               return; promotion allowed for) and be able to reach the destination in the moves
               its side has left (empty-board distance: blockers ignored, so a lower bound)
    target     a capture written later needs an enemy piece that can be on that square in time

Each check is a necessary condition, so a violation proves the candidate impossible *if the
written entry is right*. OCR misreads are handled by requiring every plausible reading of the
entry to be violated. Writer slips (a wrong but clean move) cannot be detected, which is why
the caller applies violations as a penalty and we measure how often the true move is hit.
"""

from __future__ import annotations

import chess

from recon.model import features as F
from recon.solvers.beam import _token_texts

INF = 99
MIN_READING_P = 0.05  # ignore OCR readings less likely than this


def _knight_table() -> list[list[int]]:
    jumps = [(1, 2), (2, 1), (-1, 2), (-2, 1), (1, -2), (2, -1), (-1, -2), (-2, -1)]
    table = []
    for s in chess.SQUARES:
        d = [INF] * 64
        d[s], frontier, k = 0, [s], 0
        while frontier:
            k += 1
            nxt = []
            for q in frontier:
                f, r = chess.square_file(q), chess.square_rank(q)
                for df, dr in jumps:
                    if 0 <= f + df < 8 and 0 <= r + dr < 8:
                        t = chess.square(f + df, r + dr)
                        if d[t] == INF:
                            d[t] = k
                            nxt.append(t)
            frontier = nxt
        table.append(d)
    return table


_KNIGHT = _knight_table()


def _pawn_dist(color: bool, a: int, b: int) -> int:
    """Lower bound on pawn moves from a to b (forward only; file changes need captures)."""
    fwd = (chess.square_rank(b) - chess.square_rank(a)) * (1 if color else -1)
    if fwd < 1 or abs(chess.square_file(b) - chess.square_file(a)) > fwd:
        return INF
    start = chess.square_rank(a) == (1 if color else 6)
    return fwd - 1 if start and fwd >= 2 else fwd


def _to_promo(color: bool, a: int) -> int:
    """Lower bound on pawn moves to promote."""
    fwd = (7 - chess.square_rank(a)) if color else chess.square_rank(a)
    start = chess.square_rank(a) == (1 if color else 6)
    return fwd - 1 if start else fwd


def dist(ptype: int, color: bool, a: int, b: int) -> int:
    """Lower bound on moves for a piece to go from a to b on an empty board (0 if a == b)."""
    if a == b:
        return 0
    fa, ra, fb, rb = chess.square_file(a), chess.square_rank(a), chess.square_file(b), chess.square_rank(b)
    df, dr = abs(fa - fb), abs(ra - rb)
    if ptype == chess.PAWN:
        return _pawn_dist(color, a, b)
    if ptype == chess.KNIGHT:
        return _KNIGHT[a][b]
    if ptype == chess.KING:
        return max(df, dr)
    if ptype == chess.ROOK:
        return 1 if df == 0 or dr == 0 else 2
    if ptype == chess.BISHOP:
        if (fa + ra) % 2 != (fb + rb) % 2:
            return INF
        return 1 if df == dr else 2
    return 1 if df == 0 or dr == 0 or df == dr else 2  # queen


def _moves_onto(board: chess.Board, ptype: int, color: bool, target: int) -> int:
    """Fewest moves (lower bound) before some piece of this type and colour can MOVE ONTO target."""
    best = INF
    for sq in board.pieces(ptype, color):
        if sq == target:  # must leave and come back (a pawn can't)
            best = min(best, 2 if ptype != chess.PAWN else INF)
        else:
            best = min(best, dist(ptype, color, sq, target))
    if ptype not in (chess.PAWN, chess.KING):  # a pawn may promote to it first
        for sq in board.pieces(chess.PAWN, color):
            best = min(best, _to_promo(color, sq) + 1)
    return best


def _can_occupy(board: chess.Board, color: bool, target: int, n: int) -> bool:
    """Can some piece of ``color`` be standing on target after at most n of its moves?"""
    if board.color_at(target) == color:
        return True
    for ptype in chess.PIECE_TYPES:
        for sq in board.pieces(ptype, color):
            if dist(ptype, color, sq, target) <= n:
                return True
            if ptype == chess.PAWN and _to_promo(color, sq) + 1 <= n:
                return True
    return False


def _parse(text: str):
    """(kind, piece, to, capture). kind: 'K'/'Q' castle, 'm' move, None unknown."""
    t = text.strip().replace("0", "O")
    core = t.rstrip("+#!?")
    if core in ("O-O", "OO"):
        return ("K", None, None, False)
    if core in ("O-O-O", "OOO"):
        return ("Q", None, None, False)
    piece, to, cap, _, _, _ = F.parse_entry(text)
    if not to or piece not in range(1, 7):
        return (None, None, None, False)
    return ("m", piece, to - 1, bool(cap))


def _violates(board: chess.Board, reading, mover: bool, n_mover: int, n_other: int) -> bool:
    kind, piece, to, cap = reading
    if kind is None:
        return False
    if kind == "K":
        return not board.has_kingside_castling_rights(mover)
    if kind == "Q":
        return not board.has_queenside_castling_rights(mover)
    if _moves_onto(board, piece, mover, to) > n_mover + 1:
        return True
    if cap:
        ep_rank = 5 if mover == chess.WHITE else 2
        if piece == chess.PAWN and chess.square_rank(to) == ep_rank:
            return False  # could be en passant onto an empty square
        if not _can_occupy(board, not mover, to, n_other):
            return True
    return False


class Entries:
    """Later written entries for one game: ply -> list of plausible parsed readings."""

    def __init__(self, tokens: list[dict], ply_token: dict[int, int]):
        self.by_ply = {}
        for p, t in ply_token.items():
            reads = [_parse(txt) for txt, pr in _token_texts(tokens[t]) if pr >= MIN_READING_P]
            reads = [r for r in reads if r[0] is not None] if all(r[0] is not None for r in reads) else []
            if reads:  # an unreadable reading means the entry gives no safe constraint
                self.by_ply[p] = reads
        self.plies = sorted(self.by_ply)


def violations(board: chess.Board, move: chess.Move, i: int, entries: Entries, max_entries: int = 4) -> int:
    """Number of later written entries the candidate ``move`` at ply ``i`` makes impossible.

    ``board`` is the position before the move. Castles are checked for every later entry, the
    other checks for the next ``max_entries`` written plies (further ones are rarely binding).
    """
    b = board.copy(stack=False)
    side_i = b.turn
    b.push(move)
    bad, seen = 0, 0
    for p in entries.plies:
        if p <= i:
            continue
        seen += 1
        mover = side_i if (p - i) % 2 == 0 else not side_i
        between = p - i - 1  # blank-or-written plies strictly between the candidate and ply p
        # plies i+1..p-1 alternate starting with the opponent of side_i
        n_other_of_side_i = (between + 1) // 2
        n_side_i = between // 2
        n_mover = n_side_i if mover == side_i else n_other_of_side_i
        n_other = n_other_of_side_i if mover == side_i else n_side_i
        reads = entries.by_ply[p]
        castle_only = all(r[0] in ("K", "Q") for r in reads)
        if seen > max_entries and not castle_only:
            continue
        if all(_violates(b, r, mover, n_mover, n_other) for r in reads):
            bad += 1
    return bad
