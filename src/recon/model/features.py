"""Turn (board, upcoming notation, context) into fixed-size integer arrays, and back.

Everything is encoded from the side-to-move's point of view: when Black is to move the
board is mirrored (ranks flipped, colours swapped), and rank digits in the written
notation are mirrored the same way (``"Nf6"`` -> ``"Nf3"``), so the model sees one
consistent geometry.

Per position the model receives:
    squares      [64]        piece id 0..12 (0 empty, 1-6 own PNBRQK, 7-12 opponent)
    last_move    [64]        0 none, 1 from-square of the previous ply, 2 to-square
    context      [N_CTX]     castling, en passant, my/opp rating, time control base/inc
    note_chars   [K, L]      upcoming written tokens as character ids
    note_feats   [K, 2]      relative move-number hint id, OCR confidence id
    note_mask    [K]         1 where a token exists
    note_struct  [K, 6]      parsed entry: piece, to-square, capture, from-file, from-rank, promotion
                             (0 = unknown/absent in each field; used when ModelConfig.struct_notes)
    gap_in       []          plies until the next written ply if the layout is known, else 0
                             (1 + min(gap, MAX_GAP); used when ModelConfig.gap_input)
    move_num     []          full-move number, clipped to 1..MAX_MOVENUM (used when ModelConfig.move_number)
    hist         [HIST, 4]   last HIST moves, most recent first: mover-relative piece (0 none,
                             1-6 own PNBRQK, 7-12 opponent), from+1, to+1, capture
                             (used when ModelConfig.hist_moves > 0)
    legal        [4096]      1 for legal (from, to) pairs (promotions collapse to one pair)
    fut_chars    [NF, L]     every written entry AFTER this ply (layout known), nearest first, as characters
    fut_struct   [NF, 6]     the same entries parsed (see note_struct)
    fut_feats    [NF, 2]     ply-offset bucket (1..16 exact, then coarser; 0 pad), OCR confidence id
    fut_mask     [NF]        1 where an entry exists (empty when the layout is not given)
                             (all fut_* used when ModelConfig.future > 0)
Labels:
    move         from*64+to of the move played (mirrored frame)
    promo        0 none, 1..4 = N,B,R,Q
    align        index 0..K-1 of the token that records this ply, or K if not written / beyond window
    gap          plies until the next written ply (0 if this ply is written), clipped to MAX_GAP
    fut_board    [NB, 64]    training-only target: the board just before each of the next NB written
                             plies, in this position's frame (0 empty, 1-6 own, 7-12 opponent)
    fut_board_mask [NB]      1 where that written ply exists
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import chess
import numpy as np

K = 16  # upcoming notation tokens in the window
L = 8  # characters per token
MAX_GAP = 40

CHARS = "abcdefgh12345678NBRQKOx+#=-:0?!/"
CHAR_IDS = {c: i + 2 for i, c in enumerate(CHARS)}  # 0 pad, 1 unknown
N_CHARS = len(CHARS) + 2

N_HINT = 34  # 0 unknown, 1..33 = offset -8..+24 (clipped)
N_CONF = 11  # 0 unknown, 1..10 deciles
RATING_BUCKETS = 26  # 0 unknown, 1..25 = <700, 700-799, ..., 2900+
TC_BASE = [0, 30, 60, 120, 180, 300, 420, 600, 900, 1200, 1800, 2700, 3600, 5400]
TC_INC = [0, 1, 2, 3, 5, 10, 15, 20, 30, 60]
CTX_SIZES = [16, 9, RATING_BUCKETS, RATING_BUCKETS, len(TC_BASE) + 1, len(TC_INC) + 1]
N_CTX = len(CTX_SIZES)
PROMOS = {None: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4}
PROMO_PIECES = [None, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]

_MIRROR_DIGITS = str.maketrans("12345678", "87654321")

# Parsed-entry fields. Pieces: 0 unparsed, 1..6 PNBRQK, 7 O-O, 8 O-O-O.
N_PIECE_F, N_SQ_F, N_FILE_F, N_RANK_F, N_PROMO_F = 9, 65, 9, 9, 5
N_GAP_IN = MAX_GAP + 2
MAX_MOVENUM = 100
N_MOVENUM = MAX_MOVENUM + 1
HIST = 8  # most moves of history the encoder extracts (boards must keep this much stack)
NF = 48  # most written entries after the current ply given as future tokens
NB = 2  # future boards predicted as an auxiliary training target
_OFF_EDGES = [24, 32, 48, 64]  # offsets 1..16 exact; then buckets (16,24], (24,32], (32,48], (48,64], >64
N_OFF = 16 + len(_OFF_EDGES) + 2  # 0 pad


def offset_id(off: int) -> int:
    if off <= 16:
        return off
    for b, edge in enumerate(_OFF_EDGES):
        if off <= edge:
            return 17 + b
    return 17 + len(_OFF_EDGES)
_PIECE_LETTER = {"N": 2, "B": 3, "R": 4, "Q": 5, "K": 6}
_ENTRY = re.compile(
    r"^(?P<p>[NBRQK])?(?P<ff>[a-h])?(?P<fr>[1-8])?(?P<x>[x:])?-?(?P<tf>[a-h])(?P<tr>[1-8])=?(?P<pr>[NBRQ])?$"
)
_CASTLE = re.compile(r"^O-?O(?P<long>-?O)?$")
_CLEAN = re.compile(r"[+#!?\s]|e\.p\.")
_FIG = str.maketrans({"♔": "K", "♕": "Q", "♖": "R", "♗": "B", "♘": "N", "♚": "K", "♛": "Q", "♜": "R", "♝": "B", "♞": "N"})


def parse_entry(text: str) -> tuple[int, int, int, int, int, int]:
    """Parse a written move (already in the side-to-move frame) into structured fields.

    Returns (piece, to_sq+1, capture, from_file+1, from_rank+1, promo); zeros mean unknown.
    Garbled text that matches no pattern returns all zeros, so the character view must carry it.
    """
    t = _CLEAN.sub("", text.translate(_FIG).replace("0", "O"))
    c = _CASTLE.match(t)
    if c:  # own king lands on g1 / c1 in the oriented frame
        return (8, chess.C1 + 1, 0, 0, 0, 0) if c.group("long") else (7, chess.G1 + 1, 0, 0, 0, 0)
    m = _ENTRY.match(t)
    if not m:
        return (0, 0, 0, 0, 0, 0)
    piece = _PIECE_LETTER[m.group("p")] if m.group("p") else 1
    to = chess.square(ord(m.group("tf")) - 97, int(m.group("tr")) - 1)
    ff = ord(m.group("ff")) - 96 if m.group("ff") else 0
    fr = int(m.group("fr")) if m.group("fr") else 0
    pr = {"N": 1, "B": 2, "R": 3, "Q": 4}[m.group("pr")] if m.group("pr") else 0
    return (piece, to + 1, int(bool(m.group("x"))), ff, fr, pr)


def rating_bucket(elo: int | None) -> int:
    if elo is None:
        return 0
    return int(np.clip((elo - 600) // 100, 0, RATING_BUCKETS - 2)) + 1


def _bucket(value: int | None, edges: list[int]) -> int:
    if value is None:
        return 0
    return int(np.searchsorted(edges, value, side="right"))  # 1..len(edges)


@dataclass
class Context:
    white_elo: int | None = None
    black_elo: int | None = None
    tc_initial: int | None = None
    tc_increment: int | None = None

    @classmethod
    def from_input(cls, inp: dict) -> Context:
        p, tc = inp.get("players", {}), inp.get("time_control", {})
        return cls(
            (p.get("white") or {}).get("rating"),
            (p.get("black") or {}).get("rating"),
            tc.get("initial_seconds"),
            tc.get("increment_seconds"),
        )


def _orient(board: chess.Board) -> chess.Board:
    return board if board.turn == chess.WHITE else board.mirror()


def encode_move(move: chess.Move, turn: chess.Color) -> int:
    f, t = move.from_square, move.to_square
    if turn == chess.BLACK:
        f, t = chess.square_mirror(f), chess.square_mirror(t)
    return f * 64 + t


def decode_index(idx: int, turn: chess.Color) -> tuple[int, int]:
    f, t = divmod(idx, 64)
    if turn == chess.BLACK:
        f, t = chess.square_mirror(f), chess.square_mirror(t)
    return f, t


def encode_board(board: chess.Board, last_move: chess.Move | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    b = _orient(board)
    squares = np.zeros(64, np.int64)
    for sq, piece in b.piece_map().items():
        squares[sq] = piece.piece_type + (0 if piece.color == chess.WHITE else 6)
    last = np.zeros(64, np.int64)
    if last_move is not None:
        f, t = last_move.from_square, last_move.to_square
        if board.turn == chess.BLACK:
            f, t = chess.square_mirror(f), chess.square_mirror(t)
        last[f], last[t] = 1, 2
    legal = np.zeros(4096, np.bool_)
    for m in board.legal_moves:
        legal[encode_move(m, board.turn)] = True
    return squares, last, legal


def encode_history(board: chess.Board) -> np.ndarray:
    """The last HIST moves from ``board.move_stack``, in the side-to-move frame."""
    hist = np.zeros((HIST, 4), np.int64)
    n = min(HIST, len(board.move_stack))
    if n == 0:
        return hist
    b = board.copy(stack=n)
    recent = [b.pop() for _ in range(n)]  # recent[0] = last move played
    for h in reversed(range(n)):  # replay oldest first so each move sees its own board
        m = recent[h]
        piece = b.piece_at(m.from_square)
        f, t = m.from_square, m.to_square
        if board.turn == chess.BLACK:
            f, t = chess.square_mirror(f), chess.square_mirror(t)
        own = b.turn == board.turn
        hist[h] = (piece.piece_type + (0 if own else 6) if piece else 0, f + 1, t + 1, int(b.is_capture(m)))
        b.push(m)
    return hist


def encode_context(board: chess.Board, ctx: Context) -> np.ndarray:
    b = _orient(board)
    castle = (
        int(b.has_kingside_castling_rights(chess.WHITE))
        | int(b.has_queenside_castling_rights(chess.WHITE)) << 1
        | int(b.has_kingside_castling_rights(chess.BLACK)) << 2
        | int(b.has_queenside_castling_rights(chess.BLACK)) << 3
    )
    ep = 0 if b.ep_square is None else chess.square_file(b.ep_square) + 1
    mine, theirs = (ctx.white_elo, ctx.black_elo) if board.turn == chess.WHITE else (ctx.black_elo, ctx.white_elo)
    return np.array(
        [
            castle,
            ep,
            rating_bucket(mine),
            rating_bucket(theirs),
            _bucket(ctx.tc_initial, TC_BASE),
            _bucket(ctx.tc_increment, TC_INC),
        ],
        np.int64,
    )


def token_ply_hint(tok: dict) -> int | None:
    n, side = tok.get("move_number_hint"), tok.get("side_hint")
    if n is None:
        return None
    return (n - 1) * 2 + (1 if side == "black" else 0)


def encode_notation(tokens: list[dict], start: int, ply: int, turn: chess.Color) -> tuple[np.ndarray, ...]:
    chars = np.zeros((K, L), np.int64)
    feats = np.zeros((K, 2), np.int64)
    struct = np.zeros((K, 6), np.int64)
    mask = np.zeros(K, np.bool_)
    for k, tok in enumerate(tokens[start : start + K]):
        text = tok["text"]
        if turn == chess.BLACK:
            text = text.translate(_MIRROR_DIGITS)
        struct[k] = parse_entry(text)
        for c, ch in enumerate(text[:L]):
            chars[k, c] = CHAR_IDS.get(ch, 1)
        hint = token_ply_hint(tok)
        feats[k, 0] = 0 if hint is None else int(np.clip(hint - ply, -8, 24)) + 9
        conf = tok.get("confidence")
        feats[k, 1] = 0 if conf is None else min(int(conf * 10), 9) + 1
        mask[k] = True
    return chars, feats, mask, struct


@dataclass
class TokenCache:
    """Per-scoresheet token arrays in both frames, so per-position encoding is just indexing."""

    chars: dict  # turn -> [T, L]
    struct: dict  # turn -> [T, 6]
    conf: np.ndarray  # [T]

    @classmethod
    def build(cls, tokens: list[dict]) -> TokenCache:
        T = len(tokens)
        chars = {c: np.zeros((T, L), np.int64) for c in (chess.WHITE, chess.BLACK)}
        struct = {c: np.zeros((T, 6), np.int64) for c in (chess.WHITE, chess.BLACK)}
        conf = np.zeros(T, np.int64)
        for t, tok in enumerate(tokens):
            for turn in (chess.WHITE, chess.BLACK):
                text = tok["text"] if turn == chess.WHITE else tok["text"].translate(_MIRROR_DIGITS)
                struct[turn][t] = parse_entry(text)
                for c, ch in enumerate(text[:L]):
                    chars[turn][t, c] = CHAR_IDS.get(ch, 1)
            cf = tok.get("confidence")
            conf[t] = 0 if cf is None else min(int(cf * 10), 9) + 1
        return cls(chars, struct, conf)


def encode_future(cache: TokenCache | None, layout: dict[int, int] | None, ply: int, turn: chess.Color):
    """Written entries at plies after ``ply`` (from the known layout ply -> token index), nearest first."""
    chars = np.zeros((NF, L), np.int64)
    struct = np.zeros((NF, 6), np.int64)
    feats = np.zeros((NF, 2), np.int64)
    mask = np.zeros(NF, np.bool_)
    if layout:
        later = sorted(p for p in layout if p > ply)[:NF]
        if later:
            idx = np.array([layout[p] for p in later])
            n = len(later)
            chars[:n] = cache.chars[turn][idx]
            struct[:n] = cache.struct[turn][idx]
            feats[:n, 0] = [offset_id(p - ply) for p in later]
            feats[:n, 1] = cache.conf[idx]
            mask[:n] = True
    return chars, struct, feats, mask


def raw_squares(board: chess.Board) -> np.ndarray:
    """White-frame piece codes (1-6 white PNBRQK, 7-12 black), for building future-board targets."""
    sq = np.zeros(64, np.int64)
    for s, piece in board.piece_map().items():
        sq[s] = piece.piece_type + (0 if piece.color == chess.WHITE else 6)
    return sq


_MIRROR_SQ = np.array([chess.square_mirror(s) for s in range(64)])


def orient_raw(raw: np.ndarray, turn: chess.Color) -> np.ndarray:
    """White-frame codes -> the frame of the side to move (as encode_board does)."""
    if turn == chess.WHITE:
        return raw
    out = raw[_MIRROR_SQ].copy()
    nz = out > 0
    out[nz] = np.where(out[nz] <= 6, out[nz] + 6, out[nz] - 6)
    return out


def encode(
    board: chess.Board,
    last_move: chess.Move | None,
    tokens: list[dict],
    start: int,
    ply: int,
    ctx: Context,
    gap: int | None = None,
    layout: dict[int, int] | None = None,
    cache: TokenCache | None = None,
) -> dict[str, np.ndarray]:
    """``gap``: plies until the next written ply when the scoresheet layout is known.
    ``layout``: known scoresheet layout (written ply -> token index); fills the future tokens."""
    squares, last, legal = encode_board(board, last_move)
    chars, feats, mask, struct = encode_notation(tokens, start, ply, board.turn)
    if layout and cache is None:
        cache = TokenCache.build(tokens)
    fc, fs, ff, fm = encode_future(cache, layout, ply, board.turn)
    return {
        "fut_chars": fc,
        "fut_struct": fs,
        "fut_feats": ff,
        "fut_mask": fm,
        "squares": squares,
        "last_move": last,
        "context": encode_context(board, ctx),
        "note_chars": chars,
        "note_feats": feats,
        "note_mask": mask,
        "note_struct": struct,
        "gap_in": np.int64(0 if gap is None else min(gap, MAX_GAP) + 1),
        "move_num": np.int64(min(board.fullmove_number, MAX_MOVENUM)),
        "hist": encode_history(board),
        "legal": legal,
    }


def move_from_outputs(board: chess.Board, idx: int, promo: int) -> chess.Move:
    f, t = decode_index(idx, board.turn)
    move = chess.Move(f, t)
    piece = board.piece_at(f)
    if piece and piece.piece_type == chess.PAWN and chess.square_rank(t) in (0, 7):
        move.promotion = PROMO_PIECES[promo] or chess.QUEEN
    return move
