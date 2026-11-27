"""Observed notation: tolerant token resolution and realistic error generators.

A written token only means something relative to a board: ``"Bc5"`` may be legal,
illegal or ambiguous depending on the position. ``resolve`` therefore maps
(board, text) to the set of legal moves the text could denote.
"""

from __future__ import annotations

import random
import re

import chess

_STRIP = re.compile(r"[+#!?x:\-=\s]|e\.p\.")
_FIGURINES = str.maketrans({"♔": "K", "♕": "Q", "♖": "R", "♗": "B", "♘": "N", "♚": "K", "♛": "Q", "♜": "R", "♝": "B", "♞": "N"})


def normalize(text: str) -> str:
    """Canonical comparison key: strips capture/check marks, separators and annotations.

    ``"Nxf3+"``, ``"Nf3"`` and ``"N:f3"`` all map to ``"Nf3"``; ``"0-0"`` maps to ``"OO"``.
    Promotion ``"e8=Q"`` becomes ``"e8Q"``.
    """
    t = text.translate(_FIGURINES).replace("0", "O")
    return _STRIP.sub("", t)


def _san_no_suffix(board: chess.Board, move: chess.Move) -> str:
    # Check/mate suffixes are stripped by normalize() anyway; skipping them avoids a
    # push/pop per move. Falls back to the public API if the private helper disappears.
    try:
        return board._algebraic_without_suffix(move)
    except AttributeError:
        return board.san(move)


def move_key(board: chess.Board, move: chess.Move) -> str:
    """Normalized SAN of a legal move: what a correct, tidy scoresheet entry normalizes to."""
    return normalize(_san_no_suffix(board, move))


class MoveIndex:
    """All legal moves of one position, indexed by every way a human might write them.

    Build once per position and reuse for every token compared against it.
    """

    __slots__ = ("moves", "key", "exact", "loose")

    def __init__(self, board: chess.Board):
        self.moves = list(board.legal_moves)
        self.key: dict[chess.Move, str] = {}
        self.exact: dict[str, list[chess.Move]] = {}
        self.loose: dict[str, list[chess.Move]] = {}
        for m in self.moves:
            k = normalize(_san_no_suffix(board, m))
            self.key[m] = k
            self.exact.setdefault(k, []).append(m)
            loose = {m.uci()}
            piece = board.piece_at(m.from_square)
            if piece and piece.piece_type != chess.PAWN:
                letter = piece.symbol().upper()
                loose.add(letter + m.uci())  # long algebraic "Ng1f3"
                # Under-disambiguation is common when writing by hand: "Nd2" for "Nbd2".
                loose.add(letter + chess.square_name(m.to_square))
            for lk in loose:
                self.loose.setdefault(lk, []).append(m)

    def resolve(self, text: str) -> list[chess.Move]:
        key = normalize(text)
        if not key:
            return []
        return list(self.exact.get(key) or self.loose.get(key) or ())


def resolve(board: chess.Board, text: str) -> list[chess.Move]:
    """All legal moves the written token could denote (empty if none)."""
    return MoveIndex(board).resolve(text)


def matches(board: chess.Board, move: chess.Move, text: str) -> bool:
    return move in resolve(board, text)


# --------------------------------------------------------------------------- styles

def style(san: str, rng: random.Random, p_strip_marks: float = 0.3, p_zero_castle: float = 0.3) -> str:
    """Render a SAN move the way a human might legitimately write it (no error)."""
    out = san
    if rng.random() < p_strip_marks:
        out = out.replace("+", "").replace("#", "")
    if rng.random() < p_strip_marks * 0.5:
        out = out.replace("x", "")
    if out.startswith("O-O") and rng.random() < p_zero_castle:
        out = out.replace("O", "0")
    return out


# --------------------------------------------------------------------------- OCR confusions

# Visually confusable characters in handwriting / OCR. Includes some junk outputs on purpose.
OCR_CONFUSIONS: dict[str, str] = {
    "a": "dq",
    "b": "dh6",
    "c": "e",
    "d": "ba",
    "e": "c",
    "f": "t",
    "g": "q9",
    "h": "bn",
    "1": "7l",
    "2": "z",
    "3": "8",
    "4": "9",
    "5": "6s",
    "6": "5b",
    "7": "1",
    "8": "3B",
    "N": "K",
    "B": "8R",
    "R": "B",
    "Q": "O",
    "K": "N",
}


def ocr_corrupt(text: str, rng: random.Random, max_edits: int = 2) -> str:
    """Apply 1..max_edits visually-plausible character confusions (or drop a char)."""
    chars: list[str] = list(text)
    idxs = [i for i, c in enumerate(chars) if c in OCR_CONFUSIONS]
    rng.shuffle(idxs)
    for _ in range(rng.randint(1, max_edits)):
        if idxs and rng.random() < 0.9:
            i = idxs.pop()
            chars[i] = rng.choice(OCR_CONFUSIONS[chars[i]])
        elif len(chars) > 2:
            j = rng.randrange(len(chars))
            chars[j] = ""  # dropped stroke; keeps indices stable
            if j in idxs:
                idxs.remove(j)
    out = "".join(chars)
    return out if out != text else (text[:-1] if len(text) > 2 else text + "?")


def slip_corrupt(board: chess.Board, move: chess.Move, rng: random.Random) -> str | None:
    """A legal-but-wrong move a *player* might write instead: same piece or same target.

    Returns the SAN of the wrong move, or None if no plausible alternative exists.
    """
    piece = board.piece_at(move.from_square)
    near = [
        m
        for m in board.legal_moves
        if m != move
        and (
            m.to_square == move.to_square
            or (board.piece_at(m.from_square) == piece and chess.square_distance(m.to_square, move.to_square) <= 2)
        )
    ]
    if not near:
        return None
    return board.san(rng.choice(near))


def ocr_alternatives(written: str, true_text: str | None, rng: random.Random, k: int = 3) -> list[dict]:
    """Simulated OCR n-best list for a written token. The written text is always top-1.

    When the token was garbled by OCR, the true text is included with some probability,
    mimicking a recognizer whose second guess is right.
    """
    alts = {written}
    if true_text and true_text != written and rng.random() < 0.6:
        alts.add(true_text)
    tries = 0
    while len(alts) < k and tries < 10:
        alts.add(ocr_corrupt(written, rng, max_edits=1))
        tries += 1
    rest = [a for a in alts if a != written]
    rng.shuffle(rest)
    top = rng.uniform(0.45, 0.95)
    weights = [rng.random() for _ in rest]
    s = sum(weights) or 1.0
    out = [{"text": written, "p": round(top, 3)}]
    out += [{"text": a, "p": round((1 - top) * w / s, 3)} for a, w in zip(rest, weights)]
    return out
