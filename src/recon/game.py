"""Ground-truth game records.

The latent game is always stored as UCI moves plus metadata. Observed notation
(SAN as a human wrote it) lives in :mod:`recon.observe` and is never mixed in here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import chess


@dataclass(frozen=True)
class TimeControl:
    initial_seconds: int | None = None
    increment_seconds: int | None = None

    @classmethod
    def parse(cls, text: str | None) -> TimeControl:
        """Parse a PGN TimeControl tag such as ``"300+3"``; ``"-"`` or junk gives unknowns."""
        if not text or "+" not in text:
            return cls()
        base, _, inc = text.partition("+")
        try:
            return cls(int(base), int(inc))
        except ValueError:
            return cls()

    @property
    def speed(self) -> str:
        """Lichess speed class from estimated duration (base + 40 * increment)."""
        if self.initial_seconds is None:
            return "unknown"
        est = self.initial_seconds + 40 * (self.increment_seconds or 0)
        if est < 180:
            return "bullet"
        if est < 480:
            return "blitz"
        if est < 1500:
            return "rapid"
        return "classical"

    def to_dict(self) -> dict:
        return {"initial_seconds": self.initial_seconds, "increment_seconds": self.increment_seconds}


@dataclass
class Game:
    id: str
    moves: list[str]  # UCI, one per ply
    white_elo: int | None = None
    black_elo: int | None = None
    time_control: TimeControl = field(default_factory=TimeControl)
    clocks: list[float | None] = field(default_factory=list)  # seconds left after each ply
    result: str = "*"
    termination: str = ""
    date: str = ""
    start_fen: str = chess.STARTING_FEN

    @property
    def num_plies(self) -> int:
        return len(self.moves)

    @property
    def mean_elo(self) -> float | None:
        elos = [e for e in (self.white_elo, self.black_elo) if e is not None]
        return sum(elos) / len(elos) if elos else None

    def replay(self, max_plies: int | None = None):
        """Yield ``(board_before, move)`` for each ply. ``board_before`` is a copy."""
        board = chess.Board(self.start_fen)
        for uci in self.moves[:max_plies]:
            move = chess.Move.from_uci(uci)
            yield board.copy(stack=False), move
            board.push(move)

    def sans(self, max_plies: int | None = None) -> list[str]:
        return [b.san(m) for b, m in self.replay(max_plies)]


def rating_band(elo: float | None) -> str:
    if elo is None:
        return "unrated"
    if elo < 1200:
        return "<1200"
    if elo < 1600:
        return "1200-1599"
    if elo < 2000:
        return "1600-1999"
    return "2000+"
