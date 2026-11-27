"""Synthetic scoresheet generation: turn a complete game into noisy observed notation.

Every corruption keeps a full record of what happened, so the benchmark knows the
true move sequence *and* the true token-to-ply alignment.

Token kinds in the ground truth:
    match     token written correctly for its ply (possibly in a different style)
    ocr       token garbled by handwriting recognition
    slip      player wrote a different (usually legal) move than the one played
    inserted  extra token that corresponds to no ply (duplicate or stray)
Plies that have no token at all are listed in ``omitted_plies``.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, replace

from recon.game import Game
from recon.observe import notation


@dataclass(frozen=True)
class CorruptionConfig:
    name: str = "mixed"
    max_plies: int | None = 80  # truncate games to this many plies
    # time-pressure burst omission: one contiguous run of unrecorded plies
    p_burst: float = 0.6
    burst_min: int = 4
    burst_max: int = 40
    burst_late_bias: float = 2.0  # start weight ~ (ply+1)**bias: bursts happen late
    allow_trailing_gap: bool = False
    # independent per-ply / per-token errors
    p_drop: float = 0.01
    p_ocr: float = 0.03
    p_slip: float = 0.01
    p_insert: float = 0.005
    # layout
    p_hints: float = 0.8  # a source carries move-number/side hints at all
    p_renumber_after_gap: float = 0.5  # after a gap, writer keeps writing in the next row (shift)
    ocr_alternatives: bool = False
    num_sources: int = 1
    p_shared_burst: float = 0.7  # with 2 sources, both players miss the same scramble

    def to_dict(self) -> dict:
        return asdict(self)


_BASE = CorruptionConfig()
PRESETS: dict[str, CorruptionConfig] = {
    "clean": replace(_BASE, name="clean", p_burst=0, p_drop=0, p_ocr=0, p_slip=0, p_insert=0),
    "missing": replace(_BASE, name="missing", p_burst=1.0, p_ocr=0, p_slip=0, p_insert=0),
    "substitution": replace(_BASE, name="substitution", p_burst=0, p_drop=0, p_ocr=0.06, p_slip=0.02),
    "mixed": replace(_BASE, name="mixed"),
    "mixed_ocr": replace(_BASE, name="mixed_ocr", p_ocr=0.08, ocr_alternatives=True),
    "two_sheets": replace(_BASE, name="two_sheets", num_sources=2),
}


def _sample_burst(n: int, cfg: CorruptionConfig, rng: random.Random) -> range:
    if n < cfg.burst_min + 3 or rng.random() >= cfg.p_burst:
        return range(0)
    length = rng.randint(cfg.burst_min, cfg.burst_max)
    last_end = n if cfg.allow_trailing_gap else n - 2  # leave >=2 observed plies after the gap
    starts = list(range(1, max(2, last_end - cfg.burst_min + 1)))
    weights = [(s + 1) ** cfg.burst_late_bias for s in starts]
    start = rng.choices(starts, weights)[0]
    return range(start, min(start + length, last_end))


def _jitter(burst: range, n: int, rng: random.Random) -> range:
    if not burst:
        return burst
    a = max(1, burst.start + rng.randint(-2, 2))
    b = min(n - 2, burst.stop + rng.randint(-2, 2))
    return range(a, b) if b > a else burst


def _make_source(
    game: Game, cfg: CorruptionConfig, burst: range, rng: random.Random
) -> tuple[list[dict], list[int | None], list[str], list[int]]:
    n = min(game.num_plies, cfg.max_plies or game.num_plies)
    # Without a trailing gap the last two plies are always written, so the game's end is observed.
    droppable = n if cfg.allow_trailing_gap else n - 2
    omitted = set(burst) | {i for i in range(droppable) if rng.random() < cfg.p_drop}
    hints = rng.random() < cfg.p_hints

    tokens: list[dict] = []
    token_ply: list[int | None] = []
    token_kind: list[str] = []
    row = 0  # the scoresheet slot the next token is written into
    after_gap = False

    def emit(text: str, ply: int | None, kind: str, true_text: str | None) -> None:
        nonlocal row
        tok = {
            "text": text,
            "move_number_hint": row // 2 + 1 if hints else None,
            "side_hint": ("white" if row % 2 == 0 else "black") if hints else None,
            "confidence": None,
            "alternatives": None,
        }
        if cfg.ocr_alternatives:
            tok["alternatives"] = notation.ocr_alternatives(text, true_text, rng)
            tok["confidence"] = tok["alternatives"][0]["p"]
        tokens.append(tok)
        token_ply.append(ply)
        token_kind.append(kind)
        row += 1

    for i, (board, move) in enumerate(game.replay(n)):
        if i in omitted:
            after_gap = True
            continue
        if after_gap and rng.random() >= cfg.p_renumber_after_gap:
            row = i  # writer jumped to the correct row
        after_gap = False

        written = notation.style(board.san(move), rng)
        kind, true_text = "match", written
        r = rng.random()
        if r < cfg.p_ocr:
            written, kind = notation.ocr_corrupt(written, rng), "ocr"
        elif r < cfg.p_ocr + cfg.p_slip:
            wrong = notation.slip_corrupt(board, move, rng)
            if wrong is not None:
                written, kind, true_text = notation.style(wrong, rng), "slip", None
        emit(written, i, kind, true_text if kind == "ocr" else None)

        if rng.random() < cfg.p_insert:
            emit(written, None, "inserted", None)  # duplicated entry

    omitted_sorted = sorted(omitted)
    return tokens, token_ply, token_kind, omitted_sorted


def corrupt(game: Game, cfg: CorruptionConfig, rng: random.Random) -> tuple[list[dict], dict]:
    """Return ``(sources, truth)`` for one corrupted copy of ``game``."""
    n = min(game.num_plies, cfg.max_plies or game.num_plies)
    burst = _sample_burst(n, cfg, rng)
    sources, truth_sources = [], []
    for s in range(cfg.num_sources):
        b = burst
        if s > 0:
            b = _jitter(burst, n, rng) if rng.random() < cfg.p_shared_burst else _sample_burst(n, cfg, rng)
        tokens, token_ply, token_kind, omitted = _make_source(game, cfg, b, rng)
        sources.append({"type": "scoresheet", "tokens": tokens})
        truth_sources.append({"token_ply": token_ply, "token_kind": token_kind, "omitted_plies": omitted})
    truth = {"moves_uci": game.moves[:n], "sources": truth_sources}
    return sources, truth
