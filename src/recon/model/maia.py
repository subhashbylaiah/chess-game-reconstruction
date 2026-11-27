"""Maia-3 (CSSLab, ICLR 2026; AGPL-3.0) as a pretrained human-move prior.

Maia-3 reads the side-to-move's view of the board (mirrored when Black moves) for the current
position and up to ``history - 1`` earlier ones, plus both players' ratings, and scores
64x64 from-to pairs (+256 promotion entries). Encoding follows maia3.uci (its official
inference path): each history board is tokenized from its own side to move, oldest first,
padded with the earliest board.
"""
from __future__ import annotations

from types import SimpleNamespace

import chess
import torch

SPECS = {"5m": ("UofTCSSLab/Maia3-5M", "maia3-5m.pt", "maia3-5m"),
         "23m": ("UofTCSSLab/Maia3-23M", "maia3-23m.pt", "maia3-23m"),
         "79m": ("UofTCSSLab/Maia3-79M", "maia3-79m.pt", "maia3-79m")}
DEFAULT_ELO = 1500


def load_maia(size: str = "5m", device: str = "cpu"):
    from huggingface_hub import hf_hub_download
    from maia3.model_registry import MODEL_SPECS
    from maia3.models import MAIA3Model

    repo, fname, name = SPECS[size]
    spec = next(s for s in MODEL_SPECS if s.name == name)
    cfg = SimpleNamespace(**spec.config)
    model = MAIA3Model(cfg)
    sd = torch.load(hf_hub_download(repo, fname), map_location="cpu", weights_only=True)
    sd = sd.get("model_state_dict", sd)
    missing, unexpected = model.load_state_dict({k.replace("smolgen", "gab"): v for k, v in sd.items()}, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    return model.to(device).eval(), cfg


def _mirror_sq(sq: int) -> int:
    return chess.square_mirror(sq)


def board_planes(board: chess.Board) -> torch.Tensor:
    from maia3.dataset import tokenize_board
    return tokenize_board(board)


def encode(boards: list[chess.Board], cfg) -> torch.Tensor:
    """(64, 12*history) input from the position history, oldest first, current last."""
    hist = [board_planes(b) for b in boards[-cfg.history:]]
    if len(hist) < cfg.history:
        hist = [hist[0]] * (cfg.history - len(hist)) + hist
    return torch.cat(hist, dim=1)


def move_index(move: chess.Move, turn: bool) -> int:
    """Index of a (real-board) move in Maia-3's 4352 logits."""
    f, t = move.from_square, move.to_square
    if turn == chess.BLACK:
        f, t = _mirror_sq(f), _mirror_sq(t)
    if move.promotion:
        k = "qrbn".index(chess.piece_symbol(move.promotion))
        return 4096 + (chess.square_file(f) * 8 + chess.square_file(t)) * 4 + k
    return f * 64 + t


@torch.no_grad()
def log_probs(model, cfg, histories: list[list[chess.Board]], elos: list[tuple[int | None, int | None]],
              device: str = "cpu", batch_size: int = 512) -> list[dict[chess.Move, float]]:
    """Log-probabilities over legal moves for each position (history ends with the current board).

    ``elos`` is (side to move, opponent); missing ratings default to 1500.
    """
    out = []
    for lo in range(0, len(histories), batch_size):
        hs = histories[lo:lo + batch_size]
        x = torch.stack([encode(h, cfg) for h in hs]).to(device)
        es = elos[lo:lo + batch_size]
        se = torch.tensor([e[0] or DEFAULT_ELO for e in es], device=device)
        oe = torch.tensor([e[1] or DEFAULT_ELO for e in es], device=device)
        logits = model(x, se, oe)[0].float().cpu()
        for h, lg in zip(hs, logits):
            b = h[-1]
            legal = list(b.legal_moves)
            idx = torch.tensor([move_index(m, b.turn) for m in legal])
            lp = torch.log_softmax(lg[idx], -1).tolist()
            out.append(dict(zip(legal, lp)))
    return out


# ---- compact inputs for training / the hybrid model -------------------------------------------

MAIA_HIST = 8


def hist_codes(raws: list, turns: list[bool], H: int = MAIA_HIST):
    """(H, 64) uint8 piece codes in Maia's frame for the last H positions (oldest first, current last).

    ``raws`` are white-frame codes (features.raw_squares) and ``turns`` the side to move in each.
    Each board is mirrored to its own side to move, as maia3.dataset.tokenize_board does. Short
    histories are padded with the earliest board (Maia's inference convention).
    """
    import numpy as np
    from recon.model.features import orient_raw

    codes = [orient_raw(r, t) for r, t in zip(raws[-H:], turns[-H:])]
    codes = [codes[0]] * (H - len(codes)) + codes
    return np.stack(codes).astype(np.uint8)


def planes_from_codes(codes: torch.Tensor) -> torch.Tensor:
    """[B, H, 64] codes (0 empty, 1-12 pieces) -> Maia tokens [B, 64, 12*H]."""
    B, H, _ = codes.shape
    oh = torch.nn.functional.one_hot(codes.long(), 13)[..., 1:].float()  # [B,H,64,12]
    return oh.permute(0, 2, 1, 3).reshape(B, 64, H * 12)


def elo_pair(white: int | None, black: int | None, turn: bool):
    import numpy as np
    w, b = white or DEFAULT_ELO, black or DEFAULT_ELO
    return np.array([w, b] if turn == chess.WHITE else [b, w], np.int64)


def board_inputs(board: chess.Board, ctx):
    """Maia inputs for a board whose move stack holds the recent history (as search states keep)."""
    from recon.model.features import raw_squares

    b = board.copy()
    raws, turns = [raw_squares(b)], [b.turn]
    for _ in range(MAIA_HIST - 1):
        if not b.move_stack:
            break
        b.pop()
        raws.insert(0, raw_squares(b)); turns.insert(0, b.turn)
    return hist_codes(raws, turns), elo_pair(ctx.white_elo, ctx.black_elo, board.turn)
