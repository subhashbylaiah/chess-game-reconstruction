"""Notation-aware human move model.

Sequence = [CLS] + 64 square tokens + N_CTX context tokens + K notation tokens,
run through a plain pre-norm transformer encoder.

Heads:
    move   bilinear from-square x to-square scores over the 64 square outputs -> [B, 4096]
    promo  4-way promotion piece from CLS (used only for promoting pawn moves)
    align  pointer from CLS over the K notation outputs, plus a learned "not written" slot -> [B, K+1]
    gap    plies until the next written move, classification over 0..MAX_GAP -> [B, MAX_GAP+1]
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as Fn
from torch import nn

from recon.model import features as F


@dataclass
class ModelConfig:
    d_model: int = 256
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 1024
    dropout: float = 0.0
    struct_notes: bool = False  # add parsed-entry fields to notation tokens and link to-squares
    gap_input: bool = False  # extra context token: known plies until the next written move
    move_number: bool = False  # extra context token: full-move number (game phase)
    hist_moves: int = 0  # tokens for the last N moves played (0 = only the last-move square flags)
    future: int = 0  # tokens for up to N written entries after this ply, with exact ply offsets (0 = off)
    future_boards: int = 0  # auxiliary head: predict the board before each of the next N written plies
    maia: str = ""  # hybrid: Maia-3 size ("5m", "23m") fused with this model (see hybrid.py); "" = off
    lam_init: float = 1.5  # hybrid: initial weight of Maia's move logits
    whole_notes: bool = False  # ablation: each written entry is one whole token (hashed text), no character sharing

    def to_dict(self) -> dict:
        return asdict(self)


N_WHOLE = 1 << 15  # hash buckets for whole-entry tokens (ablation)


class NotationEncoder(nn.Module):
    """One vector per written token: sum of (char + char-position) embeddings -> MLP.

    With ``whole=True`` (ablation) the text is instead one whole token: its exact character string is hashed
    to one of N_WHOLE learned embeddings, so e.g. "Rxe8+" and a misread "Rxc8+" share nothing.
    """

    def __init__(self, d: int, whole: bool = False):
        super().__init__()
        self.whole = nn.Embedding(N_WHOLE, d) if whole else None
        self.char = nn.Embedding(F.N_CHARS, d, padding_idx=0)
        self.char_pos = nn.Parameter(torch.randn(F.L, d) * 0.02)
        self.hint = nn.Embedding(F.N_HINT, d)
        self.conf = nn.Embedding(F.N_CONF, d)
        self.slot = nn.Parameter(torch.randn(F.K, d) * 0.02)
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))

    def text(self, chars):
        """[..., L] character ids -> [..., d] text vector (before the MLP)."""
        if self.whole is not None:
            powers = 37 ** torch.arange(chars.shape[-1], device=chars.device, dtype=torch.int64)
            idx = (chars.long() * powers).sum(-1) % N_WHOLE  # exact string -> bucket (char ids < 37)
            return self.whole(idx) * (chars > 0).any(-1, keepdim=True).float()
        present = (chars > 0).unsqueeze(-1).float()  # [...,L,1]
        return ((self.char(chars) + self.char_pos) * present).sum(-2)

    def forward(self, chars, feats):
        c = self.text(chars)  # [B,K,d]
        return self.mlp(c) + self.hint(feats[..., 0]) + self.conf(feats[..., 1]) + self.slot


class ReconModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.piece = nn.Embedding(13, d)
        self.last = nn.Embedding(3, d)
        self.square_pos = nn.Parameter(torch.randn(64, d) * 0.02)
        self.ctx = nn.ModuleList(nn.Embedding(n, d) for n in F.CTX_SIZES)
        self.ctx_pos = nn.Parameter(torch.randn(F.N_CTX, d) * 0.02)
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        # cls, square, context, notation[, history][, future]; sized as before when unused so old checkpoints load
        self.type_emb = nn.Embedding(6 if cfg.future else (5 if cfg.hist_moves else 4), d)
        self.notes = NotationEncoder(d, whole=cfg.whole_notes)
        n_ctx = F.N_CTX
        if cfg.struct_notes:
            self.s_piece = nn.Embedding(F.N_PIECE_F, d)
            self.s_cap = nn.Embedding(2, d)
            self.s_ff = nn.Embedding(F.N_FILE_F, d)
            self.s_fr = nn.Embedding(F.N_RANK_F, d)
            self.s_promo = nn.Embedding(F.N_PROMO_F, d)
            self.s_has_to = nn.Embedding(2, d)
            # added to a square token when upcoming entry k names it as destination
            self.dest_slot = nn.Parameter(torch.randn(F.K, d) * 0.02)
        if cfg.gap_input:
            self.gap_in = nn.Embedding(F.N_GAP_IN, d)
            n_ctx += 1
        if cfg.move_number:
            self.move_num = nn.Embedding(F.N_MOVENUM, d)
            n_ctx += 1
        self.n_ctx = n_ctx
        H = cfg.hist_moves
        assert 0 <= H <= F.HIST
        if H:
            self.h_piece = nn.Embedding(13, d)
            self.h_cap = nn.Embedding(2, d)
            self.h_from = nn.Parameter(torch.randn(d) * 0.02)  # marks square_pos as origin
            self.h_to = nn.Parameter(torch.randn(d) * 0.02)  # marks square_pos as destination
            self.h_slot = nn.Parameter(torch.randn(H, d) * 0.02)  # how many plies ago
        NFt = cfg.future
        assert 0 <= NFt <= F.NF and 0 <= cfg.future_boards <= F.NB
        if NFt:
            # characters share the notation encoder; parsed fields and offsets get their own embeddings
            self.f_piece = nn.Embedding(F.N_PIECE_F, d)
            self.f_cap = nn.Embedding(2, d)
            self.f_ff = nn.Embedding(F.N_FILE_F, d)
            self.f_fr = nn.Embedding(F.N_RANK_F, d)
            self.f_promo = nn.Embedding(F.N_PROMO_F, d)
            self.f_has_to = nn.Embedding(2, d)
            self.f_off = nn.Embedding(F.N_OFF, d)
            self.f_conf = nn.Embedding(F.N_CONF, d)
        if cfg.future_boards:
            self.fut_board = nn.Linear(d, 13 * cfg.future_boards)
        layer = nn.TransformerEncoderLayer(
            d, cfg.n_heads, cfg.d_ff, cfg.dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, cfg.n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.q_from = nn.Linear(d, d)
        self.k_to = nn.Linear(d, d)
        self.promo = nn.Linear(d, 5)
        self.align_q = nn.Linear(d, d)
        self.align_k = nn.Linear(d, d)
        self.not_written = nn.Parameter(torch.zeros(d))
        self.gap = nn.Linear(d, F.MAX_GAP + 1)
        n_seq = 1 + 64 + n_ctx + F.K + H + NFt
        types = torch.tensor([0] + [1] * 64 + [2] * n_ctx + [3] * F.K + [4] * H + [5] * NFt)
        self.register_buffer("types", types, persistent=False)
        self.n_seq = n_seq

    def forward(self, batch: dict[str, torch.Tensor], return_hidden: bool = False) -> dict[str, torch.Tensor]:
        B = batch["squares"].shape[0]
        sq = self.piece(batch["squares"]) + self.last(batch["last_move"]) + self.square_pos
        ctx = torch.stack([emb(batch["context"][:, i]) for i, emb in enumerate(self.ctx)], 1) + self.ctx_pos
        notes = self.notes(batch["note_chars"], batch["note_feats"])
        if self.cfg.struct_notes:
            st = batch["note_struct"]  # [B,K,6]
            to = st[..., 1]
            has_to = to > 0
            to_vec = self.square_pos[(to - 1).clamp(min=0)] * has_to.unsqueeze(-1)  # shared square identity
            notes = notes + (
                self.s_piece(st[..., 0]) + to_vec + self.s_has_to(has_to.long()) + self.s_cap(st[..., 2])
                + self.s_ff(st[..., 3]) + self.s_fr(st[..., 4]) + self.s_promo(st[..., 5])
            )
            onehot = Fn.one_hot(to, 65)[..., 1:].float() * batch["note_mask"].unsqueeze(-1)  # [B,K,64]
            sq = sq + torch.einsum("bks,kd->bsd", onehot, self.dest_slot)
        if self.cfg.gap_input:
            ctx = torch.cat([ctx, self.gap_in(batch["gap_in"]).unsqueeze(1)], 1)
        if self.cfg.move_number:
            ctx = torch.cat([ctx, self.move_num(batch["move_num"]).unsqueeze(1)], 1)
        parts = [self.cls.expand(B, -1, -1), sq, ctx, notes]
        pads = [torch.zeros(B, 1 + 64 + self.n_ctx, dtype=torch.bool, device=sq.device), ~batch["note_mask"]]
        H = self.cfg.hist_moves
        if H:
            hs = batch["hist"][:, :H]  # [B,H,4]
            present = hs[..., 0] > 0
            sq_of = lambda col: self.square_pos[(hs[..., col] - 1).clamp(min=0)]  # noqa: E731
            hist = (
                self.h_piece(hs[..., 0]) + sq_of(1) + self.h_from + sq_of(2) + self.h_to
                + self.h_cap(hs[..., 3]) + self.h_slot
            ) * present.unsqueeze(-1)
            parts.append(hist)
            pads.append(~present)
        NFt = self.cfg.future
        if NFt:
            fc, fs, ff = batch["fut_chars"][:, :NFt], batch["fut_struct"][:, :NFt], batch["fut_feats"][:, :NFt]
            fm = batch["fut_mask"][:, :NFt]
            c = self.notes.text(fc)
            to = fs[..., 1]
            has_to = to > 0
            fut = (
                self.notes.mlp(c) + self.f_piece(fs[..., 0]) + self.square_pos[(to - 1).clamp(min=0)] * has_to.unsqueeze(-1)
                + self.f_has_to(has_to.long()) + self.f_cap(fs[..., 2]) + self.f_ff(fs[..., 3]) + self.f_fr(fs[..., 4])
                + self.f_promo(fs[..., 5]) + self.f_off(ff[..., 0]) + self.f_conf(ff[..., 1])
            ) * fm.unsqueeze(-1)
            parts.append(fut)
            pads.append(~fm)
        x = torch.cat(parts, 1) + self.type_emb(self.types)
        pad = torch.cat(pads, 1)
        h = self.norm(self.encoder(x, src_key_padding_mask=pad))
        n0 = 1 + 64 + self.n_ctx
        cls, sqh, noteh = h[:, 0], h[:, 1:65], h[:, n0 : n0 + F.K]
        d = h.shape[-1]

        move = torch.einsum("bfd,btd->bft", self.q_from(sqh), self.k_to(sqh)).reshape(B, 4096) / math.sqrt(d)
        move = move.masked_fill(~batch["legal"], float("-inf"))

        q = self.align_q(cls)
        keys = torch.cat([self.align_k(noteh), self.not_written.expand(B, 1, -1)], 1)  # [B,K+1,d]
        align = torch.einsum("bd,bkd->bk", q, keys) / math.sqrt(d)
        align_mask = torch.cat([batch["note_mask"], torch.ones(B, 1, dtype=torch.bool, device=x.device)], 1)
        align = align.masked_fill(~align_mask, float("-inf"))

        out = {"move": move, "promo": self.promo(cls), "align": align, "gap": self.gap(cls)}
        if self.cfg.future_boards:
            out["fut_board"] = self.fut_board(sqh).reshape(B, 64, self.cfg.future_boards, 13)
        if return_hidden:  # for the hybrid model: every token's final state and the padding mask
            out["_h"], out["_pad"] = h, pad
        return out


def losses(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], w_align=1.0, w_gap=0.1, w_promo=0.1, w_fb=0.2):
    l_move = Fn.cross_entropy(out["move"], batch["move"])
    l_align = Fn.cross_entropy(out["align"], batch["align"])
    l_gap = Fn.cross_entropy(out["gap"], batch["gap"])
    is_promo = batch["promo"] > 0
    l_promo = Fn.cross_entropy(out["promo"][is_promo], batch["promo"][is_promo]) if is_promo.any() else out["promo"].sum() * 0
    total = l_move + w_align * l_align + w_gap * l_gap + w_promo * l_promo
    parts = {"move": l_move.item(), "align": l_align.item(), "gap": l_gap.item(), "promo": l_promo.item()}
    if "fut_board" in out and "fut_board" in batch:
        J = out["fut_board"].shape[2]
        logits = out["fut_board"].permute(0, 2, 1, 3)  # [B,J,64,13]
        target = batch["fut_board"][:, :J]  # [B,J,64]
        m = batch["fut_board_mask"][:, :J]
        if m.any():
            l_fb = Fn.cross_entropy(logits[m].reshape(-1, 13), target[m].reshape(-1))
            total = total + w_fb * l_fb
            parts["fb"] = l_fb.item()
    return total, parts


def build_model(cfg: ModelConfig) -> nn.Module:
    if cfg.maia:
        from recon.model.hybrid import HybridModel
        return HybridModel(cfg)
    return ReconModel(cfg)


def num_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
