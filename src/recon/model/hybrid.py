"""E19b hybrid: our scoresheet model (E16) fused with a pretrained Maia-3 human-move model.

    our tower   ReconModel (E16): board + notation + later written entries -> token states h,
                move logits, and the align / gap / promo / future-board heads
    Maia tower  Maia-3 (pretrained): last 8 boards + ratings -> 64 square states. After every
                Maia block, the squares cross-attend to our tower's tokens h (gated: tanh(g) with
                g = 0 at init, so at step 0 the tower is exactly Maia-3)
    move        ours + lam * Maia from-to scores (lam learned, init from E19a's tuned weight)

At initialisation the model equals the E19a sum (without the feasibility filter, which is
applied at suggestion time), so fine-tuning starts from our best zero-training result.
Promotions use our promo head only (Maia's promotion entries are not used).
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import torch
import torch.nn.functional as Fn
from torch import nn

from recon.model.maia import SPECS, planes_from_codes
from recon.model.net import ModelConfig, ReconModel


class GatedXAttn(nn.Module):
    """Squares attend to the scoresheet tokens; output added through tanh gates (closed at init)."""

    def __init__(self, d: int, d_mem: int, heads: int):
        super().__init__()
        self.nq, self.nkv = nn.LayerNorm(d), nn.LayerNorm(d_mem)
        self.kv = nn.Linear(d_mem, d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.ff = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.g_attn = nn.Parameter(torch.zeros(1))
        self.g_ff = nn.Parameter(torch.zeros(1))

    def forward(self, x, mem, pad):
        kv = self.kv(self.nkv(mem))
        a, _ = self.attn(self.nq(x), kv, kv, key_padding_mask=pad, need_weights=False)
        x = x + torch.tanh(self.g_attn) * a
        return x + torch.tanh(self.g_ff) * self.ff(x)


class HybridModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        from maia3.model_registry import MODEL_SPECS
        from maia3.models import MAIA3Model

        self.cfg = cfg
        self.ours = ReconModel(cfg)
        name = SPECS[cfg.maia][2]
        self.mcfg = SimpleNamespace(**next(s for s in MODEL_SPECS if s.name == name).config)
        self.maia = MAIA3Model(self.mcfg)
        dm = self.mcfg.dim_vit
        self.fuse = nn.ModuleList(GatedXAttn(dm, cfg.d_model, self.mcfg.num_heads) for _ in range(self.mcfg.num_blocks))
        self.lam = nn.Parameter(torch.tensor(float(cfg.lam_init)))

    # -- initialisation from pretrained parts ------------------------------------------------
    def load_pretrained(self, ours_ckpt: str | None = None) -> None:
        from huggingface_hub import hf_hub_download

        repo, fname, _ = SPECS[self.cfg.maia]
        sd = torch.load(hf_hub_download(repo, fname), map_location="cpu", weights_only=True)
        sd = sd.get("model_state_dict", sd)
        missing, unexpected = self.maia.load_state_dict({k.replace("smolgen", "gab"): v for k, v in sd.items()}, strict=False)
        assert not missing and not unexpected, (missing, unexpected)
        self.fresh: set[str] = set()  # parameters of our tower that the checkpoint does not have
        if ours_ckpt:
            ck = torch.load(ours_ckpt, map_location="cpu")
            # hist_moves may be added on top of a checkpoint trained without it (E20): its tokens start fresh
            skip = ("maia", "lam_init", "hist_moves")
            mine = {k: v for k, v in ck["model_cfg"].items() if k not in skip}
            want = {k: v for k, v in self.cfg.to_dict().items() if k not in skip}
            assert mine == want, f"our tower config differs from {ours_ckpt}: {mine} vs {want}"
            missing, unexpected = self.ours.load_state_dict(ck["model"], strict=False)
            assert not unexpected, unexpected
            assert all(k.startswith("h_") for k in missing), missing
            assert bool(missing) == bool(self.cfg.hist_moves and not ck["model_cfg"].get("hist_moves")), missing
            self.fresh = {"ours." + k for k in missing}

    def pretrained_names(self) -> set[str]:
        """Parameters that start from pretrained weights (trained with a smaller learning rate)."""
        fresh = getattr(self, "fresh", set())
        return {n for n, _ in self.named_parameters() if n.startswith(("ours.", "maia.")) and n not in fresh}

    # -- forward -------------------------------------------------------------------------------
    def maia_scores(self, batch, mem, pad):
        m = self.maia
        x = planes_from_codes(batch["maia_hist"])
        elo = batch["maia_elo"]
        se = m.interpolate_elo(elo[:, 0]).unsqueeze(1).expand(-1, 64, -1)
        oe = m.interpolate_elo(elo[:, 1]).unsqueeze(1).expand(-1, 64, -1)
        x = m.token_projection(torch.cat([x, se, oe], -1))
        if hasattr(m, "abs_pe"):
            x = m.abs_pe(x)
        for blk, fuse in zip(m.transformer.layers, self.fuse):
            x = fuse(blk(x), mem, pad)
        x = m.transformer.norm(x)
        f, t = m.proj_sq_from(x), m.proj_sq_to(x)
        return torch.einsum("bid,bjd->bij", f, t).reshape(x.shape[0], 4096) / math.sqrt(self.mcfg.head_hid_dim)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        out = self.ours(batch, return_hidden=True)
        h, pad = out.pop("_h"), out.pop("_pad")
        maia = self.maia_scores(batch, h, pad).float()
        lp = torch.log_softmax(maia.masked_fill(~batch["legal"], float("-inf")), -1).masked_fill(~batch["legal"], 0.0)
        out["move_ours"] = out["move"]
        out["move"] = out["move"].float() + self.lam * lp  # illegal moves stay -inf from our head
        return out
