"""Train the notation-aware move model on the Mac GPU (MPS), CUDA, or CPU."""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from recon.model.dataset import DataConfig, PositionStream, fixed_positions
from recon.model.net import ModelConfig, ReconModel, build_model, losses, num_params


@dataclass
class TrainConfig:
    out_dir: str = "runs/v1"
    steps: int = 20000
    batch_size: int = 256
    lr: float = 5e-4
    weight_decay: float = 0.05
    warmup: int = 500
    eval_every: int = 1000
    log_every: int = 100
    workers: int = 8
    amp: bool = False
    val_games: int | None = 300
    device: str = "auto"
    resume: bool = True  # continue from out_dir/state.pt if it exists
    ckpt_every: int = 2000  # steps between resumable checkpoints
    compile: bool = False  # torch.compile the model (fuses small GPU ops; same maths)
    init_from: str = ""  # hybrid: our-tower checkpoint to start from (e.g. E16's best.pt)
    pre_lr_mult: float = 0.1  # hybrid: learning-rate multiplier for pretrained parts (ours, Maia)
    init_full: str = ""  # hybrid: start from a whole trained hybrid checkpoint (fine-tuning, e.g. E19c -> E31)
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model: ReconModel, val: dict[str, torch.Tensor], device, batch_size=1024, amp=False) -> dict:
    model.eval()
    n = val["move"].shape[0]
    acc = {k: 0.0 for k in ("move@1", "move@5", "hid@1", "hid@2", "hid@3", "hid@5", "wr@1", "align", "gap", "gap±2", "loss")}
    n_hid = int(val["hidden"].sum())
    for i in range(0, n, batch_size):
        b = to_device({k: v[i : i + batch_size] for k, v in val.items()}, device)
        with torch.autocast(device.type, dtype=amp_dtype(device), enabled=amp):
            out = model(b)
        loss, _ = losses({k: v.float() for k, v in out.items()}, b)
        acc["loss"] += loss.item() * b["move"].shape[0]
        top5 = out["move"].float().topk(5, -1).indices
        hit1 = top5[:, 0] == b["move"]
        hit5 = (top5 == b["move"][:, None]).any(-1)
        hit2 = (top5[:, :2] == b["move"][:, None]).any(-1)
        hit3 = (top5[:, :3] == b["move"][:, None]).any(-1)
        h = b["hidden"]
        acc["move@1"] += hit1.sum().item()
        acc["move@5"] += hit5.sum().item()
        acc["hid@1"] += hit1[h].sum().item()
        acc["hid@2"] += hit2[h].sum().item()
        acc["hid@3"] += hit3[h].sum().item()
        acc["hid@5"] += hit5[h].sum().item()
        acc["wr@1"] += hit1[~h].sum().item()
        acc["align"] += (out["align"].argmax(-1) == b["align"]).sum().item()
        g = out["gap"].argmax(-1)
        acc["gap"] += (g == b["gap"]).sum().item()
        acc["gap±2"] += ((g - b["gap"]).abs() <= 2).sum().item()
    model.train()
    res = {k: v / n for k, v in acc.items()}
    res["hid@1"] = acc["hid@1"] / max(n_hid, 1)
    for k in ("hid@2", "hid@3", "hid@5"):
        res[k] = acc[k] / max(n_hid, 1)
    res["wr@1"] = acc["wr@1"] / max(n - n_hid, 1)
    res["n"], res["n_hidden"] = n, n_hid
    return res


def amp_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16  # no loss scaling needed
    return torch.float16


def _atomic_save(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)  # never leaves a half-written checkpoint if the machine dies mid-save


def save(model: ReconModel, cfg: TrainConfig, path: Path, step: int) -> None:
    _atomic_save({"model": model.state_dict(), "model_cfg": cfg.model.to_dict(), "step": step}, path)


def save_state(model, opt, cfg: TrainConfig, path: Path, step: int, best: float) -> None:
    """Everything needed to continue training after an interruption."""
    _atomic_save(
        {"model": model.state_dict(), "opt": opt.state_dict(), "model_cfg": cfg.model.to_dict(),
         "step": step, "best": best, "torch_rng": torch.get_rng_state()},
        path,
    )


def load(path: str | Path, device: str = "auto") -> ReconModel:
    ckpt = torch.load(path, map_location="cpu")
    model = build_model(ModelConfig(**ckpt["model_cfg"]))
    model.load_state_dict(ckpt["model"])
    return model.to(pick_device(device)).eval()


def train(cfg: TrainConfig) -> None:
    device = pick_device(cfg.device)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(asdict(cfg), indent=2))

    torch.manual_seed(cfg.data.seed)
    model = build_model(cfg.model)
    pre = set()
    if cfg.model.maia:
        model.load_pretrained(cfg.init_from or None)
        pre = model.pretrained_names()
        if cfg.init_full:
            ck = torch.load(cfg.init_full, map_location="cpu")
            assert ck["model_cfg"] == cfg.model.to_dict(), f"model config differs from {cfg.init_full}"
            model.load_state_dict(ck["model"])
            print(f"initialised the whole model from {cfg.init_full}", flush=True)
    model = model.to(device)
    print(f"device={device} params={num_params(model) / 1e6:.2f}M", flush=True)
    fwd = torch.compile(model) if cfg.compile else model  # checkpoints still save the uncompiled module

    val_cfg = DataConfig(**{**asdict(cfg.data), "split": "val", "seed": 12345})
    val = fixed_positions(val_cfg, cfg.val_games)
    print(f"val positions={val['move'].shape[0]} (hidden={int(val['hidden'].sum())})", flush=True)

    groups = []
    for is_pre in (False, True):
        named = [(n, p) for n, p in model.named_parameters() if (n in pre) == is_pre]
        scale = cfg.pre_lr_mult if is_pre else 1.0
        dec = [p for n, p in named if p.dim() >= 2 and "emb" not in n]
        nodec = [p for n, p in named if not (p.dim() >= 2 and "emb" not in n)]
        groups += [{"params": dec, "weight_decay": cfg.weight_decay, "scale": scale},
                   {"params": nodec, "weight_decay": 0.0, "scale": scale}]
    opt = torch.optim.AdamW([g for g in groups if g["params"]], lr=cfg.lr, betas=(0.9, 0.95))

    start, best = 0, -1.0
    state_path = out / "state.pt"
    if cfg.resume and state_path.exists():
        st = torch.load(state_path, map_location="cpu")
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        torch.set_rng_state(st["torch_rng"])
        start, best = st["step"], st["best"]
        print(f"resumed from step {start} (best val hidden move@1={best:.4f})", flush=True)
    # a resumed run draws fresh data instead of replaying the start of the stream
    data_cfg = cfg.data if start == 0 else replace(cfg.data, seed=cfg.data.seed + start)

    loader = DataLoader(
        PositionStream(data_cfg),
        batch_size=cfg.batch_size,
        num_workers=cfg.workers,
        persistent_workers=cfg.workers > 0,
        prefetch_factor=4 if cfg.workers > 0 else None,
    )
    def lr_at(step: int) -> float:
        if step < cfg.warmup:
            return cfg.lr * (step + 1) / cfg.warmup
        p = (step - cfg.warmup) / max(1, cfg.steps - cfg.warmup)
        return cfg.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * p)))

    log = (out / "log.jsonl").open("a")
    t0, seen = time.time(), 0
    running: dict[str, float] = {}
    if start >= cfg.steps:
        print("already finished", flush=True)
        return
    for step, batch in enumerate(loader, start=start):
        if step >= cfg.steps:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(step) * g.get("scale", 1.0)
        batch = to_device(batch, device)
        with torch.autocast(device.type, dtype=amp_dtype(device), enabled=cfg.amp):
            outp = fwd(batch)
        loss, parts = losses({k: v.float() for k, v in outp.items()}, batch)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        seen += batch["move"].shape[0]
        for k, v in parts.items():
            running[k] = running.get(k, 0.0) + v

        if (step + 1) % cfg.log_every == 0:
            dt = time.time() - t0
            avg = {k: v / cfg.log_every for k, v in running.items()}
            running = {}
            msg = f"step {step + 1} " + " ".join(f"{k}={v:.3f}" for k, v in avg.items())
            print(f"{msg} lr={lr_at(step):.2e} {seen / dt:.0f} pos/s", flush=True)
            log.write(json.dumps({"step": step + 1, "train": avg, "pos_per_s": seen / dt}) + "\n")
            log.flush()

        if (step + 1) % cfg.eval_every == 0 or step + 1 == cfg.steps:
            res = evaluate(model, val, device, amp=cfg.amp)
            print("  val " + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in res.items()), flush=True)
            log.write(json.dumps({"step": step + 1, "val": res}) + "\n")
            log.flush()
            save(model, cfg, out / "last.pt", step + 1)
            if res["hid@1"] > best:
                best = res["hid@1"]
                save(model, cfg, out / "best.pt", step + 1)
        if (step + 1) % cfg.ckpt_every == 0 or (step + 1) % cfg.eval_every == 0 or step + 1 == cfg.steps:
            save_state(model, opt, cfg, state_path, step + 1, best)
    log.close()
    print(f"done: {seen} positions in {time.time() - t0:.0f}s; best val hidden move@1={best:.4f}")
