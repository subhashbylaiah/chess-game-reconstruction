"""Command line: fetch games, build a benchmark, run a solver, evaluate predictions.

    recon fetch --source 2026-09 --limit 20000 --out data/games.parquet
    recon build-bench --games data/games.parquet --out bench/test.jsonl --num-games 200
    recon run --solver naive --bench bench/test.jsonl --out preds/naive.jsonl
    recon eval --bench bench/test.jsonl --preds preds/naive.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from recon.bench import build, metrics
from recon.data import lichess, store
from recon.observe.corrupt import PRESETS
from recon.solvers import base


def _load_solvers() -> None:
    import recon.solvers.beam  # noqa: F401  (registers)
    import recon.solvers.naive  # noqa: F401


def cmd_fetch(a: argparse.Namespace) -> None:
    flt = lichess.GameFilter(
        min_plies=a.min_plies,
        require_clocks=a.require_clocks,
        speeds=tuple(a.speeds.split(",")) if a.speeds else None,
    )
    t0 = time.time()

    def progress(games):
        for i, g in enumerate(games, 1):
            if i % 5000 == 0:
                print(f"  {i} games ({i / (time.time() - t0):.0f}/s)", file=sys.stderr)
            yield g

    n = store.write_games(progress(lichess.stream_games(a.source, a.limit, flt)), a.out)
    print(f"wrote {n} games to {a.out} in {time.time() - t0:.0f}s")


def cmd_build(a: argparse.Namespace) -> None:
    presets = list(PRESETS) if a.presets == "all" else a.presets.split(",")
    games = list(store.read_games(a.games, split=a.split))
    rows = build.build_instances(games, presets, a.num_games, seed=a.seed, split=a.split)
    n = build.write_jsonl(rows, a.out)
    print(f"wrote {n} instances ({len(presets)} presets x {min(a.num_games, len(games))} {a.split} games) to {a.out}")


def cmd_run(a: argparse.Namespace) -> None:
    _load_solvers()
    kw = dict(o.split("=", 1) for o in a.opt)
    if a.ckpt:
        kw["ckpt"] = a.ckpt
    solver = base.SOLVERS[a.solver](**kw)
    instances = list(build.read_jsonl(a.bench))
    if a.presets:
        instances = [i for i in instances if i["preset"] in a.presets.split(",")]
    if a.limit:
        instances = instances[: a.limit]
    n = build.write_jsonl(base.run(solver, instances), a.out)
    print(f"wrote {n} predictions to {a.out}")


def cmd_eval(a: argparse.Namespace) -> None:
    instances = list(build.read_jsonl(a.bench))
    preds = {p["id"]: p for p in build.read_jsonl(a.preds)}
    if a.only_predicted:
        instances = [i for i in instances if i["id"] in preds]
    report = metrics.evaluate(instances, preds)
    if a.json:
        print(json.dumps(report, indent=2))
    else:
        print(metrics.format_report(report))


def cmd_train(a: argparse.Namespace) -> None:
    from recon.model.dataset import DataConfig
    from recon.model.net import ModelConfig
    from recon.model.train import TrainConfig, train

    cfg = TrainConfig(
        out_dir=a.out,
        steps=a.steps,
        batch_size=a.batch_size,
        lr=a.lr,
        eval_every=a.eval_every,
        log_every=a.log_every,
        workers=a.workers,
        amp=a.amp,
        val_games=a.val_games,
        device=a.device,
        resume=not a.no_resume,
        ckpt_every=a.ckpt_every,
        compile=a.compile,
        init_from=a.init_from,
        pre_lr_mult=a.pre_lr_mult,
        init_full=a.init_full,
        model=ModelConfig(
            d_model=a.d_model, n_layers=a.layers, n_heads=a.heads, d_ff=4 * a.d_model,
            struct_notes=a.struct_notes, gap_input=a.gap_input,
            move_number=a.move_number, hist_moves=a.history, whole_notes=a.whole_notes,
            future=a.future, future_boards=a.future_boards,
            maia=a.maia, lam_init=a.lam_init,
        ),
        data=DataConfig(
            games_path=a.games,
            maia_inputs=bool(a.maia),
            p_gap_known=a.p_gap_known if a.p_gap_known is not None else (0.7 if a.gap_input else 0.0),
        ),
    )
    train(cfg)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="recon")
    sub = p.add_subparsers(required=True)

    f = sub.add_parser("fetch", help="stream games from Lichess into parquet")
    f.add_argument("--source", default="2026-09", help="YYYY-MM, URL, or local .pgn[.zst]")
    f.add_argument("--limit", type=int, default=20000)
    f.add_argument("--out", default="data/games.parquet")
    f.add_argument("--min-plies", type=int, default=20)
    f.add_argument("--speeds", default="blitz,rapid,classical")
    f.add_argument("--require-clocks", action="store_true")
    f.set_defaults(fn=cmd_fetch)

    b = sub.add_parser("build-bench", help="corrupt held-out games into benchmark instances")
    b.add_argument("--games", default="data/games.parquet")
    b.add_argument("--out", default="bench/test.jsonl")
    b.add_argument("--split", default="test", choices=["train", "val", "test"])
    b.add_argument("--presets", default="all", help=f"comma list of {','.join(PRESETS)} or 'all'")
    b.add_argument("--num-games", type=int, default=200)
    b.add_argument("--seed", type=int, default=0)
    b.set_defaults(fn=cmd_build)

    r = sub.add_parser("run", help="run a solver on a benchmark")
    r.add_argument("--solver", required=True)
    r.add_argument("--bench", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--presets", default="", help="only run these presets (comma list)")
    r.add_argument("--ckpt", default="", help="model checkpoint for model-based solvers")
    r.add_argument("--opt", action="append", default=[], help="solver option key=value (repeatable)")
    r.set_defaults(fn=cmd_run)

    e = sub.add_parser("eval", help="score predictions against a benchmark")
    e.add_argument("--bench", required=True)
    e.add_argument("--preds", required=True)
    e.add_argument("--only-predicted", action="store_true")
    e.add_argument("--json", action="store_true")
    e.set_defaults(fn=cmd_eval)

    t = sub.add_parser("train", help="train the notation-aware move model")
    t.add_argument("--games", default="data/games_500k.parquet")
    t.add_argument("--out", default="runs/v1")
    t.add_argument("--steps", type=int, default=20000)
    t.add_argument("--batch-size", type=int, default=256)
    t.add_argument("--lr", type=float, default=5e-4)
    t.add_argument("--eval-every", type=int, default=1000)
    t.add_argument("--log-every", type=int, default=100)
    t.add_argument("--workers", type=int, default=8)
    t.add_argument("--amp", action="store_true")
    t.add_argument("--val-games", type=int, default=300)
    t.add_argument("--device", default="auto")
    t.add_argument("--no-resume", action="store_true", help="start fresh even if OUT/state.pt exists")
    t.add_argument("--ckpt-every", type=int, default=2000, help="steps between resumable checkpoints")
    t.add_argument("--compile", action="store_true", help="torch.compile the model (CUDA speed-up)")
    t.add_argument("--d-model", type=int, default=256)
    t.add_argument("--layers", type=int, default=6)
    t.add_argument("--heads", type=int, default=8)
    t.add_argument("--whole-notes", action="store_true", help="ablation: written entries as whole tokens, not characters")
    t.add_argument("--struct-notes", action="store_true", help="parsed scoresheet entries linked to squares")
    t.add_argument("--gap-input", action="store_true", help="model is told plies until the next written move")
    t.add_argument("--move-number", action="store_true", help="model is told the full-move number")
    t.add_argument("--history", type=int, default=0, help="tokens for the last N moves (max 8)")
    t.add_argument("--future", type=int, default=0, help="tokens for up to N written entries after this ply (max 48)")
    t.add_argument("--future-boards", type=int, default=0, help="aux target: boards before the next N written plies (max 2)")
    t.add_argument("--p-gap-known", type=float, default=None, help="fraction of games with gap given (default 0.7 with --gap-input)")
    t.add_argument("--init-full", default="", help="start from a whole trained hybrid checkpoint (fine-tune)")
    t.add_argument("--maia", default="", help="hybrid with a pretrained Maia-3 tower: 5m or 23m (E19b)")
    t.add_argument("--init-from", default="", help="hybrid: start our tower from this checkpoint (E16 best.pt)")
    t.add_argument("--pre-lr-mult", type=float, default=0.1, help="hybrid: lr multiplier for pretrained towers")
    t.add_argument("--lam-init", type=float, default=1.5, help="hybrid: initial weight of Maia's move scores")
    t.set_defaults(fn=cmd_train)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
