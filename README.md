# Reading Between the Lines: Recovering Unrecorded Moves from Chess Scoresheets

Code, benchmark and experiment log for the paper *Reading Between the Lines: Recovering Unrecorded Moves from Chess Scoresheets* (Subhash Bylaiah, 2026). arXiv link: to be added.

Over-the-board chess games are recorded by hand, and players often stop writing in time trouble. Existing scoresheet tools read what was written but leave **unwritten moves** for the user to type.

This project ranks the legal moves at every empty box of a scoresheet. A user confirms one move at a time: tap one of the top two, expand to five, or type it.

| Test set (5,871 missing moves) | Top-1 | Top-2 | Top-5 | Typed moves / game | Games with no typing |
|---|---|---|---|---|---|
| Maia-3 23M alone (human-move prior, no scoresheet) | 55.6 | 74.9 | 91.3 | – | – |
| Scoresheet model alone + filter + search | 58.4 | 75.0 | 89.8 | 1.18 | 46% |
| **Final system** (scoresheet model + Maia-3, filter, search) | **66.4** | **82.6** | **94.3** | **0.66** | **61%** |

On real handwritten scoresheets (HCS), the final system reaches 58.8 / 75.0 / 88.2.

## What's here

| Path | Contents |
|---|---|
| `src/recon/` | Library: game storage, scoresheet simulator (`observe/`), benchmark builder and metrics (`bench/`), models (`model/`), filter and search (`solvers/`) |
| `bench/` | Benchmarks: `test.jsonl`, `dev.jsonl` (Lichess, CC0), `broadcast.jsonl` (real OTB tournament games, CC BY-SA 4.0). See [`bench/README.md`](bench/README.md). |
| `scripts/` | Evaluation, analysis and data scripts; `scripts/cloud/` holds cloud-GPU training helpers |
| `experiments/README.md` | The full chronological research log (E00–E32), including negative results |
| `experiments/results/` | Per-ply ranks and summaries for every reported number |
| `paper/` | LaTeX source of the paper |
| `tests/` | Unit tests (`uv run pytest`) |

## Setup

```sh
uv sync            # Python 3.12; installs PyTorch, python-chess and Maia-3 (from GitHub)
uv run pytest
```

The hybrid model downloads the Maia-3 weights from Hugging Face (`UofTCSSLab/Maia3-5M` / `Maia3-23M`) on first use.

**Model checkpoints:** to be published on Hugging Face (link to be added). Until then, models can be retrained with the commands below.

## The task in one picture

A benchmark instance holds an `input` (what a solver may see: the written entries, ratings, time control) and a `truth` (the true moves, which plies are missing, and the kind of every written entry). In the **assisted setting**, at each missing ply the solver ranks all legal moves given the confirmed moves so far and the whole scoresheet. Evaluation always continues from the true move.

## Reproducing the results

**1. Data**

```sh
uv run recon fetch --source 2026-08 --limit 500000 --out data/games_500k.parquet   # training games
uv run recon fetch --source 2026-09 --limit 50000  --out data/games.parquet        # test/dev games
```

**2. Benchmarks.** These are already in `bench/`; to rebuild them:

```sh
uv run recon build-bench --games data/games.parquet --split test --num-games 200 --out bench/test.jsonl
uv run recon build-bench --games data/games.parquet --split val  --num-games 200 --out bench/dev.jsonl
uv run python scripts/broadcast_bench.py data/broadcast/lichess_db_broadcast_2026-09.pgn.zst bench/broadcast.jsonl 150
```

**3. Training** (on a GPU; `scripts/cloud/queue.sh` has every run):

```sh
# scoresheet model (paper section 4.2), 100k steps
uv run recon train --out runs/c16_future --steps 100000 --eval-every 5000 --amp --compile \
    --games data/games_500k.parquet --future 48 --future-boards 2
# hybrid with Maia-3 23M, 50k steps from the scoresheet model
uv run recon train --out runs/c19c_hybrid23m --steps 50000 --eval-every 5000 --amp --compile \
    --games data/games_500k.parquet --future 48 --future-boards 2 \
    --maia 23m --init-from runs/c16_future/best.pt --pre-lr-mult 0.1 --lam-init 1.5
```

The OTB fine-tune (paper section 6.2) uses `scripts/broadcast_games.py` to build the training data, then `queue.sh c31`.

**4. Evaluation**

```sh
# model only (no filter, no search)
uv run python scripts/assisted_eval.py --ckpt runs/c19c_hybrid23m/best.pt --penalty 0
# final setting: filter + search within 6 plies (chosen on bench/dev.jsonl)
uv run python scripts/assisted_eval.py --ckpt runs/c19c_hybrid23m/best.pt \
    --search --max-dist 6 --penalty 6 --first-moves 10 --width 8 --save ranks.jsonl
# 95% intervals over games
uv run python scripts/bootstrap_ci.py ranks.jsonl
```

Other analyses: `scripts/feasibility_baseline.py` (legality-only baseline), `scripts/miss_analysis.py` (error analysis), `scripts/maia_eval.py` (Maia-3 alone and untrained sums). Each experiment's exact command is in `experiments/README.md`.

**Real scoresheets (HCS).** Download the HCS dataset from its authors (https://sites.google.com/view/chess-scoresheet-dataset), extract it to `data/hcs/x/`, then run:

```sh
uv run python scripts/hcs_games.py          # per-page written entries
uv run python scripts/hcs_reconstruct.py    # recover the true games
uv run python scripts/hcs_bench.py 0.15     # bench/hcs.jsonl
```

## Data and licenses

- **Code:** AGPL-3.0-or-later (`LICENSE`); the hybrid model builds on Maia-3, which is AGPL-3.0.
- **Lichess games** (`test`, `dev`): CC0, from https://database.lichess.org.
- **Lichess broadcast games** (`broadcast.jsonl`): CC BY-SA 4.0, from the Lichess broadcast database. Derived files keep that license.
- **HCS:** not redistributed. Cite Eicher et al. (2021) and obtain it from the authors.

## Citation

To be added with the arXiv identifier.
