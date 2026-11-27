# Benchmarks

One JSON object per line. Each line is one **instance**: one game under one damage preset.

```
{"id", "game_id", "split", "preset", "meta": {...},
 "input": {...},   # what a solver may see
 "truth": {...}}   # ground truth; never given to a solver
```

- **`input`:**
  - `start_fen`, player ratings and time control;
  - `sources`: one or two scoresheets. Each is a list of written `tokens`, with `text`, optional `move_number_hint`, `side_hint`, `confidence` and `alternatives` (recogniser n-best list).
- **`truth`:**
  - `moves_uci`: the true game;
  - `hidden_plies`: the plies missing from every source;
  - per source: `token_ply` (which ply each token records, or null) and `token_kind` (`match`, `ocr`, `slip`, `inserted`).

| File | Games | Presets | Missing plies (missing, mixed, mixed_ocr) | Source and license |
|---|---|---|---|---|
| `test.jsonl` | 200 (Lichess, September 2026) | all 6 | 5,871 | Lichess database, CC0 |
| `dev.jsonl` | 200 (held out, unseen in training) | all 6 | 5,787 | Lichess database, CC0 |
| `broadcast.jsonl` | 150 per speed (classical, rapid, blitz) | missing, mixed, mixed_ocr | 14,069 | Lichess broadcast database (real over-the-board tournament games), September 2026, CC BY-SA 4.0 |
| `short_gaps.jsonl` | 80 | short gaps only | – | Lichess, CC0 (used in an early experiment) |

**Presets:**

| Preset | What it simulates |
|---|---|
| `clean` | No damage |
| `missing` | One time-pressure burst of unrecorded plies |
| `substitution` | Misreadings and slips only |
| `mixed` | Burst + light errors |
| `mixed_ocr` | Heavier misreadings with recogniser n-best lists |
| `two_sheets` | Both players' records |

The paper reports `missing`, `mixed` and `mixed_ocr`; rates are in its appendix A.

**The HCS-derived set** (`hcs.jsonl`: real handwritten scoresheets with simulated gaps) is not included, because HCS is not ours to redistribute. Rebuild it from the original release with `scripts/hcs_games.py`, `hcs_reconstruct.py` and `hcs_bench.py`.

**Evaluate** a model with `scripts/assisted_eval.py --bench <file>`.
