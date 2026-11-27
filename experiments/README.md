# Experiment log

> **About this file.** This is the chronological working log of the project, kept as experiments were run, including negative results, dead ends and later corrections. The paper (`paper/`) is the curated summary; where they differ, later entries in this log and the paper are authoritative. Run names (`runs/...`) refer to checkpoints, which are not in the repository.

Every variant we try is recorded here, including negative results, so the full set can go in the write-up.
Unless noted, the test data is `bench/test.jsonl`: 200 September-2026 games, none of them seen in training, under 6 damage presets.

## Settings

- **Primary: user-assisted, gap length known.** The player reconstructs their own game move by move. The scoresheet layout says which plies are blank. At each missing ply the moves before it are confirmed, so the board is the true one. We measure the rank of the true move in the suggestion list.
  - **Metrics:** top-1/3/5, moves typed per game (true move not in the top 5), and the share of games needing no typing.
- **Ablation: automatic, gap length unknown.** The search outputs a whole game on its own.
  - **Metrics:** exact game, move accuracy, missing-move accuracy.

## Summary

| ID | Experiment | Setting | Checkpoint | Data | Headline | Status |
|---|---|---|---|---|---|---|
| E00 | Chance level (uniform over legal moves) | primary | – | same 5,871 missing plies as E05 | top-1 6.9%, top-3 16.9%, top-5 24.1% (median 33 legal moves) | done |
| E01 | Rules-only parser (baseline) | automatic | – | full bench, 1,200 | exact 18.2%, move acc 45.6%, missing 0.5% | done |
| E00b | Legality + feasibility only (no learned model; filter-feasible moves first, random order) | primary | – | 5,871 missing | expected top-1 9.6, top-2 15.8, top-5 27.7 (chance 6.9 / 12.4 / 24.1); the true move survives the filter on 98.8% of plies, median 31 candidates (`scripts/feasibility_baseline.py`) | done |
| E02 | v1 model + beam search | automatic | v1 @30k | full bench, 1,200 | exact 33.8%, move acc 75.1%, missing 15.4% | done |
| E03 | Short gaps: model vs search vs rules | automatic | v1 @26k | short_gaps, 80 | 1 ply: search 90% / model 57.5% / rules 12.5% | done |
| E04 | Error diagnosis (model vs search) | automatic | v1 @30k | missing+mixed with gaps | 71–82% of long-gap failures are model errors | done |
| E05 | **Assisted, model-only suggestions** | **primary** | v1 @30k | 507 games, 5,871 missing | **top-1 41%, top-5 77%, 2.7 typed/game** | done |
| E06 | Assisted, gap-aware search (log-sum-exp) | primary | v1 @30k | first 20 games, 277 missing | top-5 72% overall; 96% at 1 ply from next written, 63% at 9+ | done (pilot) |
| E07 | Assisted, search with averaged evidence | primary | v1 @30k | first 20 games | top-5 68%: worse than E06 at every distance | done (negative) |
| E08 | Structured notation (`--struct-notes`) | both | v2_struct @14.5k | same data, steps and val as v1 | val missing top-1: ahead of v1 by up to +1.2 points through 10k, behind at 12k (30.9% vs 31.6%) | **stopped at 14.5k** (inconclusive) |
| E09 | Structured notation + known gap length | both | v3_struct_gap | same | – | not run (replaced by E12) |
| E10 | Move number + last 8 moves (`--move-number --history 8`) | primary | v4_hist @30k | 507 games, 5,871 missing (as E05) | top-1 36.4%, top-5 68.6%, 3.6 typed/game: **worse than v1** on every metric | done (negative) |
| E11 | **Feasibility filter + search within 3 plies of a written move** (v1, no retraining) | **primary** | v1 @30k | 507 games, 5,871 missing | **top-1 46.7%, top-2 61.2%, top-5 78.9%, top-10 90.6%; 2.44 typed/game; 27% of games no typing** (E05: 41.1 / 57.3 / 76.6 / 89.9; 2.71; 22%) | done (current best) |
| E12 | v1 + known gap length (`--gap-input`) | primary | v5_gap @30k | 507 games, 5,871 missing (as E05) | gap given: top-1 40.4%, top-5 75.1% (v1 41.1 / 76.6); +10 points at 2 plies, −1 to −2 at 4+ | done (negative overall) |
| E13 | **v1 trained 100k steps instead of 30k** (only change) + E11 filter and 3-ply search | **primary** | v6_long @100k | 507 games, 5,871 missing | **top-1 57.4%, top-2 74.1%, top-5 89.2%, top-10 96.2%; 1.25 typed/game; 45% of games no typing** (model alone 54.1 / 71.9 / 88.2) | **done (current best)** |
| E13c | E13 rerun on the cloud (A10, bf16, `torch.compile`): baseline for E16 | primary | c13_long @100k | same as E13 | val 55.3 / 89.6 (Mac E13 55.7 / 89.3); test + filter + search 57.3 / 73.5 / 89.7 (Mac 57.4 / 74.1 / 89.2) | done |
| E14 | Robustness: accuracy vs misread and slip rates | primary | E13 | test games, damage swept 0→high | – | planned (analysis) |
| E15 | Identifiability: share of errors no method could fix (move order / traceless gap moves) | primary | – | E11/E13 missing plies | – | planned (analysis) |
| E16 | **Future-record model**: every later written entry as a token (piece, square, capture, castle, exact ply offset; up to 48) + auxiliary loss predicting the board before the next 2 written plies | **primary** | c16_future @100k (cloud A10) | 507 games, 5,871 missing | **model alone: top-1 57.4%, top-2 74.2% (c13: 54.7 / 72.2); + filter + 3-ply search: 58.4 / 75.0 / top-5 89.8; 1.18 typed/game; 46% no typing** | **done (current best)** |
| E16nb | E16 without the auxiliary board loss (ablation) | primary | c16_noboard @100k (A10) | 507 games, 5,871 missing | model 54.4 / 71.6 / 88.0 (≈ c13); + filter + search 55.9 / 72.7 / 88.6. **Board task is the whole E16 gain** | done |
| E16f16 | E16 with 16 instead of 48 future entries (ablation) | primary | c16_f16 @100k (A100) | 507 games, 5,871 missing | model 56.3 / 73.2 / 89.2; + filter + search 57.4 / 74.1 / 89.7. About −1 vs E16 | done |
| E17a | Bigger model (15.25M params), E16 setup, 500k games | primary | c17a_big @100k (A10) | 507 games, 5,871 missing | + filter + search 58.5 / 74.8 / 90.0: **same as E16**, capacity not the bottleneck | done |
| E17b | E17a on 2M games | primary | c17b_big2m @100k (A100) | 507 games, 5,871 missing | + filter + search 57.6 / 75.1 / 90.1: **same as E16**, data not the bottleneck either | done |
| E15a | Why E16 misses: order swaps vs indistinguishable vs other | analysis | E16 | 5,871 missing | top-1 misses: order 10.9, equivalent 3.8, other 27.9 (% of plies); set-level top-2 about 85% | done |
| E15b | Miss analysis on the hybrid (E19c, model + filter) | analysis | E19c | 5,871 missing | top-1 misses: order 9.2, equivalent 3.3, other 23.2 (% of plies); order is 39% of top-2 misses; 79.6% top-1 if picking uniformly among sheet-consistent moves (not a ceiling) | done |
| E19a | Maia-3 (5M) baseline + zero-training combination with E16 | primary | Maia3-5M + c16_future | 5,871 missing | Maia alone 54.3 / 73.9 / 90.9; **E16 + Maia + filter 61.9 / 78.2 / 92.5 (cross-fitted)** | done (new best) |
| E19b | Hybrid: E16 tower + Maia-3 5M tower, gated cross-attention, fine-tuned 50k steps (15.2M params) | primary | c19b_hybrid @50k (A100) | 5,871 missing | **63.4 / 79.9 / 93.4; + search 64.0 / 80.7 / 93.7, 0.72 typed/game, 59% no typing (new best)** | done |
| E19c | E19b with Maia-3 23M (46M params) | primary | c19c_hybrid23m @50k (A100) | 5,871 missing | 64.4 / 80.9 / 93.6 (no search); **+ search 65.0 / 81.7 / 93.9, 0.70 typed/game, 60% no typing (new best)** | done |
| E20 | E19c + tokens for the last 8 moves in our tower (`--history 8`, fresh tokens at full lr) | primary | c20_hist @50k (A100) | 5,871 missing | 64.3 / 80.8 / 93.8 (filter, no search): **no change vs E19c** | done (null) |
| E21 | Dev set + re-tune search settings on it (for E19c) | primary | c19c_hybrid23m | dev: 200 val games, 5,787 missing | dev-chosen setting (search within 6 plies) on test, run once: **66.4 / 82.6 / 94.3**, 0.66 typed/game, 61% no typing (headline) | done |
| E29 | Real scoresheets (HCS): real written entries + simulated gaps | real-data check | E19c, E16 | HCS: 80 sheets, 1,262 missing | E19c 55.5 / 71.6 / 87.4 (top-1/2/5), E16 47.7 / 63.5 / 81.5; about 9 points below the simulated test | done (first pass) |
| E30 | Simulated gaps on real over-the-board games (Lichess broadcasts, Sep 2026) by speed | real-data check | E19c, E16 | 450 games, 14,069 missing | E19c: blitz 65.1 / 81.2, rapid 64.0 / 80.8, **classical 58.6 / 75.2** (top-1/2) | done |
| E31 | E19c fine-tuned on real OTB tournament games (broadcasts 2020-01..2026-08) + 400k Lichess games | primary + real | c31_otb @20k (A100) | E30, E29, test, dev | OTB classical 58.6→**61.5** / 75.2→**77.8**; HCS real 55.5→**56.8** / 71.6→**74.2**; Lichess test 64.0 / 80.7 (E19c 64.4 / 80.9) | done |
| E32 | Ablation: written entries as whole tokens instead of characters (E16 setup, `--whole-notes`) | primary + real | c16_whole @100k (A100) | test, HCS | model + filter 56.9 / 74.0 / 89.8 vs characters 57.4 / 74.2 / 89.6: **no overall difference**; before a wrong entry chars +2.6 / +4.1 (not significant); HCS real chars +0.7 / +1.8 | done |
| E18 | Separate learned future encoder (cross-attention / ~8 summary tokens), only if E16 shows long records matter | primary | – | – | – | planned (training) |

## Details

### E00: chance level
- **Method:** the true move's rank drawn uniformly among the legal moves, on exactly the missing plies scored in E05.
- **Legal moves per position:** median 33, mean 30.4; 10th percentile 10, 90th percentile 44.
- **Command:** `uv run python scripts/chance_baseline.py`

### E01: rules-only parser
- **Method:** read the sheet left to right and keep each entry that is legal; try OCR alternatives too. It can't fill gaps.
- **Command:** `recon run --solver naive`
- **Result:** `preds/naive.jsonl`

### E02: v1 model + beam search, gap length unknown
- **Model:** 5.22M parameters, 6 layers, d=256. 87 input tokens: 64 squares, 6 context, 16 notation entries as characters.
- **Training:** 30k steps × 256 positions, on 500k August-2026 games. The search is observation-aware, with beam 32.
- **Result:** `preds/beam_v1.jsonl`
- **By gap length (exact game):**

| Gap | Exact game |
|---|---|
| 0 | 67% |
| 1–4 | 28.7% |
| 5–9 | 1.6% |
| 10+ | 0% |

- **By preset:** clean 100%, substitution 39.5%, mixed 21.5%, mixed_ocr 19.5%, two_sheets 20.5%, missing 1.5%.
- **Known limitation:** `two_sheets` reads sheet 1 only.

### E03: short gaps (`bench/short_gaps.jsonl`)
- **Data:** 40 games with a 1-ply gap and 40 with a 2-ply gap; no other errors.
- **Results:** `preds/short_*.jsonl`

| | 1-ply gap | 2-ply gap |
|---|---|---|
| Model alone, true board given: top-1 | 57.5% | 46.3% |
| Model alone, true board given: top-5 | 92.5% | 83.7% |
| Model + search: missing moves right | 90% | 46.3% |
| Model + search: whole game exact | 87.5% | 37.5% |
| Rules-only search (`uniform_prior=1`): whole game exact | 12.5% | 0% |

### E04: error diagnosis
- **Method:** for each failed game, score the true game with `BeamSolver.score_path` and compare it with the search's top answer.
- **Definitions:**
  - "model error": the true game scores lower than the answer;
  - "search error": the true game scores higher but was pruned;
  - "no answer": the search didn't finish.

| Gap | Model error | Search error | No answer |
|---|---|---|---|
| 1–4 | 51% | 7% | 10% |
| 5–9 | 82% | 6% | 11% |
| 10–19 | 72% | 5% | 23% |
| 20+ | 71% | 2% | 27% |

### E05: assisted, model-only suggestions (primary baseline)
- **Command:** `uv run python scripts/assisted_eval.py`
- **Presets:** missing, mixed, mixed_ocr.
- **Result:** `results/E05_assisted_model_v1.json`

| Longest gap | Top-5 | Typed / game | No typing |
|---|---|---|---|
| 1–4 | 84% | 0.4 | 67% |
| 5–9 | 79% | 1.5 | 23% |
| 10–19 | 76% | 3.4 | 4% |
| 20+ | 76% | 6.4 | 0% |
| **All** | **77%** | **2.7** (of 11.6 missing) | **22%** |

- **Caveat:** bad written entries in the mixed presets aren't counted; only missing moves are.
- **Full top-k ladder** (`scripts/topk_table.py results/E05_ranks_all.jsonl`; chance in brackets):

| Top-k | Model | Chance |
|---|---|---|
| 1 | 41.1% | 6.9% |
| 2 | 57.3% | 12.4% |
| 3 | 66.2% | 16.9% |
| 4 | 72.0% | 20.7% |
| 5 | 76.6% | 24.1% |
| 8 | 85.8% | 33.4% |
| 10 | 89.9% | 39.3% |

- **Rank of the true move:** median 2, mean 4.12 (chance mean 15.7).

### E06 and E07: assisted, gap-aware search (pilot, 20 games)
- **Code:** `src/recon/solvers/assisted.py`
- **Method:** for each candidate first move, run a beam of width 8 through the rest of the gap, then play the next 6 written moves on top. Written moves are scored with the observation model, so misreads are tolerated.
  - **E06:** score each candidate by log-sum-exp over its kept continuations.
  - **E07:** score by evidence averaged over its continuations.
- **Command:** `scripts/assisted_eval.py --search --limit 20`
- **Breakdown script:** `scripts/by_distance.py`
- **Results:** top-1 / top-5 by distance to the next written move, on the same 277 missing moves:

| Distance | E05 model | E06 search (log-sum-exp) | E07 search (average) |
|---|---|---|---|
| 1 | 48 / 78 | **81 / 96** | 81 / 96 |
| 2 | 30 / 65 | **55 / 80** | 45 / 80 |
| 3 | 35 / 65 | **50 / 70** | 45 / 70 |
| 4–8 | 29 / 73 | 38 / 75 | 33 / 70 |
| 9+ | **41 / 78** | 26 / 63 | 25 / 59 |

- **Takeaway:** search is a large win near the end of a gap and harmful deep inside a long gap. Next step: search only within a cutoff distance, plus a feasibility filter.
- **Code state:** `assisted.py` currently implements the E07 averaging. Revert to log-sum-exp before the next search experiment.

### E08: structured notation (`runs/v2_struct`)
- **Idea:** from "structure over scale" (gwenlake.com/blog/structure-over-scale): don't make the model learn what written moves mean from characters alone.
- **Change:** each entry is also parsed (`features.parse_entry`) into piece, to-square, capture, from-file, from-rank and promotion.
  - The to-square reuses the square-position embedding.
  - Each square token receives a learned vector when upcoming entry k names it as a destination.
  - The character view is kept, so garbled entries still work.
- **Controls:** same 500k games, 30k steps, 5.24M parameters, seed and val set (23,924 positions) as v1.
- **Command:** `scripts/queue_struct.sh`
- **Log:** `runs/v2_struct.out`
- **Outcome:** stopped at step 14,500 to make room for E10. The laptop also lost power around step 9,500–10,000; the run paused and resumed, so this doesn't change the results.

| Step | v1 top-1 | E08 top-1 | v1 top-5 | E08 top-5 |
|---|---|---|---|---|
| 2,000 | 25.2 | 25.6 | 53.5 | 54.0 |
| 4,000 | 26.8 | 28.0 | 55.3 | 56.8 |
| 6,000 | 27.6 | 28.3 | 56.7 | 58.2 |
| 8,000 | 28.6 | 29.0 | 59.5 | 59.3 |
| 10,000 | 29.8 | 30.6 | 61.9 | 60.8 |
| 12,000 | 31.6 | 30.9 | 63.6 | 61.9 |

- **Reading:** a small early gain that fades; no clear ceiling increase. The validation noise is about ±0.5 point.
- **Relation to gwenlake:** the post has no notation input (board only), so this is our own extension, not a replication.

### E10: move number + history (`runs/v4_hist`)
- **Tests:** the Markov assumption. v1 sees only the current board plus the last move's squares.
- **Changes:**
  - one context token for the full-move number (1–100), giving game phase;
  - 8 history tokens, most recent first. Each holds the mover-relative piece (own or opponent, PNBRQK), from-square and to-square (sharing the square-position embedding), a capture flag, and a "plies ago" slot.
- **Unchanged:** data, steps, seed and val set match v1. 5.25M parameters.
- **Command:** `scripts/queue_history.sh`
- **Log:** `runs/v4_hist.out`
- **Search compatibility:** boards in the searches now keep 8 moves of stack (`copy(stack=F.HIST)`) so history can be encoded there.
- **Outcome: negative.** Behind v1 from step 10k on, and the gap kept growing. Training loss lags as much as validation loss (0.911 vs 0.820 at 22k), so it learns more slowly rather than overfitting. Likely reason (not tested): the board already holds almost all the information, and the extra tokens cost training efficiency under a fixed budget.
- **Note:** the laptop slept overnight around step 6,500; the process was suspended and resumed, not restarted.

| Step | v1 top-1 / top-5 | E10 top-1 / top-5 |
|---|---|---|
| 10,000 | 29.8 / 61.9 | 29.1 / 59.8 |
| 14,000 | 33.1 / 66.6 | 31.9 / 63.5 |
| 18,000 | 36.4 / 72.0 | 33.9 / 65.5 |
| 22,000 | 38.6 / 74.9 | 34.7 / 67.2 |
| 30,000 | 41.0 / 78.2 | 37.0 / 70.2 |

- **Assisted test eval (same plies as E05):** `results/E10_assisted.txt`, `results/E10_ranks_all.jsonl`

| Metric | E05 v1 | E10 |
|---|---|---|
| Top-1 | 41.1% | 36.4% |
| Top-3 | 66.2% | 58.1% |
| Top-5 | 76.6% | 68.6% |
| Top-10 | 89.9% | 83.4% |
| Mean rank | 4.12 | 5.38 |
| Typed / game | 2.7 | 3.6 |
| No typing | 22% | 18% |

### E09: structured notation + known gap length (`runs/v3_struct_gap`)
- **Change:** an extra context token gives the plies until the next written move. It's given for 70% of training games and "unknown" for the rest, so one model serves both settings.
- **Note:** v3's validation set gives the gap for 70% of games, so its validation hidden accuracy is not directly comparable with v1 and E08. Compare on E05-style assisted eval, with and without the gap.

### E11: feasibility filter and search sweep (`results/E11/`)
- **Filter (`src/recon/solvers/feasible.py`):** deterministic necessary conditions from later written entries; no play-out. Castling rights never return; the piece making a later written move must exist (promotion allowed) and reach the destination within its side's remaining moves (empty-board distance); a written capture needs an enemy piece able to be on that square in time. An entry constrains only if every OCR reading with p ≥ 0.05 is violated. Applied as a penalty (6 per violation; hard removal gave the same results).
- **Soundness:** on the `missing` preset (entries correct) the true move was never flagged, at any distance. With slips/OCR errors (mixed, mixed_ocr) the true move was flagged 1–4% of the time.
- **Strength:** removes 37% of legal moves at 1 ply from the next written move, 25% at 2, 4–8% at 3, under 3% beyond.
- **Model + filter, 507 games:** top-1 41.1→42.7%, top-2 57.3→58.5%, top-5 76.6→77.5%, typed/game 2.71→2.61.
- **Search sweep, 20-game pilot (search at every distance), top-1 by distance:**

| Setup | 1 | 2 | 3 | 4–8 | 9+ | All top-1 / top-2 | s/suggestion |
|---|---|---|---|---|---|---|---|
| Model | 48 | 30 | 35 | 29 | 41 | 36.8 / 55.2 | 0 |
| Model + filter | 56 | 35 | 35 | 30 | 41 | 38.3 / 55.2 | 0 |
| E06 repro (16 cand × 8, no filter) | 81 | 55 | 50 | 38 | 26 | 39.0 / 50.5 | 0.7 |
| 16 × 8 + filter | 81 | 55 | 50 | 38 | 26 | 39.0 / 50.5 | 0.7 |
| 10 × 8 + filter | 81 | 55 | 50 | 40 | 26 | 40.1 / 51.6 | 0.5 |
| 5 × 8 + filter | 81 | 45 | 50 | 38 | 33 | 41.5 / 53.4 | 0.3 |
| 10 × 32 + filter | 81 | 55 | 55 | 33 | 32 | 40.4 / 53.4 | 1.7 |
| 10 × 128 + filter | 81 | 55 | 55 | 33 | 35 | 41.5 / 55.6 | 5.8 |

- **Reading:** E06 reproduces exactly. Near a written move all settings agree. Far away, wider beams help (26→32→35%) but stay below the model alone (41%): with many blank plies almost any candidate can reach the written move, so the check doesn't discriminate.
- **Estimated combination** (10 × 8 search within 8 plies, model + filter beyond; computed from the rows above, not run): top-1 about 46.5%, top-2 about 59%, top-5 about 78% on the pilot (model alone 36.8 / 55.2 / 74.4).
- **Pending:** full top-2/top-3 tree runs (stopped to free the GPU for E12); the combined setting on all 507 games.
- **Command:** `scripts/sweep_E11.sh`
- **Full run, 507 games** (filter + 10 candidates × 8 paths, search within 8 plies): `results/E11/full_c10_w8_d8.json`, 0.22 s per suggestion. Top-1 by distance: 1: 77.4%, 2: 69.1%, 3: 52.0%, 4–8: **34.5%** (model + filter 37.6%), 9+: 39.6%. So at full size search loses at 4–8 plies; the pilot's 4–8 gain was noise (n=89).
- **Cutoff, exact from the saved per-ply ranks** (`scripts/combine_cutoff.py`; the cutoff only chooses which ranking each ply gets):

| Search within | Top-1 | Top-2 | Top-5 | Top-10 | Typed/game | No typing |
|---|---|---|---|---|---|---|
| 0 (model + filter) | 42.7 | 58.5 | 77.4 | 90.3 | 2.61 | 23% |
| 1 ply | 44.5 | 59.8 | 78.3 | 90.6 | 2.51 | 25% |
| 2 plies | 45.7 | 60.5 | 78.6 | 90.6 | 2.48 | 27% |
| **3 plies** | **46.7** | **61.2** | **78.9** | **90.6** | **2.44** | **27%** |
| 8 plies | 45.8 | 59.9 | 77.8 | 90.6 | 2.57 | 27% |

- **Final setting:** filter everywhere + 10 × 8 search within 3 plies of the next written move. Ranks: `results/E11/final_cut3_ranks.jsonl`. Mean rank 3.84 (chance 15.7), median 2. Top-3 69.3%, top-4 74.9%, top-8 87.1%.
- **Caveat for the write-up:** the cutoff was chosen on the test games. It's one parameter picked from 5 values, so the bias is small, but a held-out validation benchmark should confirm it.

### E12: known gap length (`runs/v5_gap`)
- **Change:** one extra context token: plies until the next written move, given for 70% of training games and "unknown" for the rest (`--gap-input`).
- **Why:** the model already uses later entries (top-1 58% at 1 ply vs 40% at 9+) but must guess the distance from noisy move-number hints.
- **Comparability:** val gives the gap for 70% of games, so compare on the assisted test eval with `--gap-known`, not on the val curve.
- **Command:** `scripts/queue_gap.sh`; log `runs/v5_gap.out`
- **Result (assisted test eval, `results/E12/`), top-1 by distance to next written move:**

| | 1 | 2 | 3 | 4–8 | 9+ | All top-1 / top-2 / top-5 | Typed/game |
|---|---|---|---|---|---|---|---|
| v1 | 57.7 | 44.1 | 36.0 | 37.3 | 39.5 | 41.1 / 57.3 / 76.6 | 2.71 |
| E12, gap not given | 48.9 | 43.5 | 33.7 | 34.0 | 36.6 | 37.6 / 54.0 / 73.9 | 3.02 |
| E12, gap given | 55.1 | 54.1 | 36.2 | 35.2 | 38.2 | 40.4 / 55.7 / 75.1 | 2.88 |
| v1 + filter | 62.5 | 54.3 | 38.0 | 37.6 | 39.6 | 42.7 / 58.5 / 77.5 | 2.61 |
| E12, gap given + filter | 59.0 | 57.0 | 37.6 | 35.4 | 38.2 | 41.2 / 56.7 / 75.8 | 2.80 |

- **Reading:** knowing the gap helps right before a written move (2 plies: 44→54%), but the model is 1–2 points worse far from written moves, which is most of the plies, so it's slightly worse overall. With the filter, v1 is better everywhere except 2 plies. The filter already captures most of what the gap input adds near written moves. Validation (gap given for 70%): 40.7 / 75.8 vs v1 41.0 / 78.2.

### E13: v1 trained longer (`runs/v6_long`)
- **Change:** `--steps 100000` (v1: 30000). Config otherwise identical (checked by diff): 25.6M positions seen (3.3×, about 2.2 passes over the training games), cosine schedule stretched over 100k steps. 36,197 s on the M2 Pro (paused twice by battery sleep).
- **Validation, missing top-1 / top-5:** 30k 45.1 / 81.7 (v1 at 30k: 41.0 / 78.2), 50k 48.9 / 85.1, 75k 53.1 / 88.5, **100k 55.7 / 89.3**. Still improving at the end.
- **Assisted test eval (`results/E13/`):**

| | Top-1 | Top-2 | Top-5 | Top-10 | Mean rank | Typed/game | No typing |
|---|---|---|---|---|---|---|---|
| v1 model (E05) | 41.1 | 57.3 | 76.6 | 89.9 | 4.12 | 2.71 | 22% |
| v1 + filter + 3-ply search (E11) | 46.7 | 61.2 | 78.9 | 90.6 | 3.84 | 2.44 | 27% |
| E13 model | 54.1 | 71.9 | 88.2 | 96.1 | 2.66 | 1.36 | 40% |
| E13 + filter | 54.8 | 72.3 | 88.4 | 96.2 | 2.63 | 1.35 | 41% |
| **E13 + filter + 3-ply search** | **57.4** | **74.1** | **89.2** | **96.2** | **2.55** | **1.25** | **45%** |

- **Top-1 by distance (E13 model → + filter + search):** 1: 75.7 → 81.9; 2: 58.6 → 78.2; 3: 54.1 → 67.9; 4–8: 50.6 → 50.9; 9+: 49.6 → 49.6.
- **Reading:** training budget was the main limit. Far plies (4+) rose from about 39% to about 50% top-1 from training alone. Filter and search still add 6–20 points within 3 plies of a written move. Median rank of the true move is now 1.
- **Consequence:** E08, E10 and E12 were judged at 30k steps, which is now known to be badly under-trained; their negative results are "at a short budget" only.

### E13c and E16 on the cloud (`runs/c13_long`, `runs/c16_future`; `results/E13c/`, `results/E16/`)
- **Hardware:** Lambda 1× A10 each, run in parallel; `--amp` (bf16) and `--compile`; 12 data workers. c13: 5,160 pos/s, about 1.4 h. c16: 3,690 pos/s, about 1.9 h.
- **E16 changes vs E13 (only these):** `--future 48 --future-boards 2`. Every written entry after the current ply (layout known) becomes a token with characters, parsed fields, OCR confidence and exact ply offset (1..16 exact, then buckets). Auxiliary cross-entropy (weight 0.2) predicts the 64-square board before each of the next 2 written plies; it's ignored at suggestion time. Parameters 5.22M → about 5.3M.
- **Validation, missing top-1 / top-5:**

| Step | c13 | E16 |
|---|---|---|
| 30k | 43.8 / 80.7 | 49.3 / 83.6 |
| 50k | 49.2 / 85.3 | 53.7 / 87.5 |
| 65k | 52.5 / 87.2 | 55.0 / 88.7 |
| 100k | 55.3 / 89.6 | **58.4 / 90.5** |

- **Test (507 games, 5,871 missing):**

| | Top-1 | Top-2 | Top-3 | Top-5 | Top-10 | Mean rank | Typed/game | No typing |
|---|---|---|---|---|---|---|---|---|
| c13 model | 54.0 | 71.7 | 80.5 | 88.8 | 96.3 | 2.65 | 1.30 | 40% |
| c13 + filter + 3-ply search | 57.3 | 73.5 | 81.7 | 89.7 | 96.5 | 2.54 | 1.20 | 43% |
| E16 model | 57.2 | 74.1 | 81.9 | 89.6 | 96.7 | 2.52 | 1.21 | 44% |
| **E16 + filter + 3-ply search** | **58.4** | **75.0** | **82.4** | **89.8** | **96.8** | **2.48** | **1.18** | **46%** |

- **Top-1 by distance to the next written move (model + filter):** c13: 1: 77.6, 2: 67.1, 3: 54.5, 4–8: 50.4, 9+: 49.2. E16: 1: 82.2, 2: 72.1, 3: **51.8**, 4–8: 54.2, 9+: 50.9.
- **Reading:**
  - E16's model alone matches c13 with the search (57.4 vs 57.3 top-1, 74.2 vs 73.5 top-2). The model now does internally much of what the search did, with no search cost.
  - Search still adds +1 top-1 on top of E16 (mostly at 3 plies). Gains are largest at 1–2 plies and 4–8 plies (+4–5 points). Top-5 is nearly unchanged (+0.1–0.8), so the gain is mainly better ordering: more moves in the top 2.
  - Odd dip: E16 is weaker than c13 at exactly 3 plies (51.8 vs 54.5, n=442); search more than recovers it (59.5). Worth a look (offset embedding? sample noise?).
  - Validation lead was +5 points at 30k and +3.1 at 100k: faster learning, smaller but real final gain.
- **Open:** which part helps (future tokens vs exact offsets vs board task): ablate `--future-boards 0` and `--future 16`.

### E16nb: E16 without the board task (`runs/c16_noboard`, `results/E16nb/`)

- **Change vs E16 (only this):** `--future-boards 0`. Same future tokens, no auxiliary board loss. Cloud A10 #2, 100k steps.
- **Run note:** the run was killed by accident at about step 47k (while editing the instance's queue) and resumed from the step-46k checkpoint. About 1k steps were redone, and the data stream after the resume is reseeded (seed + step), so it is not bit-identical to an uninterrupted run. Model, optimizer and RNG state were restored.
- **Validation (missing top-1 / top-2 / top-5):** 25k 41.6 (E16 47.1); 45k 49.4 (E16 52.2); 100k 55.9 / 72.7 / 88.9 (E16 58.4 / – / 90.5; c13 55.3 / – / 89.6).
- **Test (507 games, 5,871 missing):**

| | Top-1 | Top-2 | Top-3 | Top-5 | Top-10 | Mean rank | Typed/game | No typing |
|---|---|---|---|---|---|---|---|---|
| c13 model | 54.0 | 71.7 | 80.5 | 88.8 | 96.3 | 2.65 | 1.30 | 40% |
| E16nb model | 54.4 | 71.6 | 79.6 | 88.0 | 96.1 | 2.73 | 1.39 | 42% |
| E16nb + filter | 54.6 | 71.6 | 79.7 | 88.0 | 96.2 | 2.71 | 1.38 | 43% |
| E16nb + filter + 3-ply search | 55.9 | 72.7 | 80.5 | 88.6 | 96.2 | 2.66 | 1.32 | 46% |
| E16 model | 57.2 | 74.1 | 81.9 | 89.6 | 96.7 | 2.52 | 1.21 | 44% |

- **Top-1 by distance (model + filter):** 1: 80.4, 2: 68.5, 3: 53.2, 4–8: 50.0, 9+: 48.2 (E16: 82.2 / 72.1 / 51.8 / 54.2 / 50.9).
- **Reading:** without the board task, the future tokens give almost nothing over c13 (model top-1 54.4 vs 54.0, top-2 equal). **E16's whole gain (+3 top-1, +2.5 top-2) comes from the auxiliary board loss.** The tokens alone carry the information, but the move loss by itself does not teach the model to use them; asking it to predict the board at the next written move does. The 3-ply dip is gone without the board task (53.2), so it may be a side effect of the board task (it predicts the board 3 plies away, where the move is most ambiguous).
- **Paper point:** a cheap auxiliary target (future board state) is the key ingredient, not the extra input.

### E16f16: E16 with only 16 future entries (`runs/c16_f16`, `results/E16f16/`)

- **Change vs E16 (only this):** `--future 16` instead of 48 (board task kept). 14.6% of missing plies have more than 16 later entries. Cloud A100, 100k steps, 4,980 pos/s (vs 3,694 for E16 on the A10).
- **Validation best top-1:** 57.7 (E16 58.4).
- **Test (507 games, 5,871 missing):**

| | Top-1 | Top-2 | Top-3 | Top-5 | Top-10 | Mean rank | Typed/game | No typing |
|---|---|---|---|---|---|---|---|---|
| E16f16 model | 56.3 | 73.2 | 81.3 | 89.2 | 96.4 | 2.58 | 1.25 | 45% |
| E16f16 + filter | 56.5 | 73.2 | 81.4 | 89.3 | 96.5 | 2.57 | 1.24 | 45% |
| E16f16 + filter + 3-ply search | 57.4 | 74.1 | 81.9 | 89.7 | 96.5 | 2.53 | 1.19 | 47% |
| E16 model | 57.2 | 74.1 | 81.9 | 89.6 | 96.7 | 2.52 | 1.21 | 44% |
| E16 + filter + 3-ply search | 58.4 | 75.0 | 82.4 | 89.8 | 96.8 | 2.48 | 1.18 | 46% |

- **Top-1 by distance (model + filter):** 1: 82.5, 2: 69.6, 3: 54.1, 4–8: 53.0, 9+: 49.8.
- **Reading:** cutting to 16 entries costs about 1 point of top-1/top-2 (within about 1 SE of noise, n=5,871 → SE ≈ 0.65), spread across distances (2 plies −2.5, 4–8 −1.2, 9+ −1.1; 3 plies is +2.3, so the E16 3-ply dip doesn't appear here either). 48 is worth keeping; going beyond 48 (0.7% of plies) is not. Most of the board-task gain is kept with 16 entries, which is useful for an on-device model.

### E17a: bigger model (`runs/c17a_big`, `results/E17a/`)

- **Change vs E16 (only this):** `--d-model 384 --layers 8 --heads 12` (15.25M parameters vs 5.3M). Same 500k games, 100k steps, future 48, board task. Cloud A10 #1, 1,882 pos/s (3.75 h).
- **Validation missing top-1 (E16 / E17a):** 30k 49.3 / 47.9; 50k 53.7 / 52.4; 70k 56.4 / 56.0; 100k 58.4 / 58.3 (E17a top-2 74.8, top-5 90.4).
- **Test (507 games, 5,871 missing):**

| | Top-1 | Top-2 | Top-3 | Top-5 | Top-10 | Mean rank | Typed/game | No typing |
|---|---|---|---|---|---|---|---|---|
| E17a model | 57.3 | 73.9 | 81.9 | 89.8 | 96.7 | 2.52 | 1.18 | 45% |
| E17a + filter | 57.4 | 73.9 | 82.0 | 89.8 | 96.7 | 2.52 | 1.18 | 45% |
| E17a + filter + 3-ply search | 58.5 | 74.8 | 82.7 | 90.0 | 96.7 | 2.48 | 1.16 | 46% |
| E16 + filter + 3-ply search | 58.4 | 75.0 | 82.4 | 89.8 | 96.8 | 2.48 | 1.18 | 46% |

- **Top-1 by distance (model + filter):** 1: 82.7, 2: 71.6, 3: 54.5, 4–8: 53.7, 9+: 50.7 (no 3-ply dip).
- **Reading:** 3× the parameters gives **no gain** (all differences within noise). Model capacity is not the bottleneck at 500k games / 100k steps. Either the data/steps are limiting (E17b tests 2M games), or we are near the information ceiling of the task: many hidden moves are not identifiable from the record (E15). For deployment this is good news: the 5M model is enough.

### E17b: bigger model on 2M games (`runs/c17b_big2m`, `results/E17b/`)

- **Change vs E17a (only this):** `--games data/games_2m.parquet` (2M games vs 500k). Same 15.25M model, 100k steps × 256 (25.6M positions, so each game is seen far less often), future 48, board task. Cloud A100, 3,343 pos/s.
- **Data check:** the first 1,000 val-split games (our validation set) are identical to `games_500k`. None of them, and none of the 200 test games, are in the 2M train split.
- **Validation missing top-1 / top-2 / top-5:** 80k 57.4 / 73.7 / 89.8; 90k 57.7 / 74.4 / 90.0; 100k 58.5 / 75.0 / 90.3 (E17a 58.3 / 74.8 / 90.4; E16 58.4 / – / 90.5).
- **Test (507 games, 5,871 missing):**

| | Top-1 | Top-2 | Top-3 | Top-5 | Top-10 | Mean rank | Typed/game | No typing |
|---|---|---|---|---|---|---|---|---|
| E17b model | 56.4 | 74.2 | 81.9 | 89.7 | 96.7 | 2.52 | 1.19 | 42% |
| E17b + filter | 56.6 | 74.3 | 81.9 | 89.8 | 96.7 | 2.51 | 1.18 | 42% |
| E17b + filter + 3-ply search | 57.6 | 75.1 | 82.6 | 90.1 | 96.7 | 2.47 | 1.14 | 44% |
| E17a + filter + 3-ply search | 58.5 | 74.8 | 82.7 | 90.0 | 96.7 | 2.48 | 1.16 | 46% |
| E16 + filter + 3-ply search | 58.4 | 75.0 | 82.4 | 89.8 | 96.8 | 2.48 | 1.18 | 46% |

- **Top-1 by distance (model + filter):** 1: 82.7, 2: 71.4, 3: 53.2, 4–8: 53.2, 9+: 49.6.
- **Reading:** 4× the games gives **no gain** either (top-2 75.1 vs 74.8 vs 75.0; top-1 differences within noise, SE about 0.65). Neither capacity (E17a) nor data diversity (E17b) moves the plateau at about 58 / 75 / 90. With E15a (order errors, indistinguishable moves), this points to a limit set by the information on the sheet and by move order, not by model or data scale. The 5.3M E16 model stays the default.

### E15a: why does E16 miss? (`scripts/miss_analysis.py`, `results/E15/`)

- **Setup:** E16 model + filter (penalty 6), test set, 5,871 missing plies. For each miss we look at the top suggestion `s` and classify it:
  - **order:** `s` is played later in the same gap by the same side, so we got the order wrong.
  - **equivalent:** replacing the true move by `s` keeps the whole TRUE continuation legal with the same written SANs. The clean sheet can't tell them apart, even with every other missing move known.
  - **other:** the sheet with the true continuation rules `s` out. This is either a model error or `s` would need a different continuation of the other blanks (not checked).
- **Top-1 by kind (% of all missing plies):** hit 57.4, **order 10.9**, equivalent 3.8, other 27.9.
- **Top-2 misses (25.9% of plies):** at least one suggestion is an order swap in 10.5 (40% of top-2 misses), equivalent 2.6, other 12.7.
  - **Set-level top-2:** counting a suggestion as right if the same side plays it somewhere in the gap gives about 84.6%. The model mostly knows *which* moves were played in a gap, and loses points on *when*.
- **By distance (top-1 hit / order / equivalent / other):**

| Distance | n | Hit | Order | Equivalent | Other |
|---|---|---|---|---|---|
| 1 | 709 | 82.2 | 0 | 4.7 | 13.1 |
| 2 | 444 | 72.1 | 0 | 10.1 | 17.8 |
| 3 | 442 | 51.8 | 10.2 | 7.5 | 30.5 |
| 4–8 | 1,809 | 54.2 | 11.0 | 4.4 | 30.3 |
| 9+ | 2,467 | 50.9 | 16.1 | 1.4 | 31.6 |

- **Equivalent-set size k** (optimistic: other missing moves known, clean sheet): k=1 for 74.5%, k=2 for 5.1%, k≥5 for 16.8%. Picking uniformly within the set would give 79.6% top-1.
  - This is a LOOSE upper bound: fixing the true continuation rules out order swaps and alternative continuations. The real ceiling is lower.
- **Reading:**
  1. Order is the largest single, identifiable source of error: about a quarter of top-1 misses and 40% of top-2 misses, growing with gap length.
  2. Truly indistinguishable moves are a small share (about 4%), mostly 1–3 plies before a written move: quiet moves that don't affect it.
  3. 'Other' (about 28%) includes real model errors and alternatives that need different blanks. Splitting these needs a search over the other blanks (E15 proper).
- **Ideas from this:** a UI step that orders a gap's moves (the model proposes the set, the player orders it); a set-level metric; model work on ordering cues (tempo, checks/replies, piece paths).

### E19a: Maia-3 baseline and zero-training combination (`scripts/maia_eval.py`, `results/E19a/`)

- **Maia-3** (CSSLab, ICLR 2026, "Chessformer"; AGPL-3.0): a human-move transformer (64 square tokens, geometric attention bias, from×to head, last 8 positions + both ratings).
  - Trained on Lichess games Jan 2023 – Jul 2025.
  - Our train and test games are Aug–Sep 2026, so there is no overlap.
  - We use the 5M model (`UofTCSSLab/Maia3-5M`, 21 MB), pinned to maia3 commit `1e13597`.
- **Setting:** same test set and assisted setting (earlier moves confirmed, so Maia gets the true history and ratings). Maia sees nothing from the scoresheet.
- **Combination:** E16 log-prob + λ × Maia log-prob − 6 × violations. λ is tuned by 2-fold cross-fitting over games (tune on even games, test on odd, and vice versa). λ picked: 2.0 and 1.0. The full-grid curve is flat for λ 1–2.
- **Test (5,871 missing plies):**

| | Top-1 | Top-2 | Top-3 | Top-5 | Mean rank |
|---|---|---|---|---|---|
| E16 + filter | 57.4 | 74.1 | 81.9 | 89.6 | 2.52 |
| Maia-3 alone (no scoresheet) | 54.3 | 73.9 | 82.1 | 90.9 | 2.41 |
| Maia-3 + filter | 56.7 | 75.4 | 83.3 | 91.6 | 2.32 |
| **E16 + λ·Maia-3 + filter (cross-fitted)** | **61.9** | **78.2** | **85.5** | **92.5** | **2.16** |

- **Top-1 by distance:**

| | 1 | 2 | 3 | 4–8 | 9+ |
|---|---|---|---|---|---|
| E16 | 82.2 | 72.1 | 51.8 | 54.2 | 50.9 |
| Maia | 57.3 | 51.4 | 52.0 | 53.2 | 55.2 |
| Combo | 84.3 | 72.5 | 58.8 | 57.9 | 56.9 |

- **Reading:**
  1. A strong human-move model with NO scoresheet input nearly matches our model and beats it far from written moves (55.2 vs 50.9 at 9+). Our model's human-play prior is weak. E17a/E17b (bigger model, 4× games) did not close that gap, but Maia-3's much larger training set and history input did.
  2. The two are complementary: ours wins near written moves (sheet signal), Maia wins far from them (prior). The zero-training sum gives **+4.5 top-1 and +4.1 top-2**: the best result so far, and the first real jump since E13.
  3. This is a clean baseline for the paper ("human-move model without the sheet").
- **Maia-3 23M** (`results/E19a/maia23m.*`): alone 55.6 / 74.9 / 91.3; + filter 57.9 / 76.2 / 91.8. **E16 + λ·Maia-23M + filter (cross-fitted, λ=1.5 both folds): 62.7 / 79.4 / 92.9**, about +0.8 / +1.2 over the 5M Maia.
  4. Next: E19b, the hybrid (Maia-3 backbone + gated cross-attention to our scoresheet tokens + board task), fine-tuned on the cloud. It should beat the plain sum, since the sheet can then condition the prior instead of only being added to it.

### E19b: hybrid, E16 + Maia-3 5M with gated cross-attention (`src/recon/model/hybrid.py`, `runs/c19b_hybrid`)

- **Model** (15.2M params; 4.7M new):
  - **Our tower:** E16, initialized from `runs/c16_future/best.pt`.
  - **Maia tower:** Maia-3 5M, pretrained.
  - **Fusion:** after each of Maia's 8 blocks, a cross-attention lets the 64 squares read all of our tower's final tokens (board, notation, 48 later entries). It also has a small feed-forward. Both outputs go through tanh gates set to 0 at init.
  - **Move logits:** ours + λ × Maia log-softmax over legal from-to moves, with λ learned (init 1.5).
  - **Other heads:** promo, align, gap and the future-board heads stay on our tower.
- **Sanity checks:**
  - At init, the model equals E16 + 1.5 × Maia exactly (checked on val batches).
  - The untrained checkpoint scores 62.1 / 78.4 / 92.8 on the test set through `assisted_eval` (penalty 6), the same as the E19a λ=1.5 sweep.
- **Inputs:** Maia gets the last 8 positions (Maia's frame, padded with the earliest) and both ratings (1500 when the rating is dropped). Built in the data stream (`DataConfig.maia_inputs`); checked identical to `maia3.dataset.tokenize_board`.
- **Training:** 500k games, 50k steps × 256.
  - New parts lr 5e-4 (cosine, warmup 500).
  - Pretrained parts (both towers) lr × 0.1.
  - bf16, compiled, A100, about 2,000–2,400 pos/s.
- **Eval:** model and model + filter. The 3-ply search is not yet adapted to the hybrid's extra inputs.
- **Validation (missing top-1 / top-2 / top-5):**

| Step | Top-1 | Top-2 | Top-5 |
|---|---|---|---|
| 5k | 63.4 | 79.6 | 93.4 |
| 20k | 63.8 | 80.0 | 93.8 |
| 30k | 64.2 | 80.9 | 93.8 |
| 40k | 64.9 | 81.0 | 94.2 |
| 50k | **65.0** | **81.2** | 94.1 |

  E16 final: 58.4 / – / 90.5.
- **Training time:** 4,431 s (1.2 h), 2,890 pos/s at steady state.
- **Test (507 games, 5,871 missing):**

| | Top-1 | Top-2 | Top-3 | Top-5 | Top-10 | Mean rank | Typed/game | No typing |
|---|---|---|---|---|---|---|---|---|
| E16 + filter + 3-ply search | 58.4 | 75.0 | 82.4 | 89.8 | 96.8 | 2.48 | 1.18 | 46% |
| Hybrid untrained (= E16 + 1.5·Maia) + filter | 62.1 | 78.4 | 85.5 | 92.8 | 98.3 | 2.14 | 0.84 | 54% |
| **E19b hybrid** | **63.4** | **79.9** | **87.3** | **93.4** | **98.3** | **2.07** | **0.76** | **57%** |
| E19b hybrid + filter | 63.4 | 79.9 | 87.3 | 93.4 | 98.3 | 2.07 | 0.76 | 58% |

- **Top-1 / top-2 by distance:** 1: 85.9 / 94.8; 2: 77.5 / 88.5; 3: 61.5 / 76.2; 4–8: 59.5 / 79.0; 9+: 57.6 / 75.2.
- **Reading:**
  1. **New best:** +5.0 top-1 and +4.9 top-2 over E16 + filter + search, with no search. Fine-tuning adds +1.3 / +1.5 over the zero-training sum.
  2. Typed moves per game drop from 1.18 to 0.76 (−36%). Games needing no typing rise from 46% to 57%.
  3. Gains at every distance, largest far from written moves (9+: 50.9 → 57.6 top-1).
  4. The feasibility filter adds nothing now: the model has absorbed it.
  5. Validation was still rising slowly at 50k.
- **+ 3-ply search** (filter penalty 6, 10 candidates × width 8; `beam._score` now feeds Maia's inputs from each search state's move stack). Run on the Mac (MPS), 0.26 s/suggestion:
  - **Overall: 64.0 / 80.7 / 87.8 / 93.7** (top-1/2/3/5), mean rank 2.04, **0.72 typed/game, 59% no typing**. +0.6 top-1 and +0.8 top-2 over the hybrid alone.
  - **By distance (top-1 / top-2):** 1: 87.0 / 95.3; 2: 80.6 / 91.2; 3: 64.3 / 83.3; 4–8: 59.5 / 79.2; 9+: 57.6 / 75.3 (beyond 3 plies the search is off).
  - **First result over 80% test top-2.** Caveat: the search settings (max-dist 3, penalty 6) were chosen on this test set in E11. Re-confirm on a dev set before the paper.

### E19c: hybrid with Maia-3 23M (`runs/c19c_hybrid23m`)

- **Setup:** identical to E19b, but the Maia tower is Maia-3 23M (46M params total). 50k steps × 256, A100, 5,622 s (1.6 h), about 2,280 pos/s.
- **Validation (missing top-1 / top-2 / top-5):**

| Step | Top-1 | Top-2 | Top-5 |
|---|---|---|---|
| 5k | 64.4 | 80.4 | 93.8 |
| 20k | 64.2 | 80.8 | 94.3 |
| 30k | 64.9 | 81.2 | 94.4 |
| 40k | 65.6 | 81.6 | 94.5 |
| 50k | **65.9** | **81.9** | **94.6** |

  E19b at 50k: 65.0 / 81.2 / 94.1.
- **Test (507 games, 5,871 missing), no search:**

| | Top-1 | Top-2 | Top-3 | Top-5 | Top-10 | Mean rank | Typed/game | No typing |
|---|---|---|---|---|---|---|---|---|
| E19b (5M) | 63.4 | 79.9 | 87.3 | 93.4 | 98.3 | 2.07 | 0.76 | 57% |
| **E19c (23M)** | **64.4** | **80.9** | **87.5** | **93.6** | **98.4** | **2.04** | **0.74** | **58%** |
| E19c + filter | 64.3 | 80.9 | 87.5 | 93.6 | 98.4 | 2.04 | 0.74 | 58% |

- **Reading:**
  1. The bigger Maia tower adds +1.0 top-1 and +1.0 top-2. This is the same gap as Maia 23M vs 5M alone (E19a: +0.8 / +1.2), so the hybrid passes the prior's gain through.
  2. Top-2 crosses 80% without search for the first time.
  3. Validation was still rising at 45–50k, so a longer run may help a little more.
- **+ 3-ply search** (same settings as E19b; Mac MPS, 0.40 s/suggestion): **65.0 / 81.7 / 88.3 / 93.9** (top-1/2/3/5), mean rank 2.00, **0.70 typed/game, 60% no typing**. New best: +1.0 top-1 / +1.0 top-2 over E19b + search. The search adds +0.6 / +0.8, as it did for E19b.
- **Top-1 / top-2 by distance (with search):** 1: 87.3 / 94.8 ; 2: 80.6 / 92.6 ; 3: 65.8 / 84.8 ; 4-8: 60.3 / 79.8 ; 9+: 59.1 / 76.7.

### E20: E19c + move history in our tower (`runs/c20_hist`)

- **Question:** Maia-3 found board history helps a strong model. E10 (history on the weak v1 model, 30k steps, together with the move number) was negative. Does history help our scoresheet tower once it is strong and fused with Maia?
- **Single change from E19c:** our tower gets 8 tokens for the last 8 moves (`--history 8`, the E10 encoding: piece, from, to, capture, plies ago). The moves plus the current board carry the same information as 8 past boards. The Maia tower already sees 8 boards, as in E19c.
- **Init:** as E19c (E16 + Maia-3 23M). The new history parameters start fresh and train at the full lr; pretrained parts at 0.1×.
- **Start check** (60 val games, 5M Maia, untrained): without history 61.6 / 78.6 / 92.0 (top-1/2/5), loss 0.558; with history 61.6 / 78.4 / 92.5, loss 0.596. So it starts almost exactly where E19c started.
- **Run:** 50k steps × 256, A100.
- **Validation (missing top-1 / top-2 / top-5):**

| Step | E20 | E19c |
|---|---|---|
| 5k | 64.2 / 80.5 / 93.9 | 64.4 / 80.4 / 93.8 |
| 25k | 64.8 / 81.4 / 94.3 | 65.1 / 81.2 / 94.3 |
| 50k | 66.0 / 81.9 / 94.6 | 65.9 / 81.9 / 94.6 |

- **Test (507 games, 5,871 missing), no search:**

| | Top-1 | Top-2 | Top-3 | Top-5 | Typed/game | No typing |
|---|---|---|---|---|---|---|
| E19c | 64.4 | 80.9 | 87.5 | 93.6 | 0.74 | 58% |
| E20 | 64.2 | 80.9 | 87.6 | 93.7 | 0.73 | 58% |
| E20 + filter | 64.3 | 80.8 | 87.7 | 93.8 | 0.72 | 59% |

- **Training time:** 5,744 s.
- **Outcome: null.** Every difference is within ±0.2, which is noise. Move history in our tower adds nothing once the Maia tower (which already sees 8 boards) is fused in. History is covered; this direction is closed. E19c stays the model of record. The search eval was skipped, since there is nothing to gain.
- **Check that history reaches the model at inference:** in both the data stream and `assisted_eval`, boards are played forward, so `encode_history` reads the real move stack (verified on a sample position; the checkpoint has `hist_moves=8`).
- **History ablation at inference** (300 val games, 2,546 hidden plies): with history 67.1 / 82.5 / 94.9 (top-1/2/5), loss 0.4118; history erased 67.0 / 82.4 / 94.9, loss 0.4142. The model gets the history but has learned to almost ignore it: the Maia tower's 8 boards already carry this information.

### E21: dev set and search re-tuning (`bench/dev.jsonl`, `scripts/tune_dev.sh`, `results/E21/`)

- **Why:** the search settings (filter penalty 6, 10 candidates × width 8, search within 3 plies) were chosen on the test set in E11, with the much weaker v1 model. The hybrid's λ is learned in training, so it needs no tuning.
- **Dev set:** `recon build-bench --games data/games.parquet --split val --num-games 200 --out bench/dev.jsonl`.
  - Same 6 presets and seed as the test set.
  - 5,787 missing plies in the 3 primary presets (test: 5,871).
  - These 489 `games.parquet` val games appear in none of the training files (500k, 2M) and not in the 500k val set used for checkpoint selection.
- **Sweep:**
  - model and model + filter;
  - search within 8 plies at 5 / 10 / 16 candidates (width 8);
  - then the search cutoff (1–8 plies) and candidate count chosen on dev from the saved ranks (`BENCH=bench/dev.jsonl scripts/combine_cutoff.py`).
- **Protocol:** freeze the chosen setting, then report test once.
- **Model alone on dev:** 64.7 / 80.8 / 87.6 / 94.2 (top-1/2/3/5), 0.66 typed/game, 59% no typing. Test: 64.4 / 80.9 / 87.5 / 93.6. The test result is not a lucky draw.
- **Search:** only 10 candidates × width 8 within 8 plies was run (about 1.5 h per setting on the Mac). The 5- and 16-candidate runs were dropped. The cutoff is then chosen on dev.

### E29: real scoresheets from HCS (`scripts/hcs_games.py`, `hcs_reconstruct.py`, `hcs_bench.py`; `bench/hcs.jsonl`; `results/E29/`)

- **Data:** the HCS dataset (Eicher et al. 2021), from the authors' Google Drive ("HCS Dataset December 2021.zip", 1.38 GB, in `data/hcs/`).
  - 158 games, 215 phone photos of real sheets.
  - The labels are per move box: **what the player wrote**, slips included.
  - A game's two pages are the **two players' copies** (48 games have both).
  - The "testing" labels repeat training games.
- **Real sheets are messy:** only 13 of 224 sheets replay legally as written. The median sheet becomes illegal after 17% of its moves.
  - Kinds of mess: one-character slips (c↔e, f↔g, 3↔4↔5, R↔K/B/Q are the most common), short forms ("cd" for cxd4), typos ("Bhd4"), skipped or shifted rows, and a set-up start position (game 039).
- **Recovering the true game:** beam search (width 64) over legal games. Each written entry costs the edit distance to the played move's SAN; a blank costs 1.
  - Substitutions only, so row shifts come out expensive.
  - Result: per-sheet share of differing entries, quartiles 4.9% / 12.6% / 28.6%. The top quartile is mostly recovery failures (shifts, short forms), not true slip rates.
- **Benchmark:** 80 sheets (one per game), keeping standard start, at least 30 plies and at most 15% of entries differing. The gaps are the `missing` preset's bursts.
  - `hcs_real`: tokens are the real written text. **5.6% of the visible entries are real slips**, inside our simulator's 4.6–9.5% range for wrong entries.
  - `hcs_clean`: same gaps, correct text.
  - No ratings (Maia gets 1500).
- **Results (1,262 missing plies, filter penalty 6, no search):**

| | Top-1 | Top-2 | Top-3 | Top-5 | Typed/game | No typing |
|---|---|---|---|---|---|---|
| E16, real text | 47.7 | 63.5 | 72.5 | 81.5 | 2.92 | 20% |
| E16, correct text | 48.2 | 63.5 | 72.8 | 81.7 | 2.86 | 22% |
| **E19c, real text** | **55.5** | **71.6** | **79.6** | **87.4** | **1.99** | **31%** |
| E19c, correct text | 56.5 | 72.0 | 79.6 | 87.7 | 1.93 | 30% |
| E19c, real text, ratings set to 1900 | 57.0 | 72.5 | 80.2 | 87.6 | 1.96 | 30% |
| E19c, real text, ratings set to 2200 | 56.0 | 72.3 | 80.6 | 87.5 | 1.98 | 34% |

- **By gap size (E19c real vs simulated test, top-1 / top-2):**

| Gap | HCS real | Simulated test |
|---|---|---|
| 1–4 | 80.8 / 84.6 (n=26) | 78.8 / 89.8 |
| 5–9 | 55.4 / 73.4 | 69.0 / 84.6 |
| 10–19 | 53.2 / 68.4 | 62.5 / 79.6 |
| 20+ | 56.2 / 73.2 | 62.2 / 79.3 |

- **Reading:**
  1. Real slips cost only about 1 point; the model handles real player errors.
  2. The hybrid's gain over E16 holds on real sheets: +7.8 top-1, +8.1 top-2.
  3. **Real games are about 9 points harder than simulated ones**, in every gap bucket, so it is not the gap mix. Missing ratings explain about 1 point. The rest is likely domain shift: over-the-board (probably slower) games versus Lichess online blitz/rapid, which both our training data and Maia-3 come from.
  4. **Caveats:** 80 games (game-clustered SE about 2–3 points). The gaps are simulated. The truth is recovered automatically, so a few truth moves may be wrong.
- **Next:**
  - Fine-tune on over-the-board-like games: Lichess classical/rapid, or an OTB database such as TWIC or Lumbras.
  - Model row shifts in the recovery to keep more sheets.
  - Calibrate the simulator's slip confusions to the HCS confusion table.

### E30: real over-the-board games, simulated sheets (`scripts/broadcast_bench.py`, `bench/broadcast.jsonl`, `results/E30/`)

- **Question:** is E29's drop on real sheets caused by over-the-board play (domain shift)? This uses real tournament games with our usual simulator, so the sheet noise is identical to the test set.
- **Data:** the Lichess broadcast database (relayed OTB tournament games), `lichess_db_broadcast_2026-09.pgn.zst` (18 MB, 28,638 games).
  - **License: CC BY-SA 4.0** (not CC0 like the main database): attribution and share-alike if we release derived data.
  - Kept: standard start, at least 30 plies, deduplicated.
  - Time controls are free text, mostly in minutes; parsed by `tc()`. By speed: 9,814 classical, 3,042 rapid, 1,757 blitz, 12,099 with no time control.
  - 84% rated (mostly FIDE), median mean rating 2047, passed to the model as if Lichess ratings.
- **Benchmark:** 150 games per speed × the 3 primary presets, same simulator and seed as `bench/test.jsonl`.
- **Results (penalty 6, no search; top-1 / top-2 / top-5):**

| | Blitz OTB | Rapid OTB | Classical OTB | All |
|---|---|---|---|---|
| Missing plies | 4,457 | 4,714 | 4,898 | 14,069 |
| E16 | 54.3 / 71.6 / 88.1 | 53.2 / 70.4 / 88.5 | 48.7 / 64.4 / 83.7 | 52.0 / 68.7 / 86.7 |
| **E19c** | **65.1 / 81.2 / 94.5** | **64.0 / 80.8 / 94.0** | **58.6 / 75.2 / 91.8** | **62.5 / 79.0 / 93.4** |

  Lichess simulated test, for reference: E19c 64.4 / 80.9 / 93.6.
- **Reading:**
  1. **Over-the-board play by itself is not the problem:** OTB blitz and rapid score the same as Lichess online games.
  2. **Classical (slow) games are the shift:** −6.5 top-1 and −5.7 top-2 against blitz. Our training data is 1% classical. Slow play is more accurate and further from the online-blitz habits that both our tower and Maia-3 learned.
  3. This explains most of E29's 9-point drop on HCS (tournament sheets are typically classical). The rest is likely HCS-specific: unknown players, recovered truth, and 80 games.
  4. Maia's prior transfers: E19c's gain over E16 is larger on OTB games (+10.5 top-1) than on Lichess (+6.6).
- **Next:** fine-tune E19c on classical games, from broadcast classical games (about 10k/month, several years available) plus Lichess classical. Evaluate on E30 classical and E29 HCS.

### E31: fine-tune E19c on over-the-board games (`scripts/broadcast_games.py`, `runs/c31_otb`)

- **Why:** E30 shows classical games cost about 6 points, and our training data is 1% classical.
- **Data:** `data/games_otb_mix.parquet`.
  - **Broadcast games:** every Lichess broadcast month 2020-01..2026-08 (80 files, 736 MB, CC BY-SA 4.0). Standard start, at least 30 plies, deduplicated by move sequence. Every game also present in the 2026-09 evaluation month is dropped.
  - **Plus 400k Lichess training games** (from `games_500k`), so online play is not forgotten.
  - Id-hash splits as usual.
- **Training:**
  - Start from the whole E19c checkpoint (new `--init-full`).
  - 20k steps × 256; lr 2e-4 for the fusion layers and λ, × 0.1 for both towers.
  - A100.
- **Evaluation:** E30 classical / rapid / blitz, E29 HCS (real and clean), and the Lichess test set, to check nothing is lost.
- **Cutoff chosen on dev** (10 × 8 search within N plies, model + filter beyond; exact from saved ranks):

| Search within | Top-1 | Top-2 | Top-3 | Top-5 | Typed/game | No typing |
|---|---|---|---|---|---|---|
| 0 (model + filter) | 64.7 | 80.8 | 87.6 | 94.2 | 0.66 | 59% |
| 1 | 64.9 | 80.9 | 87.7 | 94.3 | 0.66 | 60% |
| 3 (old setting, from E11) | 65.5 | 81.3 | 88.1 | 94.4 | 0.64 | 61% |
| 5 | 66.7 | 81.9 | 88.6 | 94.7 | 0.61 | 63% |
| **6** | **66.9** | **82.3** | **88.9** | **94.9** | **0.59** | **64%** |
| 8 | 67.1 | 82.1 | 88.8 | 94.6 | 0.62 | 62% |

- **Frozen setting: filter (penalty 6) + 10 × 8 search within 6 plies.**
  - With the strong hybrid, search keeps helping further from the written move than with v1 (where 3 was best).
  - Cost: search at 8 plies averages 2.3 s per suggestion on the Mac (MPS), so 6 plies is slow for a phone. Cutoff 3 is the fast option (−0.9 top-2).
  - The 5- and 16-candidate runs were not done.
- **Test, run once with the frozen setting** (`results/E21/test_c10_d6*`): **66.4 / 82.6 / 88.9 / 94.3** (top-1/2/3/5), mean rank 1.96, 0.66 typed/game, 61% no typing, 0.64 s/suggestion on average.
  - By distance (top-1 / top-2): 1: 87.3 / 94.8; 2: 80.6 / 92.6; 3: 65.8 / 84.8; 4–8: 64.8 / 82.8; 9+: 59.1 / 76.7.
  - **This is the paper's headline number:** the setting was chosen on dev only. The old E19c + search (cutoff 3, chosen on test) was 65.0 / 81.7 / 93.9.
- **Run:** 20k steps, 37 min on the A100 (started 00:09 UTC, checkpoint pulled 00:51). The instance was then terminated automatically through the Lambda API (`scripts/cloud/terminate_after.sh`).
- **Validation (OTB + Lichess mix):**

| Step | Top-1 | Top-2 | Top-5 |
|---|---|---|---|
| 5k | 64.0 | 80.3 | 94.9 |
| 10k | 64.7 | 81.0 | 95.1 |
| 15k | 65.1 | 81.5 | 95.0 |
| 20k | 65.8 | 81.8 | 95.2 |

  Still rising at the end.
- **Results (penalty 6, no search; top-1 / top-2 / top-5):**

| | E19c | E31 | Change (top-1 / top-2) |
|---|---|---|---|
| OTB blitz (4,457) | 65.1 / 81.2 / 94.5 | 65.6 / 81.7 / 95.1 | +0.5 / +0.5 |
| OTB rapid (4,714) | 64.0 / 80.8 / 94.0 | 65.6 / 81.9 / 94.4 | +1.6 / +1.1 |
| **OTB classical (4,898)** | 58.6 / 75.2 / 91.8 | **61.5 / 77.8 / 93.0** | **+2.9 / +2.6** |
| **HCS, real text (1,262)** | 55.5 / 71.6 / 87.4 | **56.8 / 74.2 / 88.2** | **+1.3 / +2.6** |
| HCS, correct text (1,252) | 56.5 / 72.0 / 87.7 | 57.6 / 74.7 / 88.6 | +1.1 / +2.7 |
| Lichess test (5,871) | 64.4 / 80.9 / 93.6 | 64.0 / 80.7 / 93.8 | −0.4 / −0.2 |
| Lichess dev (5,787) | 64.7 / 80.8 / 94.2 | 64.9 / 81.3 / 94.2 | +0.2 / +0.5 |

- **Reading:**
  1. Fine-tuning on real OTB games closes about 40% of the classical gap (blitz − classical: 6.5 → 4.1 top-1) and lifts real HCS sheets by 2.6 top-2.
  2. Online play is unchanged: test −0.4 / −0.2 and dev +0.2 / +0.5 are within noise (SE about 0.6).
  3. Validation was still rising, so a longer run (or more classical weight) may close more of the gap.
  4. **E31 becomes the model of record for real use:** it is equal on online games and better on OTB.
  5. Search with E31 has not been run yet.

### E15b: miss analysis on the hybrid (`scripts/miss_analysis.py --ckpt runs/c19c_hybrid23m/best.pt`, `results/E15b/`)

- **Change:** the script now feeds Maia's inputs, so it works with hybrid models.
- **Top-1 (% of plies):** hit 64.3, order 9.2, equivalent 3.3, other 23.2. E16 was 57.4 / 10.9 / 3.8 / 27.9.
- **Top-2 misses (19.1% of plies):** at least one order swap 7.5, equivalent 2.1, other 9.5.
- **Order errors by distance:** 0 at 1–2 plies, 7.7 at 3, 9.8 at 4–8, 13.4 at 9+.
- **Identifiability** (true continuation known):
  - 74.5% of missing plies are uniquely determined by the clean sheet.
  - Picking uniformly among the sheet-consistent moves gives 79.6% top-1. This is not a ceiling: a prior can choose better within the set.
  - The equivalent set fits in the top 2 for 79.5% of plies.
- **Reading:** the hybrid fixed mostly "other" errors (27.9 → 23.2); order errors barely moved (10.9 → 9.2). Ordering is now the clearest remaining target (whole-gap ranking).
- **E31 + dev-tuned search** (filter 6, 10 × 8 within 6 plies; `results/E31/search_*`):

| | Top-1 | Top-2 | Top-5 | Typed/game | No typing |
|---|---|---|---|---|---|
| Lichess test | 66.4 [64.8, 68.1] | 82.6 [81.2, 83.9] | 94.3 | 0.66 | 61% |
| HCS real sheets | **58.8** [55.6, 62.0] | **75.0** [71.5, 78.2] | 88.2 | 1.86 | 32% |
| OTB all | 65.8 | 81.8 | 94.5 | 0.68 | 61% |
| OTB blitz | 67.6 | 83.1 | 95.5 | – | – |
| OTB rapid | 67.2 | 83.2 | 94.7 | – | – |
| OTB classical | **62.8** | **79.1** | 93.5 | – | – |

  Brackets are game-clustered 95% intervals.
- **Reading:**
  1. On the Lichess test set, E31 + search equals E19c + search exactly (66.4 / 82.6 / 94.3).
  2. On real data it is clearly better: HCS 55.5 / 71.6 (E19c, no search) → 58.8 / 75.0; OTB classical 58.6 / 75.2 → 62.8 / 79.1.
  3. **E31 + search is the single model of record.**

### E31-gates: how much does the gated cross-attention contribute? (analysis of `runs/c31_otb`)

- **Learned gates:** $\tanh(g)$ stays tiny in every block (|value| ≤ 0.007). Measured on a validation batch, though, the fusion changes Maia's square states by about **10% (block 1)** and **8.5% (block 7)** in norm, and by about 0 elsewhere: the attention and feed-forward outputs grew to compensate.
- **Ablation at test time** (all gates set to 0; 300 val games, 2,546 hidden plies): 66.4 / 81.8 / 94.8 → 65.2 / 81.2 / 94.2 (top-1/2/5). Loss 0.419 → 0.433.
- **Reading:**
  1. The cross-attention is worth about **1 point top-1** (0.6 top-2).
  2. Most of the hybrid's gain over the scoresheet tower comes from adding Maia's prior (the untrained sum: 58.4 → 62.1 / 62.7). Fine-tuning, the bigger prior and search add the rest.
  3. The paper was corrected accordingly (it had credited the fusion with the whole gain).

### E32: characters vs whole-entry tokens (`--whole-notes`, `runs/c16_whole`)

- **Question:** does reading written entries character by character matter, or would whole-entry tokens (as in move-sequence models) do as well? This decides whether the character encoding can be claimed as a contribution (robustness to noisy notation) or is just a design choice.
- **Single change from E16 (c16_future):**
  - The text vector of every notation and future token is one learned embedding of the exact written string, hashed into 32,768 buckets (`NotationEncoder.text`).
  - There is no character sharing, so "Rxe8+" and a misread "Rxc8+" are unrelated tokens.
  - Parsed fields (piece, destination, …), offsets, the board task, data, steps and seed are unchanged.
  - The embedding table adds 8.4M parameters; compute is unchanged.
- **Evaluation (planned):**
  - Test set: model + filter, by preset (`mixed_ocr` is the noisiest).
  - Plies whose next written entry is wrong (ocr or slip).
  - HCS real and clean.
  - Compared against E16's existing results.
- **Run:** A100 (launched through the API after approval), 100k steps.
  - The new instance's driver supports only CUDA 12.8, so the locked PyTorch (cu132) saw no GPU. Fixed on the instance with PyTorch 2.11.0+cu128 and `UV_NO_SYNC=1`.
  - Pulled 00:26; the instance was terminated automatically.
- **Results (model + filter, test set; top-1 / top-2; chars = E16, whole = E32):**

| Subset | Plies | Characters | Whole tokens | Chars − whole, top-1 [95% CI] |
|---|---|---|---|---|
| All | 5,871 | 57.4 / 74.2 | 56.9 / 74.0 | +0.5 |
| missing | 2,509 | 57.3 / 73.9 | 57.1 / 74.1 | +0.2 [−1.3, +1.5] |
| mixed | 1,697 | 59.3 / 75.8 | 58.0 / 75.3 | +1.2 [−0.3, +2.8] |
| mixed_ocr | 1,665 | 55.6 / 73.0 | 55.4 / 72.5 | +0.1 [−1.5, +1.8] |
| Next written entry correct | 5,605 | 57.6 / 74.3 | 57.3 / 74.3 | +0.4 [−0.8, +1.4] |
| **Next written entry wrong (misread or slip)** | 266 | 52.3 / 71.8 | 49.6 / 67.7 | **+2.6 [−1.4, +7.0]**, top-2 +4.1 |
| HCS real (no search) | 1,262 | 47.7 / 63.5 | 47.0 / 61.7 | +0.7, top-2 +1.8 |
| HCS correct text | 1,252 | 48.2 / 63.5 | 47.7 / 62.9 | +0.5 |

- **Reading:**
  1. Overall, whole-entry tokens do as well as characters. A likely reason is that future tokens also carry parsed fields (piece, destination, …), but removing the parsed fields was **not** tested, so this is a hypothesis.
  2. Where it should matter (just before a wrong written entry, and on real sheets), characters are ahead (+2.6 / +4.1 and +0.7 / +1.8), but the intervals include zero (266 plies, 80 games).
  3. **Verdict:** the character encoding is a reasonable design choice with a small, unconfirmed robustness benefit, not a contribution. The paper states it this way.
