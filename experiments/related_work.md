# Related work (survey of 2026-11-25)

## Closest: Zugwise (Trippen, arXiv 2611.14544, Oct 2026, preprint)

- **Task:** digitizing real handwritten scoresheets: photo → grid detection → BiLSTM+CTC OCR (on HCS) → game-level reconstruction with an operator confirming flagged moves. It also merges both players' sheets (Needleman-Wunsch alignment).
- **Ranking:** candidates from the OCR beam, scored by hand-built signals, CTC re-scoring and its own Maia-style CNN prior (2.4M params; lichess games rated 1900–2300). No engine-quality signals, by design.
- **Headline:** on 1,290 replacement decisions in 45 real club games, where the correct move is already an OCR candidate, top-1 rises from 58.8% (hand-built) to 78.0% (+CTC +prior), and top-3 reaches 93.8%. Out of sample (37 games): 80.6% top-1.
- **Missing moves:** one ply at a time, by insertion at a stuck position. A legal interior omission is not recovered. When the correct move is in no candidate list, no search can find it; the operator must supply it (43/45 games needed at least one such correction).
- **Release:** app under AGPL; OCR and prior as ONNX models on Hugging Face. Images not released; no public benchmark.
- **Relation to us:**
  - Complementary. They solve misreadings of *written* moves on real images. We solve moves that are *not written* (multi-ply gaps), plus noisy entries, with a learned model that reads the rest of the sheet and a strong pretrained prior (Maia-3).
  - Their "candidate-list bound" is exactly our task.
  - Their numbers are not comparable to ours (different task and data).

## Scoresheet recognition (OCR)

- **Eicher, Farmer, Li, Majid:** ICDAR 2021 workshops (CBDAR), convolutional BiLSTM.
  - **HCS dataset:** 158 games, 215 pages, public (https://sites.google.com/view/chess-scoresheet-dataset).
  - **Journal version:** Majid & Eicher, *Journal of Imaging* 8(2), 2022. Legality post-processing; semi-autonomous mode asks the user.
- **Hayashi & Hirata:** attention encoder-decoder on scoresheets, ICPR 2022 (arXiv 2406.06538).
- **Zhu:** RNN digitization, IEEE MIT URTC 2024.
- **Kostka & Fischer:** TrOCR web app, 2025 (about 45% CER).
- **Student theses:** ZHAW (Horváth & Dreher 2020 "ChessReader"; Hostettler & Boner 2023, a 62-game corpus) and Kochańska 2017.
- **Baird & Thompson:** "Reading chess", TPAMI 1990 (printed text).

## Missing-move filling

- **Narayanan:** blog, 2019 (saumikn.com/blog/finding-missing-chess-moves).
  - Method: enumerate legal moves at each gap and keep the branches that replay the rest legally.
  - No ranking and no evaluation; cost is exponential in gap length.
  - This is our legality-only baseline idea.

## Human-move models (our prior)

- **Maia:** McIlroy-Young et al., KDD 2020.
- **Maia-2:** NeurIPS 2024.
- **Maia-3 / Chessformer:** ICLR 2026, arXiv 2605.19091.
- **Allie:** ICLR 2025.
- **Board-free move-sequence transformer:** Li & Jiang, arXiv 2603.29761, 2026; 55.2% top-1.

## Masked / gap prediction in chess sequences

- **RePAIR:** Koller, Fürnkranz, Bertram, IEEE CoG 2026, arXiv 2606.11860.
  - JEPA-style prediction of masked positions in latent space; representation learning.
  - No legal move recovery and no scoresheet task.

## Takeaways for the paper

1. No prior work (found) defines or benchmarks reconstruction of *unwritten* multi-ply gaps, or releases a benchmark for it. Contribution 1 holds, framed as complementary to OCR work, especially Zugwise.
2. The real-data gap is fillable: **HCS is public (158 real games with transcriptions)**. We can evaluate on real sheets by blanking moves in HCS transcriptions (real games and handwriting-derived noise), or by feeding a real OCR front end (Zugwise's ONNX recognizer, AGPL).
3. Baselines to include:
   - legality-only enumeration (Narayanan-style);
   - prior-only ranking (Maia-3, the analogue of Zugwise's prior);
   - ours.
