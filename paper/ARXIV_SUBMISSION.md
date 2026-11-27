# arXiv submission: what to enter

**Upload file:** `paper/arxiv_upload.zip`. It contains `main.tex`, `main.bbl`, `fig_task.tex`, `fig_arch.tex` and `figs/by_distance.pdf`. It was tested to compile on its own: 12 pages, no missing references.

| Form field | Value |
|---|---|
| **Primary category** | Computer Science > Artificial Intelligence (cs.AI) |
| **Cross-list (optional)** | cs.LG (Machine Learning) |
| **License** | arXiv.org perpetual, non-exclusive license (most flexible if you later submit to a journal or conference that takes copyright). CC BY 4.0 is the alternative if you want maximum reuse. |
| **Title** | Reading Between the Lines: Recovering Unrecorded Moves from Chess Scoresheets |
| **Authors** | Subhash Bylaiah |
| **Comments** | 12 pages, 3 figures, 5 tables. Code and benchmark: https://github.com/subhashbylaiah/chess-game-reconstruction |
| **Report number, journal reference, DOI, MSC/ACM classes** | leave empty |

**Abstract (plain text, paste as is):**

Over-the-board chess games are recorded by hand, and the records are often incomplete: players stop writing in time trouble, skip moves, and make slips. Existing scoresheet tools recognise what was written and flag illegal moves, but leave unwritten moves for the user to type. We study reconstructing moves that were never written, in an interactive setting where a user confirms one move at a time from a short list of suggestions. We release a benchmark built from public games, with a scoresheet simulator and user-facing metrics. Our model reads the board and the rest of the scoresheet, is trained to predict the board at the next written move, and is combined with a pretrained human-move model (Maia-3). It ranks the true move first 66.4% of the time and in the top two 82.6% of the time; a user types 0.66 moves per game, and 61% of games need no typing. The prior alone ranks the true move first 55.6% of the time. On real handwritten scoresheets, real player slips cost about one point. Accuracy is lower on real tournament games, which we trace to slow, classical play; fine-tuning on tournament games recovers about 40% of this gap, and the final system reaches 75.0% top-2 on real handwritten sheets.

**After it is announced:** add the arXiv link and a BibTeX citation to the repository README.
