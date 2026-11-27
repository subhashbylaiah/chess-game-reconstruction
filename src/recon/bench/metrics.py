"""Reconstruction metrics (writing-block §13.2).

The evaluator re-checks legality itself and derives "which tokens did the solver
treat as errors" from the solver's alignment, so solvers cannot score points by
self-report.

Prediction format (one JSON object per instance)::

    {"id": ..., "candidates": [
        {"moves_uci": [...], "score": float,
         "alignment": [[ply | null per token] per source],   # optional
         "move_confidence": [float per ply]}                 # optional
     ], "stats": {"seconds": float, "model_evals": int, "expanded": int}}

Candidates must be ordered best first.
"""

from __future__ import annotations

from collections import defaultdict

import chess

from recon.bench.build import gap_bucket
from recon.observe import notation


def _replay(moves: list[str], start_fen: str) -> list[chess.Board] | None:
    """Boards *before* each ply, or None if the sequence is illegal."""
    board = chess.Board(start_fen)
    boards = []
    for uci in moves:
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            return None
        if move not in board.legal_moves:
            return None
        boards.append(board.copy(stack=False))
        board.push(move)
    return boards


def score_instance(inst: dict, pred: dict | None) -> dict:
    truth = inst["truth"]["moves_uci"]
    cands = [c for c in (pred or {}).get("candidates", [])]
    seqs = [c["moves_uci"] for c in cands]
    rec: dict = {
        "exact@1": float(bool(seqs) and seqs[0] == truth),
        "exact@5": float(truth in seqs[:5]),
        "exact@10": float(truth in seqs[:10]),
        "recall@all": float(truth in seqs),
        "num_candidates": len(seqs),
        "seconds": (pred or {}).get("stats", {}).get("seconds"),
        "model_evals": (pred or {}).get("stats", {}).get("model_evals"),
    }
    top = cands[0] if cands else None
    top_moves = top["moves_uci"] if top else []
    boards = _replay(top_moves, inst["input"]["start_fen"]) if top else None
    rec["legal@1"] = float(boards is not None and bool(top_moves))
    if boards is None:
        top_moves, boards = [], []
    rec["length_ok@1"] = float(len(top_moves) == len(truth))
    rec["move_acc@1"] = sum(a == b for a, b in zip(top_moves, truth)) / len(truth)
    hidden = inst["truth"]["hidden_plies"]
    rec["hidden_acc@1"] = (
        sum(p < len(top_moves) and top_moves[p] == truth[p] for p in hidden) / len(hidden) if hidden else None
    )

    # alignment + correction detection
    tp = fp = fn = 0
    align_hits = align_total = 0
    has_alignment = bool(top and top.get("alignment") and rec["legal@1"])
    for s, (src, tsrc) in enumerate(zip(inst["input"]["sources"], inst["truth"]["sources"])):
        pred_align = top["alignment"][s] if has_alignment and s < len(top["alignment"]) else None
        for t, tok in enumerate(src["tokens"]):
            truly_err = tsrc["token_kind"][t] != "match"
            if pred_align is None:
                continue
            p = pred_align[t]
            align_total += 1
            align_hits += p == tsrc["token_ply"][t]
            flagged = (
                p is None
                or not (0 <= p < len(top_moves))
                or not notation.matches(boards[p], chess.Move.from_uci(top_moves[p]), tok["text"])
            )
            tp += flagged and truly_err
            fp += flagged and not truly_err
            fn += (not flagged) and truly_err
    rec["align_acc@1"] = align_hits / align_total if align_total else None
    rec["_corr"] = (tp, fp, fn)

    conf = top.get("move_confidence") if top and rec["legal@1"] else None
    rec["_calib"] = (
        [(c, float(i < len(truth) and top_moves[i] == truth[i])) for i, c in enumerate(conf)] if conf else []
    )
    return rec


def _ece(pairs: list[tuple[float, float]], bins: int = 10) -> float | None:
    if not pairs:
        return None
    buckets: dict[int, list] = defaultdict(list)
    for c, y in pairs:
        buckets[min(int(c * bins), bins - 1)].append((c, y))
    n = len(pairs)
    return sum(len(b) / n * abs(sum(c for c, _ in b) / len(b) - sum(y for _, y in b) / len(b)) for b in buckets.values())


_MEAN_KEYS = [
    "exact@1", "exact@5", "exact@10", "recall@all", "legal@1", "length_ok@1",
    "move_acc@1", "hidden_acc@1", "align_acc@1", "seconds", "model_evals", "num_candidates",
]


def aggregate(recs: list[dict]) -> dict:
    out: dict = {"n": len(recs)}
    for k in _MEAN_KEYS:
        vals = [r[k] for r in recs if r.get(k) is not None]
        out[k] = sum(vals) / len(vals) if vals else None
    tp = sum(r["_corr"][0] for r in recs)
    fp = sum(r["_corr"][1] for r in recs)
    fn = sum(r["_corr"][2] for r in recs)
    out["corr_precision"] = tp / (tp + fp) if tp + fp else None
    out["corr_recall"] = tp / (tp + fn) if tp + fn else None
    out["ece"] = _ece([p for r in recs for p in r["_calib"]])
    return out


_ORDER = {
    "gap": ["0", "1-4", "5-9", "10-19", "20+"],
    "rating": ["<1200", "1200-1599", "1600-1999", "2000+", "unrated"],
    "speed": ["bullet", "blitz", "rapid", "classical", "unknown"],
}


def evaluate(instances: list[dict], predictions: dict[str, dict]) -> dict:
    recs, keys = [], []
    for inst in instances:
        recs.append(score_instance(inst, predictions.get(inst["id"])))
        m = inst["meta"]
        keys.append(
            {
                "preset": inst["preset"],
                "gap": gap_bucket(m["max_gap"]),
                "rating": m["rating_band"],
                "speed": m["speed"],
            }
        )
    report = {"overall": aggregate(recs), "by": {}}
    for dim in ("preset", "gap", "rating", "speed"):
        groups: dict[str, list] = defaultdict(list)
        for r, k in zip(recs, keys):
            groups[k[dim]].append(r)
        order = _ORDER.get(dim)
        names = sorted(groups, key=lambda g: (order.index(g) if order and g in order else 99, g))
        report["by"][dim] = {g: aggregate(groups[g]) for g in names}
    report["missing_predictions"] = sum(inst["id"] not in predictions for inst in instances)
    return report


def format_report(report: dict, cols: tuple[str, ...] = (
    "n", "exact@1", "exact@5", "recall@all", "move_acc@1", "hidden_acc@1",
    "align_acc@1", "corr_precision", "corr_recall", "ece", "seconds",
)) -> str:
    def fmt(v):
        if v is None:
            return "-"
        if isinstance(v, float):
            return f"{v:.3f}"
        return str(v)

    header = f"{'group':<22}" + "".join(f"{c:>{max(10, len(c) + 2)}}" for c in cols)
    lines = [header, "-" * len(header)]
    lines.append(f"{'overall':<22}" + "".join(f"{fmt(report['overall'][c]):>{max(10, len(c) + 2)}}" for c in cols))
    for dim, groups in report["by"].items():
        lines.append("")
        for g, agg in groups.items():
            lines.append(f"{dim + '=' + g:<22}" + "".join(f"{fmt(agg[c]):>{max(10, len(c) + 2)}}" for c in cols))
    if report.get("missing_predictions"):
        lines.append(f"\nmissing predictions: {report['missing_predictions']}")
    return "\n".join(lines)
