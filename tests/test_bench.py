import random

import chess
import pytest

from recon.bench import build, metrics
from recon.game import Game, TimeControl
from recon.observe import notation
from recon.observe.corrupt import PRESETS, corrupt
from recon.solvers.naive import NaiveSolver

# A 60-ply game (Lichess-style), used as a fixture.
SANS = (
    "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 Nb8 d4 Nbd7 "
    "c4 c6 cxb5 axb5 Nc3 Bb7 Bg5 b4 Nb1 h6 Bh4 c5 dxe5 Nxe4 Bxe7 Qxe7 exd6 Qf6 "
    "Nbd2 Nxd6 Nc4 Nxc4 Bxc4 Nb6 Ne5 Rae8 Bxf7+ Rxf7 Nxf7 Rxe1+ Qxe1 Kxf7 Qe3 Qg5 "
    "Qxg5 hxg5 b3 Ke6 a3 Kd6"
).split()


@pytest.fixture
def game() -> Game:
    b = chess.Board()
    moves = []
    for s in SANS:
        m = b.parse_san(s)
        moves.append(m.uci())
        b.push(m)
    return Game(id="testgame1", moves=moves, white_elo=1500, black_elo=1600, time_control=TimeControl(600, 5))


def _check_truth_consistent(game, sources, truth):
    """Every token's ground truth must be consistent with what was written."""
    boards = [b for b, _ in game.replay(len(truth["moves_uci"]))]
    for src, t in zip(sources, truth["sources"]):
        covered = [p for p in t["token_ply"] if p is not None]
        assert covered == sorted(covered), "tokens must follow ply order"
        assert set(covered) | set(t["omitted_plies"]) == set(range(len(truth["moves_uci"])))
        assert not set(covered) & set(t["omitted_plies"])
        for tok, ply, kind in zip(src["tokens"], t["token_ply"], t["token_kind"]):
            if kind == "match":
                assert notation.matches(boards[ply], chess.Move.from_uci(truth["moves_uci"][ply]), tok["text"]), tok
            elif kind == "inserted":
                assert ply is None


@pytest.mark.parametrize("preset", list(PRESETS))
def test_corruption_ground_truth(game, preset):
    for seed in range(30):
        sources, truth = corrupt(game, PRESETS[preset], random.Random(seed))
        assert len(sources) == PRESETS[preset].num_sources
        _check_truth_consistent(game, sources, truth)


def test_clean_has_no_errors(game):
    sources, truth = corrupt(game, PRESETS["clean"], random.Random(0))
    assert all(k == "match" for k in truth["sources"][0]["token_kind"])
    assert truth["sources"][0]["omitted_plies"] == []


def test_missing_preset_has_late_gap(game):
    starts = []
    for seed in range(200):
        _, truth = corrupt(game, PRESETS["missing"], random.Random(seed))
        om = truth["sources"][0]["omitted_plies"]
        assert om, "missing preset always has a gap on a 60-ply game"
        assert max(om) <= len(SANS) - 3, "no trailing gap"
        starts.append(min(om))
    assert sum(starts) / len(starts) > len(SANS) / 2, "bursts are biased late"


def test_instances_deterministic(game):
    a = list(build.build_instances([game], ["mixed"], 1, seed=3))
    b = list(build.build_instances([game], ["mixed"], 1, seed=3))
    assert a == b


def test_naive_solves_clean_exactly(game):
    inst = build.make_instance(game, PRESETS["clean"], seed=0, split="test")
    pred = {"id": inst["id"], **NaiveSolver().solve(inst["input"])}
    rec = metrics.score_instance(inst, pred)
    assert rec["exact@1"] == 1.0
    assert rec["align_acc@1"] == 1.0
    assert rec["_corr"] == (0, 0, 0)


def test_metrics_reject_illegal_and_score_oracle(game):
    inst = build.make_instance(game, PRESETS["mixed"], seed=0, split="test")
    truth = inst["truth"]
    oracle = {"candidates": [{"moves_uci": truth["moves_uci"], "score": 0, "alignment": [s["token_ply"] for s in truth["sources"]]}]}
    rec = metrics.score_instance(inst, oracle)
    assert rec["exact@1"] == rec["align_acc@1"] == rec["move_acc@1"] == 1.0
    tp, fp, fn = rec["_corr"]
    assert fn == 0 and fp == 0  # oracle alignment flags exactly the corrupted tokens

    illegal = {"candidates": [{"moves_uci": ["e2e5"] + truth["moves_uci"][1:], "score": 0}]}
    rec = metrics.score_instance(inst, illegal)
    assert rec["legal@1"] == 0.0 and rec["move_acc@1"] == 0.0
