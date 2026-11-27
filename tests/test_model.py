import random

import chess
import torch

from recon.model import features as F
from recon.model.dataset import game_positions
from recon.model.net import ModelConfig, ReconModel, losses
from recon.observe.corrupt import PRESETS, corrupt

from test_bench import game  # noqa: F401  (fixture)


def test_move_encoding_roundtrip_both_colours():
    for fen in [chess.STARTING_FEN, "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"]:
        b = chess.Board(fen)
        for m in b.legal_moves:
            idx = F.encode_move(m, b.turn)
            assert F.move_from_outputs(b, idx, 0) == m


def test_black_view_is_mirrored():
    b = chess.Board()
    b.push_san("e4")
    sq_w, _, _ = F.encode_board(chess.Board(), None)
    sq_b, last, legal = F.encode_board(b, b.peek())
    # After 1.e4 from Black's view, "my" pieces sit on ranks 1-2 like White's at the start.
    assert (sq_b[:16] == sq_w[:16]).all()
    # Black's e7e5 is encoded like White's e2e4 in the mirrored frame.
    assert legal[F.encode_move(chess.Move.from_uci("e7e5"), chess.BLACK)]
    assert F.encode_move(chess.Move.from_uci("e7e5"), chess.BLACK) == chess.E2 * 64 + chess.E4
    # The last move (e2e4) appears mirrored as e7->e5.
    assert last[chess.E7] == 1 and last[chess.E5] == 2


def test_promotion_decoding():
    b = chess.Board("8/4P3/8/8/8/8/k7/4K3 w - - 0 1")
    idx = F.encode_move(chess.Move.from_uci("e7e8q"), chess.WHITE)
    assert F.move_from_outputs(b, idx, 0).promotion == chess.QUEEN
    assert F.move_from_outputs(b, idx, 1).promotion == chess.KNIGHT


def test_labels_point_at_the_right_token(game):  # noqa: F811
    for seed in range(10):
        rng = random.Random(seed)
        cfg = PRESETS["mixed"]
        sources, truth = corrupt(game, cfg, random.Random(seed))
        # Re-run with the same seed inside game_positions (keep all written plies).
        xs = list(game_positions(game, cfg, random.Random(seed), 1.0, F.Context(1500, 1600, 600, 5)))
        assert len(xs) == len(truth["moves_uci"])
        for i, x in enumerate(xs):
            assert x["legal"][x["move"]], "label move must be legal"
            assert x["hidden"] == (i in truth["sources"][0]["omitted_plies"])
            if x["hidden"]:
                assert x["align"] == F.K
            else:
                assert x["align"] < F.K  # window always contains the next token for this ply
                assert x["gap"] == 0
        del rng


def test_forward_and_loss_shapes():
    model = ReconModel(ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128))
    b = chess.Board()
    xs = [F.encode(b, None, [{"text": "e4", "move_number_hint": 1, "side_hint": "white"}], 0, 0, F.Context())] * 3
    batch = {k: torch.from_numpy(__import__("numpy").stack([x[k] for x in xs])) for k in xs[0]}
    batch |= {
        "move": torch.full((3,), F.encode_move(chess.Move.from_uci("e2e4"), chess.WHITE)),
        "promo": torch.zeros(3, dtype=torch.long),
        "align": torch.zeros(3, dtype=torch.long),
        "gap": torch.zeros(3, dtype=torch.long),
    }
    out = model(batch)
    assert out["move"].shape == (3, 4096) and out["align"].shape == (3, F.K + 1)
    loss, _ = losses(out, batch)
    loss.backward()
    assert torch.isfinite(loss)


def test_parse_entry():
    import chess
    from recon.model.features import parse_entry

    assert parse_entry("Nf3") == (2, chess.F3 + 1, 0, 0, 0, 0)
    assert parse_entry("Bxf3+") == (3, chess.F3 + 1, 1, 0, 0, 0)
    assert parse_entry("exd5") == (1, chess.D5 + 1, 1, 5, 0, 0)
    assert parse_entry("e4") == (1, chess.E4 + 1, 0, 0, 0, 0)
    assert parse_entry("Nbd2") == (2, chess.D2 + 1, 0, 2, 0, 0)
    assert parse_entry("R1e2") == (4, chess.E2 + 1, 0, 0, 1, 0)
    assert parse_entry("e8=Q") == (1, chess.E8 + 1, 0, 0, 0, 4)
    assert parse_entry("0-0") == (7, chess.G1 + 1, 0, 0, 0, 0)
    assert parse_entry("O-O-O") == (8, chess.C1 + 1, 0, 0, 0, 0)
    assert parse_entry("Bxt3") == (0, 0, 0, 0, 0, 0)  # garbled: left to the character view


def test_struct_and_gap_model_runs():
    import numpy as np
    import torch
    import chess
    from recon.model import features as F
    from recon.model.net import ModelConfig, ReconModel

    toks = [{"text": "e5"}, {"text": "Nf3"}, {"text": "Nc6"}]
    xs = [F.encode(chess.Board(), None, toks, 0, 0, F.Context(), gap) for gap in (None, 1)]
    batch = {k: torch.from_numpy(np.stack([x[k] for x in xs])) for k in xs[0]}
    for struct, gap in [(True, False), (True, True), (False, True)]:
        m = ReconModel(ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128, struct_notes=struct, gap_input=gap))
        out = m(batch)
        assert out["move"].shape == (2, 4096) and out["align"].shape == (2, F.K + 1)


def test_history_and_move_number():
    import chess
    from recon.model import features as F

    b = chess.Board()
    for u in ["e2e4", "e7e5", "g1f3"]:
        b.push_uci(u)
    x = F.encode(b, b.peek(), [], 0, 3, F.Context())
    assert x["move_num"] == 2  # Black to play move 2
    h = x["hist"]
    # Black to move: frame mirrored, so White's Ng1-f3 is an opponent knight from g8 to f6 (mirrored)
    assert tuple(h[0]) == (2 + 6, chess.G8 + 1, chess.F6 + 1, 0)
    assert tuple(h[1]) == (1, chess.E2 + 1, chess.E4 + 1, 0)  # own pawn e7-e5 seen as e2-e4
    assert tuple(h[2]) == (1 + 6, chess.E7 + 1, chess.E5 + 1, 0)  # opponent e2-e4 seen as e7-e5
    assert (h[3:] == 0).all()
    # a board copied with limited stack still yields full history
    assert (F.encode(b.copy(stack=F.HIST), b.peek(), [], 0, 3, F.Context())["hist"] == h).all()


def test_history_model_runs():
    import numpy as np
    import torch
    import chess
    from recon.model import features as F
    from recon.model.net import ModelConfig, ReconModel

    b = chess.Board()
    xs = [F.encode(b, None, [{"text": "e4"}], 0, 0, F.Context())]
    b.push_uci("e2e4")
    xs.append(F.encode(b, b.peek(), [{"text": "e5"}], 0, 1, F.Context()))
    batch = {k: torch.from_numpy(np.stack([x[k] for x in xs])) for k in xs[0]}
    m = ReconModel(ModelConfig(d_model=64, n_layers=2, n_heads=4, d_ff=128, move_number=True, hist_moves=8, gap_input=True))
    out = m(batch)
    assert torch.isfinite(out["move"].max(-1).values).all() and out["align"].shape == (2, F.K + 1)


def test_feasibility_filter():
    import chess
    from recon.solvers.feasible import Entries, dist, violations

    assert dist(chess.KNIGHT, True, chess.G1, chess.F3) == 1
    assert dist(chess.BISHOP, True, chess.C1, chess.D3) == 99  # wrong colour
    assert dist(chess.PAWN, True, chess.E2, chess.E4) == 1  # double step
    assert dist(chess.PAWN, True, chess.E4, chess.E3) == 99  # pawns never go back

    b = chess.Board()
    # White's ply 0 is blank; ply 2 (White) is written as O-O: moving the king or h-rook now is impossible
    tokens = [{"text": "e5"}, {"text": "O-O"}]
    ent = Entries(tokens, {1: 0, 2: 1})
    assert violations(b, chess.Move.from_uci("e2e4"), 0, ent) == 0
    b2 = chess.Board("rnbqkbnr/pppppppp/8/8/8/5NP1/PPPPPPBP/RNBQK2R w KQkq - 0 1")
    assert violations(b2, chess.Move.from_uci("h1g1"), 0, ent) == 1
    assert violations(b2, chess.Move.from_uci("e1f1"), 0, ent) == 1
    # next White entry Nc3 one ply later than the opponent: only one White move left -> knight must reach c3 now+1
    ent2 = Entries([{"text": "e5"}, {"text": "Nc3"}], {1: 0, 2: 1})
    assert violations(b, chess.Move.from_uci("b1c3"), 0, ent2) == 1  # knight already on c3 can't move onto it
    assert violations(b, chess.Move.from_uci("e2e4"), 0, ent2) == 0
    # OCR: if any plausible reading is feasible, no violation
    ent3 = Entries([{"text": "e5"}, {"text": "Nc3", "alternatives": [{"text": "Nc3", "p": 0.6}, {"text": "Ne2", "p": 0.4}]}], {1: 0, 2: 1})
    assert violations(b, chess.Move.from_uci("b1c3"), 0, ent3) == 0


def test_future_encoding_and_targets():
    import random
    import chess
    import numpy as np
    from recon.model import features as F

    assert F.offset_id(1) == 1 and F.offset_id(16) == 16 and F.offset_id(17) == 17 and F.offset_id(100) == F.N_OFF - 1
    tokens = [{"text": "e4"}, {"text": "e5"}, {"text": "Nf3"}, {"text": "Nc6"}]
    layout = {0: 0, 1: 1, 4: 2, 5: 3}  # plies 2,3 blank
    b = chess.Board()
    b.push_uci("e2e4"); b.push_uci("e7e5")
    x = F.encode(b, chess.Move.from_uci("e7e5"), tokens, 2, 2, F.Context(), layout=layout)
    assert x["fut_mask"].sum() == 2  # entries at plies 4 and 5
    assert list(x["fut_feats"][:2, 0]) == [2, 3]  # offsets from ply 2
    assert x["fut_struct"][0, 1] == chess.F3 + 1  # Nf3 destination (White to move: no mirroring)
    # Black to move at ply 3: "Nc6" mirrors to c3 in Black's frame
    b.push_uci("d2d4")
    y = F.encode(b, chess.Move.from_uci("d2d4"), tokens, 2, 3, F.Context(), layout=layout)
    assert y["fut_mask"].sum() == 2 and y["fut_struct"][1, 1] == chess.C3 + 1
    # orient_raw matches encode_board's frame
    raw = F.raw_squares(b)
    assert (F.orient_raw(raw, b.turn) == F.encode_board(b, None)[0]).all()


def test_future_model_and_dataset_run():
    import random
    import torch
    from recon.model import features as F
    from recon.model.dataset import game_positions
    from recon.model.net import ModelConfig, ReconModel, losses
    from recon.observe.corrupt import PRESETS
    from recon.game import Game

    g = Game(id="t", moves=["e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5a4", "g8f6", "e1g1", "f8e7"])
    rows = list(game_positions(g, PRESETS["missing"], random.Random(1), 1.0, F.Context()))
    assert rows and all("fut_board" in r for r in rows)
    batch = {k: torch.from_numpy(__import__("numpy").stack([r[k] for r in rows])) for k in rows[0]}
    m = ReconModel(ModelConfig(d_model=32, n_layers=1, n_heads=2, d_ff=64, future=F.NF, future_boards=F.NB))
    out = m(batch)
    assert out["fut_board"].shape == (len(rows), 64, F.NB, 13)
    loss, parts = losses(out, batch)
    assert torch.isfinite(loss) and "fb" in parts
