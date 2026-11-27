import random

import chess

from recon.observe import notation


def board_after(*sans: str) -> chess.Board:
    b = chess.Board()
    for s in sans:
        b.push_san(s)
    return b


def test_normalize_variants():
    assert notation.normalize("Nxf3+") == "Nf3"
    assert notation.normalize("0-0") == notation.normalize("O-O") == "OO"
    assert notation.normalize("e8=Q#") == "e8Q"
    assert notation.normalize("♘f3") == "Nf3"


def test_resolve_styles():
    b = chess.Board()
    assert notation.resolve(b, "Nf3") == [chess.Move.from_uci("g1f3")]
    assert notation.resolve(b, "g1f3") == [chess.Move.from_uci("g1f3")]
    assert notation.resolve(b, "Ng1-f3") == [chess.Move.from_uci("g1f3")]
    assert notation.resolve(b, "Nf5") == []


def test_resolve_capture_without_x_and_castling_zeros():
    b = board_after("e4", "d5")
    assert notation.resolve(b, "ed5") == [chess.Move.from_uci("e4d5")]
    b = board_after("e4", "e5", "Nf3", "Nc6", "Bc4", "Bc5")
    assert notation.resolve(b, "0-0") == [chess.Move.from_uci("e1g1")]


def test_under_disambiguated_returns_all():
    # Both knights can reach d2 after 1.d4 d5 2.Nf3 Nf6 3.e3 e6 4.Bd3 Bd6 5.c3 c6 ... use a simple case:
    b = chess.Board("4k3/8/8/8/8/8/8/1N2KN2 w - - 0 1")  # knights b1, f1 both reach d2
    got = set(notation.resolve(b, "Nd2"))
    assert got == {chess.Move.from_uci("b1d2"), chess.Move.from_uci("f1d2")}
    assert notation.resolve(b, "Nbd2") == [chess.Move.from_uci("b1d2")]


def test_ocr_corrupt_changes_text():
    rng = random.Random(0)
    for san in ["Nf3", "e4", "Qxd8+", "O-O", "Rae1"]:
        for _ in range(20):
            assert notation.ocr_corrupt(san, rng) != san


def test_slip_is_legal_and_different():
    rng = random.Random(1)
    b = board_after("e4", "e5")
    move = chess.Move.from_uci("g1f3")
    for _ in range(20):
        wrong = notation.slip_corrupt(b, move, rng)
        assert wrong is not None
        assert b.parse_san(wrong) != move
