"""Feature extraction for Lichess puzzles.

Given a puzzle row (FEN, Moves, optional Themes), compute a fixed-length
numeric feature vector suitable for gradient-boosted-tree regression.

Design notes
------------
- The provided FEN is the position BEFORE the opponent's setup move.
  The puzzle "starts" AFTER the first UCI move in Moves is applied.
- We compute features on both the setup position and the puzzle-start
  position, plus features aggregated over the solution sequence.
- Do NOT use NbPlays, RatingDeviation, Popularity — they are labels.
"""

from __future__ import annotations
from typing import Iterable
import chess
import numpy as np


# -------- theme vocabulary (top-K themes from Lichess) --------
# We pin the vocabulary to keep the feature vector stable across runs.
THEME_VOCAB = [
    "advantage", "crushing", "endgame", "middlegame", "opening",
    "short", "long", "veryLong", "oneMove", "mate", "mateIn1",
    "mateIn2", "mateIn3", "mateIn4", "mateIn5", "equality",
    "fork", "pin", "skewer", "hangingPiece", "trappedPiece",
    "discoveredAttack", "doubleCheck", "sacrifice", "quietMove",
    "attraction", "deflection", "clearance", "interference",
    "backRankMate", "smotheredMate", "arabianMate",
    "kingsideAttack", "queensideAttack",
    "defensiveMove", "counterAttack", "exposedKing",
    "promotion", "underPromotion",
    "enPassant", "castling", "capturingDefender",
    "advancedPawn", "xRayAttack", "pawnEndgame", "rookEndgame",
    "queenEndgame", "knightEndgame", "bishopEndgame",
    "queenRookEndgame",
    "zugzwang", "intermezzo",
]
THEME_INDEX = {t: i for i, t in enumerate(THEME_VOCAB)}

PIECE_TYPES = [chess.PAWN, chess.KNIGHT, chess.BISHOP,
               chess.ROOK, chess.QUEEN, chess.KING]

# Value assignments for material balance (kings excluded)
PIECE_VALUE = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
               chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}


def _mobility(board: chess.Board) -> int:
    return sum(1 for _ in board.legal_moves)


def _material(board: chess.Board):
    w = b = 0
    for pt in PIECE_TYPES:
        w += PIECE_VALUE[pt] * len(board.pieces(pt, chess.WHITE))
        b += PIECE_VALUE[pt] * len(board.pieces(pt, chess.BLACK))
    return w, b


def _king_ring_attackers(board: chess.Board, color: bool) -> int:
    """Count enemy pieces attacking squares in the king's 8-neighborhood."""
    king_sq = board.king(color)
    if king_sq is None:
        return 0
    kx = chess.square_file(king_sq)
    ky = chess.square_rank(king_sq)
    enemy = not color
    count = 0
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            x, y = kx + dx, ky + dy
            if 0 <= x < 8 and 0 <= y < 8:
                sq = chess.square(x, y)
                if board.is_attacked_by(enemy, sq):
                    count += 1
    return count


def _position_features(board: chess.Board, prefix: str) -> dict:
    """Return a dict of features for a static position."""
    feats: dict = {}
    # side to move
    feats[f"{prefix}_stm_white"] = int(board.turn == chess.WHITE)

    # piece counts by type/color
    total = 0
    for pt in PIECE_TYPES:
        wc = len(board.pieces(pt, chess.WHITE))
        bc = len(board.pieces(pt, chess.BLACK))
        feats[f"{prefix}_w_{chess.piece_symbol(pt)}"] = wc
        feats[f"{prefix}_b_{chess.piece_symbol(pt)}"] = bc
        total += wc + bc
    feats[f"{prefix}_total_pieces"] = total

    # material
    wm, bm = _material(board)
    feats[f"{prefix}_material_white"] = wm
    feats[f"{prefix}_material_black"] = bm
    feats[f"{prefix}_material_diff"] = wm - bm
    feats[f"{prefix}_material_diff_stm"] = (
        (wm - bm) if board.turn == chess.WHITE else (bm - wm)
    )

    # castling rights
    feats[f"{prefix}_castle_wk"] = int(board.has_kingside_castling_rights(chess.WHITE))
    feats[f"{prefix}_castle_wq"] = int(board.has_queenside_castling_rights(chess.WHITE))
    feats[f"{prefix}_castle_bk"] = int(board.has_kingside_castling_rights(chess.BLACK))
    feats[f"{prefix}_castle_bq"] = int(board.has_queenside_castling_rights(chess.BLACK))

    # king safety proxies
    feats[f"{prefix}_king_ring_att_w"] = _king_ring_attackers(board, chess.WHITE)
    feats[f"{prefix}_king_ring_att_b"] = _king_ring_attackers(board, chess.BLACK)
    stm_color = board.turn
    feats[f"{prefix}_king_ring_att_stm"] = _king_ring_attackers(board, stm_color)
    feats[f"{prefix}_king_ring_att_opp"] = _king_ring_attackers(board, not stm_color)

    # mobility (for side to move)
    feats[f"{prefix}_mobility_stm"] = _mobility(board)

    # in-check flag
    feats[f"{prefix}_in_check"] = int(board.is_check())

    # halfmove and fullmove clocks (proxy for opening-vs-endgame)
    feats[f"{prefix}_halfmove"] = board.halfmove_clock
    feats[f"{prefix}_fullmove"] = board.fullmove_number

    # king distance (Chebyshev)
    wk = board.king(chess.WHITE)
    bk = board.king(chess.BLACK)
    if wk is not None and bk is not None:
        feats[f"{prefix}_king_dist"] = max(
            abs(chess.square_file(wk) - chess.square_file(bk)),
            abs(chess.square_rank(wk) - chess.square_rank(bk)),
        )
    else:
        feats[f"{prefix}_king_dist"] = 0

    # pawn structure
    wpawn_ranks = [chess.square_rank(sq) for sq in board.pieces(chess.PAWN, chess.WHITE)]
    bpawn_ranks = [chess.square_rank(sq) for sq in board.pieces(chess.PAWN, chess.BLACK)]
    feats[f"{prefix}_wpawn_advance_max"] = max(wpawn_ranks) if wpawn_ranks else 0
    feats[f"{prefix}_bpawn_advance_max"] = 7 - min(bpawn_ranks) if bpawn_ranks else 0

    return feats


def _sequence_features(board_after_setup: chess.Board, solution: list[str]) -> dict:
    """Features aggregated over the solution move sequence.

    `board_after_setup` is the puzzle-start position; `solution` is the
    list of UCI moves the solver must play (plus opponent responses).
    """
    feats: dict = {}
    n = len(solution)
    feats["sol_len"] = n
    feats["sol_solver_moves"] = (n + 1) // 2  # solver plays moves 0,2,4,...
    feats["sol_opp_moves"] = n // 2

    checks = captures = promotions = under_promotions = 0
    en_passants = castles = 0
    solver_checks = solver_captures = 0
    quiet_moves = 0
    from_squares = []
    to_squares = []
    move_distances = []

    b = board_after_setup.copy(stack=False)
    piece_change = 0
    ends_in_mate = 0

    for i, uci in enumerate(solution):
        try:
            mv = chess.Move.from_uci(uci)
        except Exception:
            continue
        if mv not in b.legal_moves:
            # Some Lichess puzzles pass move validation via san; fallback
            # to skipping the rest gracefully.
            break
        is_solver = (i % 2 == 0)
        is_capture = b.is_capture(mv)
        gives_check = b.gives_check(mv)
        piece = b.piece_at(mv.from_square)
        from_squares.append(mv.from_square)
        to_squares.append(mv.to_square)
        move_distances.append(max(
            abs(chess.square_file(mv.from_square) - chess.square_file(mv.to_square)),
            abs(chess.square_rank(mv.from_square) - chess.square_rank(mv.to_square)),
        ))
        if is_capture:
            captures += 1
            if is_solver:
                solver_captures += 1
        else:
            quiet_moves += 1
        if gives_check:
            checks += 1
            if is_solver:
                solver_checks += 1
        if mv.promotion is not None:
            promotions += 1
            if mv.promotion != chess.QUEEN:
                under_promotions += 1
        if b.is_en_passant(mv):
            en_passants += 1
        if b.is_castling(mv):
            castles += 1
        b.push(mv)
        if b.is_checkmate():
            ends_in_mate = 1

    feats["sol_checks"] = checks
    feats["sol_captures"] = captures
    feats["sol_promotions"] = promotions
    feats["sol_under_promotions"] = under_promotions
    feats["sol_en_passants"] = en_passants
    feats["sol_castles"] = castles
    feats["sol_solver_checks"] = solver_checks
    feats["sol_solver_captures"] = solver_captures
    feats["sol_quiet"] = quiet_moves
    feats["sol_ends_mate"] = ends_in_mate
    feats["sol_quiet_ratio"] = quiet_moves / n if n else 0.0
    feats["sol_check_ratio"] = checks / n if n else 0.0
    feats["sol_capture_ratio"] = captures / n if n else 0.0
    feats["sol_max_move_dist"] = max(move_distances) if move_distances else 0
    feats["sol_mean_move_dist"] = (
        float(np.mean(move_distances)) if move_distances else 0.0
    )
    # spatial spread of solution
    if to_squares:
        files = [chess.square_file(sq) for sq in to_squares]
        ranks = [chess.square_rank(sq) for sq in to_squares]
        feats["sol_to_file_range"] = max(files) - min(files)
        feats["sol_to_rank_range"] = max(ranks) - min(ranks)
    else:
        feats["sol_to_file_range"] = 0
        feats["sol_to_rank_range"] = 0

    return feats


def _theme_features(themes: Iterable[str] | None) -> dict:
    feats = {f"theme_{t}": 0 for t in THEME_VOCAB}
    feats["theme_missing"] = 1
    feats["theme_count"] = 0
    if themes is None:
        return feats
    seen = 0
    for t in themes:
        seen += 1
        idx = THEME_INDEX.get(t)
        if idx is not None:
            feats[f"theme_{t}"] = 1
    feats["theme_missing"] = int(seen == 0)
    feats["theme_count"] = seen
    return feats


def extract_features(fen: str, moves: str, themes=None) -> dict:
    """Extract all features for one puzzle.

    Parameters
    ----------
    fen : str
        FEN BEFORE the opponent's setup move (as stored in the parquet).
    moves : str
        Space-separated UCI moves. First move is opponent setup;
        remaining moves are the puzzle solution sequence.
    themes : list[str] | None
        Optional theme tags.
    """
    feats: dict = {}
    board = chess.Board(fen)
    feats.update(_position_features(board, "pre"))

    move_list = moves.split() if moves else []
    setup_ok = False
    if move_list:
        try:
            setup = chess.Move.from_uci(move_list[0])
            if setup in board.legal_moves:
                board.push(setup)
                setup_ok = True
        except Exception:
            pass
    feats["setup_move_ok"] = int(setup_ok)
    feats.update(_position_features(board, "post"))

    solution = move_list[1:] if setup_ok else move_list
    feats.update(_sequence_features(board, solution))
    feats.update(_theme_features(themes))
    return feats


def feature_names() -> list[str]:
    """Fixed feature order — used for building arrays deterministically."""
    fen0 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    sample = extract_features(fen0, "e2e4 e7e5", themes=["opening"])
    return list(sample.keys())


def features_to_array(feats: dict, names: list[str]) -> np.ndarray:
    return np.array([feats.get(n, 0.0) for n in names], dtype=np.float32)
