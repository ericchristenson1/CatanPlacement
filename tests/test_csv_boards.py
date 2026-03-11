"""
Tests that the solver runs end-to-end on each CSV board fixture and
produces valid optimal placements for all player counts (2, 3, 4).
"""

import os
import sys
import pytest

# Make the parent directory importable so we can import solve_custom, board, solver
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from solve_custom import load_board_from_csv
from board import Board
from solver import Solver

BOARDS_DIR = os.path.join(os.path.dirname(__file__), "boards")

BOARD_FILES = sorted(
    os.path.join(BOARDS_DIR, f)
    for f in os.listdir(BOARDS_DIR)
    if f.endswith(".csv")
)

# Parametrize over every CSV × every player count
@pytest.mark.parametrize("csv_path", BOARD_FILES, ids=lambda p: os.path.basename(p))
@pytest.mark.parametrize("num_players", [2, 3, 4])
def test_solver_runs_and_finds_solution(csv_path, num_players):
    tile_config = load_board_from_csv(csv_path)
    assert len(tile_config) == 19, "CSV must produce exactly 19 tiles"

    board = Board.from_config(tile_config, num_players=num_players)
    assert len(board.tiles) == 19
    assert len(board.vertices) == 54

    solver = Solver(board, enable_pruning=True)
    final_state, player1_positions, player1_quality = solver.solve()

    assert final_state is not None, "Solver must find a solution"

    for player in range(1, num_players + 1):
        houses = final_state.houses[player]
        assert len(houses) == 2, (
            f"Player {player} must have exactly 2 settlements, got {houses}"
        )
        for v in houses:
            assert v in board.vertices, (
                f"Player {player} settlement vertex {v} not on board"
            )
        quality = final_state.quality_of_player(player)
        assert quality > 0, (
            f"Player {player} quality must be positive, got {quality}"
        )


@pytest.mark.parametrize("csv_path", BOARD_FILES, ids=lambda p: os.path.basename(p))
def test_all_players_get_distinct_vertex_pairs(csv_path):
    """No two players should share any settlement vertex."""
    tile_config = load_board_from_csv(csv_path)
    board = Board.from_config(tile_config, num_players=4)
    solver = Solver(board, enable_pruning=True)
    final_state, _, _ = solver.solve()

    assert final_state is not None
    all_vertices = []
    for player in range(1, 5):
        all_vertices.extend(final_state.houses[player])

    assert len(all_vertices) == len(set(all_vertices)), (
        "Two players share a settlement vertex"
    )


@pytest.mark.parametrize("csv_path", BOARD_FILES, ids=lambda p: os.path.basename(p))
def test_all_qualities_positive_and_bounded(csv_path):
    """
    Every player's placement quality must be positive and no greater than the
    unconstrained best-pair quality (what a single player gets with the whole
    board to themselves).
    """
    tile_config = load_board_from_csv(csv_path)
    board = Board.from_config(tile_config, num_players=4)

    # Upper bound: solve for 1 player (unconstrained)
    solo_board = Board.from_config(tile_config, num_players=1)
    solo_solver = Solver(solo_board, enable_pruning=True)
    solo_state, _, solo_quality = solo_solver.solve()
    assert solo_state is not None
    upper_bound = solo_quality

    solver = Solver(board, enable_pruning=True)
    final_state, _, _ = solver.solve()
    assert final_state is not None

    for player in range(1, 5):
        q = final_state.quality_of_player(player)
        assert q > 0, f"Player {player} quality must be positive, got {q}"
        assert q <= upper_bound + 1e-9, (
            f"Player {player} quality {q:.4f} exceeds unconstrained best {upper_bound:.4f}"
        )
