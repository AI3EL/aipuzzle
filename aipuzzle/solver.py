import random
from typing import Protocol

import numpy as np

from aipuzzle.piece_prioritizer import PiecePrioritizer
from aipuzzle.puzzle import PieceID, Puzzle, PuzzleSolution, Side, get_opposite_side, side_to_delta_pos


class Solver(Protocol):
    def solve(self, puzzle: Puzzle) -> PuzzleSolution: ...


class PrioritizedBruteforceSolver(Solver):
    def __init__(self, prioritizer: PiecePrioritizer):
        super().__init__()
        self._prioritizer = prioritizer

    def solve(self, puzzle: Puzzle) -> PuzzleSolution:
        # Just annoying
        assert puzzle.dims[0] >= 3
        assert puzzle.dims[1] >= 3

        solution: PuzzleSolution = {}
        placed_pos: dict[PieceID, tuple[int, int]] = {}
        unpluggeds: list[set[Side]] = [piece.plugs.copy() for piece in puzzle.pieces]
        placed_and_unplugged: set[PieceID] = set()

        for corner_id in puzzle.get_corner_pieces():
            corner = puzzle.pieces[corner_id]
            corner_pos: tuple[int, int]
            if corner.plugs == set([Side.UP, Side.RIGHT]):
                corner_pos = (0, 0)
            elif corner.plugs == set([Side.RIGHT, Side.DOWN]):
                corner_pos = (0, puzzle.dims[1] - 1)
            elif corner.plugs == set([Side.LEFT, Side.DOWN]):
                corner_pos = (puzzle.dims[0] - 1, puzzle.dims[1] - 1)
            elif corner.plugs == set([Side.LEFT, Side.UP]):
                corner_pos = (puzzle.dims[0] - 1, 0)
            else:
                raise ValueError

            solution[corner_pos] = corner.id_
            placed_pos[corner.id_] = corner_pos
            placed_and_unplugged.add(corner_id)

        while len(solution) != len(puzzle.pieces):
            found = False

            ref_piece = puzzle.pieces[random.choice(list(placed_and_unplugged))]
            candidate_pieces = [piece for piece in puzzle.pieces if piece.id_ not in placed_pos]
            priroitized_candidate_piece_and_sides = self._prioritizer.sort(
                ref_piece, candidate_pieces, unpluggeds[ref_piece.id_]
            )
            for candidate_piece, unplugged_side in priroitized_candidate_piece_and_sides:
                ref_piece_pos = placed_pos[ref_piece.id_]
                if not puzzle.clicks(ref_piece, candidate_piece, unplugged_side):
                    continue

                found = True
                arr_click_pos = np.array(ref_piece_pos, np.int32) + np.array(
                    side_to_delta_pos(unplugged_side), np.int32
                )
                click_pos = (int(arr_click_pos[0]), int(arr_click_pos[1]))
                solution[click_pos] = candidate_piece.id_
                placed_pos[candidate_piece.id_] = click_pos
                placed_and_unplugged.add(candidate_piece.id_)
                # print(f"Placed {candidate_piece.id_} at {click_pos}")

                # Marking as plugged neighbours and candidate (including ref )
                for side in Side:
                    neigh_pos = tuple(
                        map(int, np.array(click_pos, np.int32) + np.array(side_to_delta_pos(side), np.int32))
                    )
                    if neigh_pos in solution:
                        unpluggeds[candidate_piece.id_].remove(side)
                        # print(f"Plugged {candidate_piece.id_} {side}")
                        if not unpluggeds[candidate_piece.id_]:
                            placed_and_unplugged.remove(candidate_piece.id_)
                        unpluggeds[solution[neigh_pos]].remove(get_opposite_side(side))
                        # print(f"Plugged {solution[neigh_pos]} {get_opposite_side(side)}")
                        if not unpluggeds[solution[neigh_pos]]:
                            placed_and_unplugged.remove(solution[neigh_pos])
                break

            assert found

        return solution
