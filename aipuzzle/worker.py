from typing import Protocol

from aipuzzle.env import (
    PuzzleAction,
    PuzzleEnv,
    PuzzleObs,
    Side,
    get_any_corner,
    get_free_plug_piece,
    get_side_shifted,
    get_unplaced_pieces,
)
from aipuzzle.piece_prioritizer import PiecePrioritizer


class PuzzleWorker(Protocol):
    def run_episode(self, puzzle: PuzzleEnv) -> list[PuzzleObs]: ...


class AdjacentPuzzleWorker(PuzzleWorker):
    def __init__(self, prioritizer: PiecePrioritizer):
        super().__init__()
        self._prioritizer = prioritizer

    def run_episode(self, puzzle: PuzzleEnv) -> list[PuzzleObs]:
        out = []
        obs = puzzle.reset()
        out.append(obs)

        corner_piece = get_any_corner(obs)
        piece_pos: tuple[int, int]
        if corner_piece.plugs == set([Side.UP, Side.RIGHT]):
            piece_pos = (0, 0)
        elif corner_piece.plugs == set([Side.RIGHT, Side.DOWN]):
            piece_pos = (0, obs.dims[1] - 1)
        elif corner_piece.plugs == set([Side.LEFT, Side.DOWN]):
            piece_pos = (obs.dims[0] - 1, obs.dims[1] - 1)
        elif corner_piece.plugs == set([Side.LEFT, Side.UP]):
            piece_pos = (obs.dims[0] - 1, 0)
        else:
            raise ValueError
        obs, _reward, done, _infos = puzzle.step(PuzzleAction(corner_piece.id_, piece_pos))
        out.append(obs)

        while not done:
            ref_piece_pos, ref_piece_id, free_plug_sides = get_free_plug_piece(obs)
            candidate_pieces = get_unplaced_pieces(obs)
            priroitized_candidate_piece_and_sides = self._prioritizer.sort(
                obs.pieces[ref_piece_id], candidate_pieces, free_plug_sides
            )
            for candidate_piece, unplugged_side in priroitized_candidate_piece_and_sides:
                action = PuzzleAction(candidate_piece.id_, get_side_shifted(ref_piece_pos, unplugged_side))
                next_obs, _reward, done, _infos = puzzle.step(action)
                out.append(next_obs)
                if len(next_obs.layout.keys()) > len(obs.layout.keys()):
                    obs = next_obs
                    break

        return out
