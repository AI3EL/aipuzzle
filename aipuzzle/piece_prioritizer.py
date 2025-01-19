import random
from typing import Protocol

import numpy as np
import numpy.typing as npt

from aipuzzle.puzzle import Piece, Side


class PiecePrioritizer(Protocol):
    def sort(self, ref_piece: Piece, candidates: list[Piece], sides: set[Side]) -> list[tuple[Piece, Side]]: ...


class KeyQueryPredicter(Protocol):
    def get_keys(self, piece: list[Piece], sides: set[Side]) -> dict[Side, npt.NDArray[np.float32]]: ...
    def get_query(self, piece: Piece, sides: set[Side]) -> dict[Side, npt.NDArray[np.float32]]: ...


class RandomPiecePrioritizer(PiecePrioritizer):
    def sort(self, ref_piece: Piece, candidates: list[Piece], sides: set[Side]) -> list[tuple[Piece, Side]]:
        out = [(candidate, side) for side in sides for candidate in candidates]
        random.shuffle(out)
        return out


class KeyQueryPiecePrioritizer(PiecePrioritizer):
    def __init__(self, predicter: KeyQueryPredicter):
        self.predicter = predicter

    def sort(self, ref_piece: Piece, candidates: list[Piece], sides: set[Side]) -> list[tuple[Piece, Side]]:
        query = self.predicter.get_query(ref_piece, sides)
        keys = self.predicter.get_keys(candidates, sides)
        candidate_ids = [piece.id_ for piece in candidates]
        affinities = []
        for side in Side:
            affinities.extend([(affinity, i, side) for i, affinity in zip(candidate_ids, query[side].dot(keys[side]))])
        sorted_affinities = sorted(affinities, key=lambda x: -x[0])  # type: ignore
        return [(i, side) for (_affinity, i, side) in sorted_affinities]
