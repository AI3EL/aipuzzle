import random
from typing import Protocol

from aipuzzle.env import Piece, Side
from aipuzzle.img_embedder import KeyQueryPredicter


class PiecePrioritizer(Protocol):
    def sort(self, ref_piece: Piece, candidates: list[Piece], sides: set[Side]) -> list[tuple[Piece, Side]]: ...


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
        affinities = []
        for side in sides:
            affinities.extend(
                [(affinity, piece, side) for piece, affinity in zip(candidates, keys[side].dot(query[side]))]
            )
        sorting_fn = lambda x: -x[0]  # type: ignore
        sorted_affinities = sorted(affinities, key=sorting_fn)
        return [(piece, side) for (_affinity, piece, side) in sorted_affinities]
