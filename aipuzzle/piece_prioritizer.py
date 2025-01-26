import random
from typing import Protocol

from aipuzzle.env import Piece, Side
from aipuzzle.img_embedder import PieceSideEmbedder


class PiecePrioritizer(Protocol):
    def sort(self, ref_piece: Piece, candidates: list[Piece], sides: set[Side]) -> list[tuple[Piece, Side]]: ...


class RandomPiecePrioritizer(PiecePrioritizer):
    def sort(self, ref_piece: Piece, candidates: list[Piece], sides: set[Side]) -> list[tuple[Piece, Side]]:
        out = [(candidate, side) for side in sides for candidate in candidates]
        random.shuffle(out)
        return out


class KeyQueryPiecePrioritizer(PiecePrioritizer):
    def __init__(self, query_predicter: PieceSideEmbedder, key_predicter: PieceSideEmbedder):
        self.query_predicter = query_predicter
        self.key_predicter = key_predicter

    def sort(self, ref_piece: Piece, candidates: list[Piece], sides: set[Side]) -> list[tuple[Piece, Side]]:
        query = {side: embeddings[0] for side, embeddings in self.query_predicter.predict([ref_piece], sides).items()}
        keys = self.key_predicter.predict(candidates, sides)
        affinities = []
        for side in sides:
            affinities.extend(
                [(affinity, piece, side) for piece, affinity in zip(candidates, keys[side].dot(query[side]))]
            )
        sorting_fn = lambda x: -x[0]  # type: ignore
        sorted_affinities = sorted(affinities, key=sorting_fn)
        return [(piece, side) for (_affinity, piece, side) in sorted_affinities]
