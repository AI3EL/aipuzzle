from typing import Callable, Protocol

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image

from aipuzzle.env import Piece, PieceID, PuzzleEnv, Side, get_side_shifted


class PieceSideEmbedder(Protocol):
    def predict(self, pieces: list[Piece], sides: set[Side]) -> dict[Side, npt.NDArray[np.float32]]: ...


class NNPieceSideEmbedder(PieceSideEmbedder):
    def __init__(
        self,
        model: torch.nn.Module,
        transform: Callable[[Image.Image], torch.Tensor],
        input_resolution: tuple[int, int],
        device: str = "cpu",
    ):
        self._model = model
        self._transform = transform
        self._input_resolution = input_resolution
        self._device = device

        self._model.to(device)
        self._model.eval()

        self._inference_cache: dict[PieceID, npt.NDArray[np.float32]] = {}

    def _update_inference_cache(self, pieces: list[Piece]) -> None:
        is_in_cache = [piece.id_ in self._inference_cache for piece in pieces]
        if all(is_in_cache):
            return

        model_input = torch.stack(
            [self._transform(piece.texture) for (in_cache, piece) in zip(is_in_cache, pieces) if not in_cache],
            dim=0,
        )

        with torch.no_grad():
            new_embeddings = self._model(model_input).cpu().numpy()

        embedding_id = 0
        for piece, is_in_cache in zip(pieces, is_in_cache):
            if is_in_cache:
                continue
            self._inference_cache[piece.id_] = new_embeddings[embedding_id]
            embedding_id += 1

        return

    def predict(self, pieces: list[Piece], sides: set[Side]) -> dict[Side, npt.NDArray[np.float32]]:
        self._update_inference_cache(pieces)
        embeddings = np.stack(
            [self._inference_cache[piece.id_] for piece in pieces],
            axis=0,
        )
        return {side: embeddings[:, side.value - 1] for side in sides}


def get_difficulty_heatmap(
    puzzle: PuzzleEnv, query_embedder: PieceSideEmbedder, key_embedder: PieceSideEmbedder
) -> Image.Image:
    pieces = puzzle.get_pieces()
    solution = puzzle.get_solution()
    piece_w, piece_h = pieces[0].size
    image_w = puzzle.dims[0] * piece_w
    image_h = puzzle.dims[1] * piece_h
    arr = np.zeros((image_h, image_w, 3), np.float32)

    query_embeddings = query_embedder.predict(pieces, set(Side))
    key_embeddings = key_embedder.predict(pieces, set(Side))
    for pos, piece_id in solution.items():
        piece = pieces[piece_id]
        affinity = 0
        for side in piece.plugs:
            neigh_piece_id = solution[get_side_shifted(pos, side)]
            affinity += query_embeddings[side][piece_id].dot(key_embeddings[side][neigh_piece_id])
        affinity /= len(piece.plugs)
        arr[pos[1] * piece_h : (pos[1] + 1) * piece_h, pos[0] * piece_w : (pos[0] + 1) * piece_w] = affinity
    arr = (arr / arr.max() * 255).astype(np.uint8)
    return Image.fromarray(arr)
