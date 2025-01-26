from typing import Callable, Protocol

import numpy as np
import numpy.typing as npt
import torch
from PIL import Image

from aipuzzle.env import Piece, PieceID, Side


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
