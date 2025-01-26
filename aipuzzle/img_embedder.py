from typing import Protocol

import numpy as np
import numpy.typing as npt
import timm
import torch
from timm.data import resolve_data_config  # type: ignore
from timm.data.transforms_factory import create_transform

from aipuzzle.env import Piece, PieceID, Side


class KeyQueryPredicter(Protocol):
    def get_query(self, piece: Piece, sides: set[Side]) -> dict[Side, npt.NDArray[np.float32]]: ...
    def get_keys(self, pieces: list[Piece], sides: set[Side]) -> dict[Side, npt.NDArray[np.float32]]: ...


class PretrainedImgEmbedder(KeyQueryPredicter):
    """Returning a copy of the same embedding on all sides, for query and keys"""

    def __init__(self, model_name: str, input_resolution: tuple[int, int], device: str = "cpu"):
        self._model: torch.nn.Module = timm.create_model(model_name, pretrained=True)  # type: ignore
        self._input_resolution = input_resolution
        self._device = device
        self._transform = create_transform(**resolve_data_config(self._model.pretrained_cfg, model=self._model))

        self._model.to(device)
        self._model.eval()

        self._inference_cache: dict[PieceID, npt.NDArray[np.float32]] = {}

    def _update_inference_cache(self, pieces: list[Piece]) -> None:
        is_in_cache = [piece.id_ in self._inference_cache for piece in pieces]
        if all(is_in_cache):
            return

        model_input = torch.stack(
            [self._transform(piece.texture) for (in_cache, piece) in zip(is_in_cache, pieces) if not in_cache],  # type: ignore
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

    def get_query(self, piece: Piece, sides: set[Side]) -> dict[Side, npt.NDArray[np.float32]]:
        self._update_inference_cache([piece])
        return {side: self._inference_cache[piece.id_].copy() for side in sides}

    def get_keys(self, pieces: list[Piece], sides: set[Side]) -> dict[Side, npt.NDArray[np.float32]]:
        self._update_inference_cache(pieces)
        embeddings = np.stack(
            [self._inference_cache[piece.id_] for piece in pieces],
            axis=0,
        )
        return {side: embeddings.copy() for side in sides}
