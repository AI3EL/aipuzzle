from typing import Protocol

import numpy as np
import numpy.typing as npt
import timm
import torch
from timm.data import resolve_data_config  # type: ignore
from timm.data.transforms_factory import create_transform

from aipuzzle.puzzle import Piece, Side


class KeyQueryPredicter(Protocol):
    def get_query(self, piece: Piece, sides: set[Side]) -> dict[Side, npt.NDArray[np.float32]]: ...
    def get_keys(self, pieces: list[Piece], sides: set[Side]) -> dict[Side, npt.NDArray[np.float32]]: ...


# TODO: add caching
class PretrainedImgEmbedder(KeyQueryPredicter):
    """Returning a copy of embedding on all sides"""

    def __init__(self, model_name: str, input_resolution: tuple[int, int], device: str = "cpu"):
        self._model: torch.nn.Module = timm.create_model(model_name, pretrained=True)  # type: ignore
        self._input_resolution = input_resolution
        self._device = device
        self._transform = create_transform(**resolve_data_config(self._model.pretrained_cfg, model=self._model))

        self._model.to(device)
        self._model.eval()

    def get_query(self, piece: Piece, sides: set[Side]) -> dict[Side, npt.NDArray[np.float32]]:
        model_input = self._transform(piece.texture).unsqueeze(0)  # type: ignore
        with torch.no_grad():
            out = self._model(model_input).cpu().numpy()[0]
        return {side: out.copy() for side in sides}

    def get_keys(self, pieces: list[Piece], sides: set[Side]) -> dict[Side, npt.NDArray[np.float32]]:
        model_input = torch.stack([self._transform(piece.texture) for piece in pieces], dim=0)  # type: ignore
        with torch.no_grad():
            out = self._model(model_input).cpu().numpy()
        return {side: out.copy() for side in sides}
