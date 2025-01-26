import enum
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

import numpy as np
from PIL import Image


class Side(enum.Enum):
    UP = 1
    RIGHT = 2
    DOWN = 3
    LEFT = 4


def get_opposite_side(side: Side) -> Side:
    if side == Side.UP:
        return Side.DOWN
    if side == Side.RIGHT:
        return Side.LEFT
    if side == Side.DOWN:
        return Side.UP
    if side == Side.LEFT:
        return Side.RIGHT


def side_to_delta_pos(side: Side) -> tuple[int, int]:
    if side == Side.UP:
        return (0, 1)
    if side == Side.RIGHT:
        return (1, 0)
    if side == Side.DOWN:
        return (0, -1)
    if side == Side.LEFT:
        return (-1, 0)
    raise ValueError


def get_side_shifted(pos: tuple[int, int], side: Side) -> tuple[int, int]:
    arr_click_pos = np.array(pos, np.int32) + np.array(side_to_delta_pos(side), np.int32)
    return int(arr_click_pos[0]), int(arr_click_pos[1])


PieceID = int


@dataclass
class Piece:
    id_: PieceID
    texture: Image.Image
    plugs: set[Side]
    size: tuple[int, int]  # (w, h)


PuzzleSolution = dict[tuple[int, int], PieceID]


@dataclass
class PuzzleObs:
    pieces: list[Piece]
    layout: dict[tuple[int, int], PieceID]
    dims: tuple[int, int]

    def render(self) -> Image.Image:
        image_w = self.dims[0] * self.pieces[0].size[0]
        image_h = self.dims[1] * self.pieces[0].size[1]
        out = np.zeros((image_h, image_w, 3), np.uint8)
        for (i, j), piece_id in self.layout.items():
            piece = self.pieces[piece_id]
            w, h = piece.size
            out[j * h : (j + 1) * h, i * w : (i + 1) * w] = np.array(piece.texture)
        return Image.fromarray(out)


@dataclass
class PuzzleAction:
    piece_id: int
    pos: tuple[int, int]


class PuzzleEnv:
    @classmethod
    def make_frome_img_file(cls, path: Path, w: int, h: int) -> Self:
        return cls.make_frome_img(Image.open(path), w, h)

    @classmethod
    def make_frome_img(cls, img: Image.Image, w: int, h: int) -> Self:
        piece_width = img.width // w
        piece_height = img.height // h
        print(f"{(piece_width, piece_height)=}")
        if ((img.width % w) != 0) or ((img.height % h) != 0):
            new_img_size = (piece_width * w, piece_height * h)
            print(f"resizing {img.size} -> {new_img_size}")
            img = img.resize(new_img_size)

        pieces = []
        solution = {}
        id_ = 0
        for x in range(w):
            for y in range(h):
                plugs = set()
                if x != 0:
                    plugs.add(Side.LEFT)
                if x != w - 1:
                    plugs.add(Side.RIGHT)
                if y != 0:
                    plugs.add(Side.DOWN)
                if y != h - 1:
                    plugs.add(Side.UP)

                crop_box = (x * piece_width, y * piece_height, (x + 1) * piece_width, (y + 1) * piece_height)
                pieces.append(Piece(id_, img.crop(crop_box), plugs, (piece_width, piece_height)))
                solution[(x, y)] = id_
                id_ += 1

        return cls(pieces, (w, h), solution)

    def __init__(self, pieces: list[Piece], dimensions: tuple[int, int], solution: PuzzleSolution):
        self._pieces = pieces
        self._dims = dimensions
        self._piece_size = pieces[0].size
        self._solution = solution
        self._piece_id_to_pos = dict([(v, k) for k, v in solution.items()])

        self._layout: dict[tuple[int, int], PieceID] = {}
        self._piece_to_layout_pos: dict[PieceID, tuple[int, int]] = {}

    def _get_obs(self) -> PuzzleObs:
        return PuzzleObs(self._pieces, self._layout.copy(), self._dims)

    def reset(self) -> PuzzleObs:
        self._layout = {}
        self._piece_to_layout_pos = {}
        return self._get_obs()

    def step(self, action: PuzzleAction) -> tuple[PuzzleObs, float, bool, dict[str, Any]]:
        print(action)
        self._check_action(action)
        if not self._is_insertable(action):
            return self._get_obs(), 0.0, False, {}
        self._layout[action.pos] = action.piece_id
        self._piece_to_layout_pos[action.piece_id] = action.pos
        obs = self._get_obs()
        done = self._layout == self._solution
        return obs, float(done), done, {}

    def _is_insertable(self, action: PuzzleAction) -> bool:
        for side in Side:
            neigh_pos = get_side_shifted(action.pos, side)
            if neigh_pos in self._layout:
                neigh_piece = self._layout[neigh_pos]
                if not self._clicks(action.piece_id, neigh_piece, side):
                    return False
        return True

    def _clicks(self, ref_piece_id: PieceID, candidate_piece_id: PieceID, side: Side) -> bool:
        delta_pos = np.array(self._piece_id_to_pos[candidate_piece_id], np.int32) - np.array(
            self._piece_id_to_pos[ref_piece_id], np.int32
        )
        return (delta_pos == np.array(side_to_delta_pos(side), np.int32)).all()

    def _check_action(self, action: PuzzleAction):
        if action.pos in self._layout:
            raise ValueError
        if action.piece_id in self._piece_to_layout_pos:
            raise ValueError

    def get_solution(self) -> PuzzleSolution:
        return self._solution


def get_any_corner(obs: PuzzleObs) -> Piece:
    for piece in obs.pieces:
        if len(piece.plugs) != 2:
            continue
        return piece
    raise ValueError


def get_free_plug_piece(obs: PuzzleObs) -> tuple[tuple[int, int], PieceID, set[Side]]:
    for pos, piece_id in obs.layout.items():
        for side in obs.pieces[piece_id].plugs:
            free_plugs = set()
            if get_side_shifted(pos, side) not in obs.layout:
                free_plugs.add(side)
            if free_plugs:
                return pos, piece_id, free_plugs
    raise ValueError


def get_unplaced_pieces(obs: PuzzleObs) -> list[Piece]:
    out = []
    inversed_layout = {v: k for k, v in obs.layout.items()}
    for piece in obs.pieces:
        if piece.id_ not in inversed_layout:
            out.append(piece)
    return out
