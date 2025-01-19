import enum
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
import numpy.typing as npt
import PIL
import PIL.Image

# TODO: click unplaced pieces together --> create clusters if pieces


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


PieceID = int


@dataclass
class Piece:
    id_: PieceID
    texture: npt.NDArray[np.uint8]
    plugs: set[Side]
    size: tuple[int, int]


PuzzleSolution = dict[tuple[int, int], PieceID]


class Puzzle:
    @classmethod
    def make_frome_image(cls, path: Path, w: int, h: int) -> Self:
        img = PIL.Image.open(path)
        piece_width = img.width // w
        piece_height = img.height // h
        print(f"{(piece_width, piece_height)=}")
        if ((img.width % w) != 0) or ((img.height % h) != 0):
            new_img_size = (piece_height * w, piece_width * h)
            print(f"resizing {img.size} -> {new_img_size}")
            img = img.resize(new_img_size)
        img_arr = np.array(img)

        pieces = []
        solution = {}
        id_ = 0
        for x in range(w):
            for y in range(h):
                slice_x = slice(x * piece_width, (x + 1) * piece_width)
                slice_y = slice(y * piece_height, (y + 1) * piece_height)

                plugs = set()
                if x != 0:
                    plugs.add(Side.LEFT)
                if x != w - 1:
                    plugs.add(Side.RIGHT)
                if y != 0:
                    plugs.add(Side.DOWN)
                if y != h - 1:
                    plugs.add(Side.UP)

                pieces.append(Piece(id_, img_arr[slice_x, slice_y], plugs, (piece_width, piece_height)))
                solution[(x, y)] = id_
                id_ += 1

        return cls(pieces, (w, h), solution)

    def __init__(self, pieces: list[Piece], dimensions: tuple[int, int], solution: PuzzleSolution):
        self.pieces = pieces
        self.dims = dimensions
        self.piece_size = pieces[0].size
        self._solution = solution
        self._piece_id_to_pos = dict([(v, k) for k, v in solution.items()])

    def size(self) -> tuple[int, int]:
        return (self.dims[0] * self.piece_size[0], self.dims[1] * self.piece_size[1])

    def n_pieces(self) -> int:
        return len(self.pieces)

    def get_solution(self) -> PuzzleSolution:
        return self._solution

    def is_valid_solution(self, solution: PuzzleSolution) -> bool:
        return (set(range(self.n_pieces())) == set(solution.values())) and (
            set(self._solution.keys()) == set(solution.keys())
        )

    def clicks(self, ref_piece: Piece, candidate_piece: Piece, side: Side) -> bool:
        delta_pos = np.array(self._piece_id_to_pos[candidate_piece.id_], np.int32) - np.array(
            self._piece_id_to_pos[ref_piece.id_], np.int32
        )
        return (delta_pos == np.array(side_to_delta_pos(side), np.int32)).all()

    def get_corner_pieces(self) -> set[PieceID]:
        return set(
            [
                self.pieces[self._solution[0, 0]].id_,
                self.pieces[self._solution[0, self.dims[1] - 1]].id_,
                self.pieces[self._solution[self.dims[0] - 1, 0]].id_,
                self.pieces[self._solution[self.dims[0] - 1, self.dims[1] - 1]].id_,
            ]
        )

    def get_exclusive_border_pieces(self) -> set[PieceID]:
        return set(
            sum(
                [
                    [self.pieces[self._solution[x, 0]].id_ for x in range(1, self.dims[0] - 1)],
                    [self.pieces[self._solution[x, self.dims[1] - 1]].id_ for x in range(1, self.dims[0] - 1)],
                    [self.pieces[self._solution[0, y]].id_ for y in range(1, self.dims[1] - 1)],
                    [self.pieces[self._solution[self.dims[0] - 1, y]].id_ for y in range(1, self.dims[1] - 1)],
                ],
                [],
            )
        )

    def piece_layout_to_image(self, piece_layout: dict[tuple[int, int], Piece]) -> PIL.Image.Image:
        out = np.zeros(self.size() + (3,), np.uint8)
        for (i, j), piece in piece_layout.items():
            w, h = piece.size
            out[i * w : (i + 1) * w, j * h : (j + 1) * h] = piece.texture
        return PIL.Image.fromarray(out)
