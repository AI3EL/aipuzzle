import enum
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
from PIL import Image

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
    texture: Image.Image
    plugs: set[Side]
    size: tuple[int, int]  # (w, h)


PuzzleSolution = dict[tuple[int, int], PieceID]


class Puzzle:
    @classmethod
    def make_frome_image(cls, path: Path, w: int, h: int) -> Self:
        img = Image.open(path)
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
        self.pieces = pieces
        self.dims = dimensions
        self.piece_size = pieces[0].size
        self._solution = solution
        self._piece_id_to_pos = dict([(v, k) for k, v in solution.items()])

        self._failed_click_counter = 0
        self._succesful_click_counter = 0

    def reset(self):
        self._failed_click_counter = 0
        self._succesful_click_counter = 0

    def is_solution(self, solution_candidate: PuzzleSolution):
        return self._solution == solution_candidate

    def get_stats(self) -> dict[str, str]:
        click_count = self._failed_click_counter + self._succesful_click_counter
        success_rate = self._succesful_click_counter / click_count if click_count else np.nan
        return {
            "failed_click_counter": str(self._failed_click_counter),
            "succesful_click_counter": str(self._succesful_click_counter),
            "click_count": str(click_count),
            "click_success_rate": f"{success_rate:.3f}",
        }

    def get_image_w(self) -> int:
        return self.dims[0] * self.piece_size[0]

    def get_image_h(self) -> int:
        return self.dims[1] * self.piece_size[1]

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
        out = (delta_pos == np.array(side_to_delta_pos(side), np.int32)).all()

        if out:
            self._succesful_click_counter += 1
        else:
            self._failed_click_counter += 1

        return out

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

    def piece_layout_to_image(self, piece_layout: dict[tuple[int, int], Piece]) -> Image.Image:
        out = np.zeros((self.get_image_h(), self.get_image_w(), 3), np.uint8)
        for (i, j), piece in piece_layout.items():
            w, h = piece.size
            out[j * h : (j + 1) * h, i * w : (i + 1) * w] = np.array(piece.texture)
        return Image.fromarray(out)
