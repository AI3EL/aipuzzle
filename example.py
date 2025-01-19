from pathlib import Path

from aipuzzle.piece_prioritizer import RandomPiecePrioritizer
from aipuzzle.puzzle import Puzzle
from aipuzzle.solver import PrioritizedBruteforceSolver

puzzle = Puzzle.make_frome_image(Path("res", "IMG_0901.JPG"), 10, 10)
partial_layout = {k: puzzle.pieces[v] for k, v in puzzle.get_solution().items() if k in [(0, 0), (0, 1), (1, 2)]}
puzzle.piece_layout_to_image(partial_layout).save(Path("res", "test.png"), "png")

solution = PrioritizedBruteforceSolver(RandomPiecePrioritizer()).solve(puzzle)
bruteforce_layout = {k: puzzle.pieces[v] for k, v in solution.items()}
puzzle.piece_layout_to_image(bruteforce_layout).save(Path("res", "test2.png"), "png")
