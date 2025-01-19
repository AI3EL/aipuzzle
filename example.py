from pathlib import Path

from aipuzzle.img_embedder import PretrainedImgEmbedder
from aipuzzle.piece_prioritizer import KeyQueryPiecePrioritizer, RandomPiecePrioritizer
from aipuzzle.puzzle import Puzzle
from aipuzzle.solver import PrioritizedBruteforceSolver

puzzle = Puzzle.make_frome_image(Path("res", "IMG_0901.JPG"), 5, 5)
partial_layout = {k: puzzle.pieces[v] for k, v in puzzle.get_solution().items() if k in [(0, 0), (0, 1), (1, 2)]}
puzzle.piece_layout_to_image(partial_layout).save(Path("res", "partial.png"), "png")

puzzle.reset()
solution = PrioritizedBruteforceSolver(RandomPiecePrioritizer()).solve(puzzle)
print(puzzle.get_stats())
print(puzzle.is_solution(solution))
bruteforce_layout = {k: puzzle.pieces[v] for k, v in solution.items()}
puzzle.piece_layout_to_image(bruteforce_layout).save(Path("res", "bruteforce.png"), "png")


puzzle.reset()
solution = PrioritizedBruteforceSolver(
    KeyQueryPiecePrioritizer(PretrainedImgEmbedder("timm/vit_base_patch16_clip_224.openai", (224, 224)))
).solve(puzzle)
print(puzzle.get_stats())
print(puzzle.is_solution(solution))
bruteforce_layout = {k: puzzle.pieces[v] for k, v in solution.items()}
puzzle.piece_layout_to_image(bruteforce_layout).save(Path("res", "ai_bruteforce.png"), "png")
