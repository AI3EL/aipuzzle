import time
from pathlib import Path

from aipuzzle.img_embedder import PretrainedImgEmbedder
from aipuzzle.piece_prioritizer import KeyQueryPiecePrioritizer, RandomPiecePrioritizer
from aipuzzle.puzzle import Puzzle
from aipuzzle.solver import PrioritizedBruteforceSolver

puzzle = Puzzle.make_frome_img_file(Path("res", "IMG_0901.JPG"), 10, 10)
partial_layout = {k: puzzle.pieces[v] for k, v in puzzle.get_solution().items() if k in [(0, 0), (0, 1), (1, 2)]}
puzzle.piece_layout_to_image(partial_layout).save(Path("res", "partial.png"), "png")

puzzle.reset()
start_time = time.time()
solution = PrioritizedBruteforceSolver(RandomPiecePrioritizer()).solve(puzzle)
print(f"Took {time.time() - start_time:.2f}")
print(puzzle.get_stats())
print(puzzle.is_solution(solution))
bruteforce_layout = {k: puzzle.pieces[v] for k, v in solution.items()}
puzzle.piece_layout_to_image(bruteforce_layout).save(Path("res", "bruteforce.png"), "png")


puzzle.reset()
start_time = time.time()
solution = PrioritizedBruteforceSolver(
    KeyQueryPiecePrioritizer(PretrainedImgEmbedder("timm/vit_base_patch16_clip_224.openai", (224, 224)))
).solve(puzzle)
print(f"Took {time.time() - start_time:.2f}")
print(puzzle.get_stats())
print(puzzle.is_solution(solution))
bruteforce_layout = {k: puzzle.pieces[v] for k, v in solution.items()}
puzzle.piece_layout_to_image(bruteforce_layout).save(Path("res", "ai_bruteforce.png"), "png")
