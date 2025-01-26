import time
from pathlib import Path

from aipuzzle.env import PuzzleEnv
from aipuzzle.img_embedder import PretrainedImgEmbedder
from aipuzzle.piece_prioritizer import KeyQueryPiecePrioritizer, RandomPiecePrioritizer
from aipuzzle.worker import AdjacentPuzzleWorker

puzzle = PuzzleEnv.make_frome_img_file(Path("res", "IMG_0901.JPG"), 4, 4)

start_time = time.time()
obs_list = AdjacentPuzzleWorker(RandomPiecePrioritizer()).run_episode(puzzle)
print(f"Took {time.time() - start_time:.2f}")


puzzle.reset()
start_time = time.time()
obs_list = AdjacentPuzzleWorker(
    KeyQueryPiecePrioritizer(PretrainedImgEmbedder("timm/vit_base_patch16_clip_224.openai", (224, 224)))
).run_episode(puzzle)
print(f"Took {time.time() - start_time:.2f}")
