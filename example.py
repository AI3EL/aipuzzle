import time
from pathlib import Path

import timm
from PIL import Image
from timm.data import resolve_data_config  # type: ignore
from timm.data.transforms_factory import create_transform
from torch.utils.data import DataLoader

from aipuzzle.env import PuzzleEnv
from aipuzzle.img_embedder import NNPieceSideEmbedder
from aipuzzle.piece_prioritizer import KeyQueryPiecePrioritizer, RandomPiecePrioritizer
from aipuzzle.training import PieceModel, PuzzleDataset, collate_fn, train
from aipuzzle.worker import AdjacentPuzzleWorker

puzzle = PuzzleEnv.make_frome_img_file(Path("res", "IMG_0901.JPG"), 8, 8)

start_time = time.time()
obs_list = AdjacentPuzzleWorker(RandomPiecePrioritizer()).run_episode(puzzle)
print(f"Took {time.time() - start_time:.2f}s")
print(f"Took {len(obs_list)} attempts")


puzzle.reset()
start_time = time.time()
query_model = PieceModel(timm.create_model("timm/resnet18.a1_in1k", pretrained=True))
key_model = PieceModel(timm.create_model("timm/resnet18.a1_in1k", pretrained=True))
transform = create_transform(**resolve_data_config(query_model.backbone.pretrained_cfg, model=query_model.backbone))

obs_list = AdjacentPuzzleWorker(
    KeyQueryPiecePrioritizer(
        NNPieceSideEmbedder(
            query_model,
            transform,
            (224, 224),
        ),
        NNPieceSideEmbedder(
            key_model,
            transform,
            (224, 224),
        ),
    )
).run_episode(puzzle)
print(f"Took {time.time() - start_time:.2f}s")
print(f"Took {len(obs_list)} attempts")


# images = [
#     x["image"].resize((2240, 2240)) for x in load_dataset("imagenet-1k", split="train[10:20]", trust_remote_code=True)
# ]
images = [Image.open(Path("res", "IMG_0901.JPG")).resize((2240, 2240))]
dataloader = DataLoader(PuzzleDataset(images, (10, 10)), collate_fn=collate_fn, batch_size=32, shuffle=True)
train(query_model, key_model, dataloader, 2)


obs_list = AdjacentPuzzleWorker(
    KeyQueryPiecePrioritizer(
        NNPieceSideEmbedder(
            query_model,
            transform,
            (224, 224),
        ),
        NNPieceSideEmbedder(
            key_model,
            transform,
            (224, 224),
        ),
    )
).run_episode(puzzle)
print(f"Took {time.time() - start_time:.2f}s")
print(f"Took {len(obs_list)} attempts")
