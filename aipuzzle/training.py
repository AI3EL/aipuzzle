import timm
import torch
import torch.nn.functional as F
import torchvision.transforms.functional
from datasets import Image, load_dataset
from torch.utils.data import DataLoader, Dataset

from aipuzzle.env import PieceID, Puzzle, Side, get_side_shifted


def get_labels(puzzle: Puzzle) -> dict[PieceID, dict[Side, PieceID]]:
    out = {}
    solution = puzzle.get_solution()
    for pos, piece_id in solution.items():
        piece = puzzle.pieces[piece_id]
        out[piece.id_] = {}
        for side in Side:
            shifted_pos = get_side_shifted(pos, side)
            if shifted_pos in solution:
                out[piece.id_][side] = solution[shifted_pos]
    return out


def get_key_query_loss(query_logits: torch.Tensor, key_logits: torch.Tensor, temperature: float = 1.0):
    """source: https://github.com/moein-shariatnia/OpenAI-CLIP/blob/master/CLIP.py"""

    assert key_logits.ndim == 2
    logits = (query_logits @ key_logits.T) / temperature
    key_similarity = key_logits @ key_logits.T
    texts_similarity = query_logits @ query_logits.T
    targets = F.softmax((key_similarity + texts_similarity) / 2 * temperature, dim=-1)
    query_loss = F.cross_entropy(logits, targets, reduction="none")  # WARNING: was custom implem in source
    key_loss = F.cross_entropy(logits.T, targets.T, reduction="none")  # WARNING: was custom implem in source
    return ((key_loss + query_loss) / 2.0).mean()  # shape: (batch_size)


def collate_fn(examples):
    return tuple(map(torch.stack, zip(*examples)))


class PieceModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._backbone = timm.create_model("timm/vit_base_patch16_clip_224.openai", pretrained=True)  # type: ignore
        self._head = torch.nn.Linear(512, 4)

    def forward(self, x):
        return self._head(self._backbone(x))


class TorchDataset(Dataset):
    def __init__(self):
        self._query_imgs = []
        self._key_imgs = []
        self._sides = []
        for example in load_dataset("imagenet-1k", split="train[10:20]", trust_remote_code=True):
            puzzle = Puzzle.make_frome_img(example["image"].resize((2240, 2240)), 10, 10)
            for query_id, labels in get_labels(puzzle).items():
                for side, key_id in labels.items():
                    self._query_imgs.append(
                        torchvision.transforms.functional.pil_to_tensor(puzzle.pieces[query_id].texture).float() / 255
                    )
                    self._key_imgs.append(
                        torchvision.transforms.functional.pil_to_tensor(puzzle.pieces[key_id].texture).float() / 255
                    )
                    self._sides.append(torch.tensor(side.value - 1, dtype=torch.int32))

    def __len__(self):
        return len(self._query_imgs)

    def __getitem__(self, idx: int):
        return self._query_imgs[idx], self._key_imgs[idx], self._sides[idx]


if __name__ == "__main__":
    query_model, key_model = PieceModel(), PieceModel()
    opt = torch.optim.Adam(list(query_model.parameters()) + list(key_model.parameters()))  # pyright: ignore[reportPrivateImportUsage]
    query_model.train(), key_model.train()
    dataloader = DataLoader(TorchDataset(), collate_fn=collate_fn, batch_size=3, shuffle=True)
    for query_imgs, key_imgs, sides in dataloader:
        query_embeddings = query_model.forward(query_imgs)
        key_embeddings = key_model.forward(key_imgs)
        loss = get_key_query_loss(query_embeddings[:, sides], key_embeddings[:, sides])
        opt.zero_grad()
        loss.backward()
        opt.step()
