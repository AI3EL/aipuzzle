import torch
import torch.nn.functional as F
import torchvision.transforms.functional  # type: ignore
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from aipuzzle.env import PieceID, PuzzleEnv, Side, get_side_shifted


def get_labels(puzzle: PuzzleEnv) -> dict[PieceID, dict[Side, PieceID]]:
    out = {}
    solution = puzzle.get_solution()
    for pos, piece_id in solution.items():
        piece = puzzle.get_pieces()[piece_id]
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


def collate_fn(examples: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    return tuple(map(torch.stack, zip(*examples)))


class PieceModel(torch.nn.Module):
    def __init__(self, backbone: torch.nn.Module, n_backbone_out: int):
        super().__init__()
        self.backbone = backbone
        self.head = torch.nn.Linear(n_backbone_out, 4 * 256)

    def forward(self, x: torch.Tensor):
        return self.head(self.backbone(x)).reshape(-1, 4, 256)


class PuzzleDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, images: list[Image.Image], n_pieces: tuple[int, int]):
        self._query_imgs = []
        self._key_imgs = []
        self._sides = []
        for img in images:
            puzzle = PuzzleEnv.make_frome_img(img, *n_pieces)
            for query_id, labels in get_labels(puzzle).items():
                for side, key_id in labels.items():
                    self._query_imgs.append(
                        torchvision.transforms.functional.pil_to_tensor(puzzle.get_pieces()[query_id].texture).float()
                        / 255
                    )
                    self._key_imgs.append(
                        torchvision.transforms.functional.pil_to_tensor(puzzle.get_pieces()[key_id].texture).float()
                        / 255
                    )
                    self._sides.append(torch.tensor(side.value - 1, dtype=torch.int32))

    def __len__(self):
        return len(self._query_imgs)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._query_imgs[idx], self._key_imgs[idx], self._sides[idx]


def train(
    query_model: torch.nn.Module,
    key_model: torch.nn.Module,
    dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    n_epoch: int,
):
    query_model.train()
    key_model.train()
    opt = torch.optim.Adam(list(query_model.parameters()) + list(key_model.parameters()))
    for i_epoch in range(n_epoch):
        print(i_epoch)
        for query_imgs, key_imgs, sides in dataloader:
            batch_size = query_imgs.shape[0]
            query_embeddings = query_model.forward(query_imgs)
            key_embeddings = key_model.forward(key_imgs)
            loss = get_key_query_loss(
                query_embeddings[range(batch_size), sides],
                key_embeddings[range(batch_size), sides],
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            print(loss)
