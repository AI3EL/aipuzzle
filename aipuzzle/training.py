import torch
import torch.nn.functional as F

from aipuzzle.puzzle import PieceID, Puzzle, Side, get_side_shifted


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


def get_key_query_loss(
    query_logits: dict[Side, torch.Tensor], key_logits: dict[Side, torch.Tensor], temperature: float = 1.0
):
    """source: https://github.com/moein-shariatnia/OpenAI-CLIP/blob/master/CLIP.py"""

    loss = torch.zeros(0, dtype=torch.float32, device=query_logits[Side.UP].device)
    for side in Side:
        assert key_logits[side].ndim == 2
        logits = (query_logits[side] @ key_logits[side].T) / temperature
        key_similarity = key_logits[side] @ key_logits[side].T
        texts_similarity = query_logits[side] @ query_logits[side].T
        targets = F.softmax((key_similarity + texts_similarity) / 2 * temperature, dim=-1)
        query_loss = F.cross_entropy(logits, targets, reduction="none")  # WARNING: was custom implem in source
        key_loss = F.cross_entropy(logits.T, targets.T, reduction="none")  # WARNING: was custom implem in source
        loss = loss + ((key_loss + query_loss) / 2.0).mean()  # shape: (batch_size)
    return loss
