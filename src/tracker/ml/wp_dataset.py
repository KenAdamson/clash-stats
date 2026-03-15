"""Dataset for win probability training (ADR-004).

Adapts SequenceDataset's feature extraction for per-tick labels.
Each tick gets the final game result as its label. A mask tracks
which ticks are real (not padding) for loss computation.
"""

import logging

import torch

from tracker.ml.sequence_dataset import SequenceDataset, collate_fn as _base_collate

logger = logging.getLogger(__name__)


def wp_collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor, float]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad sequences and create per-tick labels + mask.

    Returns:
        card_ids: (batch, max_len) int64
        features: (batch, max_len, 17) float32
        lengths: (batch,) int64
        labels: (batch, max_len) float32 — game result broadcast to all ticks
        mask: (batch, max_len) float32 — 1.0 for real ticks, 0.0 for padding
    """
    card_ids, features, lengths, game_labels = _base_collate(batch)
    batch_size = card_ids.size(0)
    max_len = card_ids.size(1)

    # Broadcast game-level label to all ticks
    labels = game_labels.unsqueeze(1).expand(batch_size, max_len)

    # Build mask from lengths
    arange = torch.arange(max_len).unsqueeze(0)  # (1, max_len)
    mask = (arange < lengths.unsqueeze(1)).float()  # (batch, max_len)

    return card_ids, features, lengths, labels, mask
