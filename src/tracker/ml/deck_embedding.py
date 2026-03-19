"""Deep Sets permutation-invariant deck embedder (ADR-006).

Maps an 8-card deck (represented as vocabulary indices) to a fixed-size
32-dim embedding. Permutation invariance is achieved via element-wise
aggregation (mean + max pooling) over per-card representations.

Architecture:
  Input: (batch, 8) card vocabulary indices
    -> nn.Embedding(vocab_size, 16)
    -> Linear(16, 32) -> ReLU -> Linear(32, 32)  [per-card phi]
    -> cat(mean_pool, max_pool) -> 64-dim
    -> Linear(64, 64) -> ReLU -> Linear(64, 32)  [rho]
  Output: (batch, 32) deck embedding
"""

import torch
import torch.nn as nn


class DeckEmbedder(nn.Module):
    """Permutation-invariant deck embedding via Deep Sets.

    Args:
        vocab_size: Card vocabulary size (including PAD/UNK tokens).
        card_embed_dim: Dimension of per-card embedding lookup.
        hidden_dim: Hidden dimension in phi/rho networks.
        output_dim: Final deck embedding dimension.
    """

    def __init__(
        self,
        vocab_size: int,
        card_embed_dim: int = 16,
        hidden_dim: int = 32,
        output_dim: int = 32,
    ):
        super().__init__()
        self.card_embedding = nn.Embedding(vocab_size, card_embed_dim, padding_idx=0)

        # Per-card transform (phi)
        self.phi = nn.Sequential(
            nn.Linear(card_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Aggregation transform (rho): mean + max -> 2 * hidden_dim
        self.rho = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, output_dim),
        )

        self.output_dim = output_dim

    def forward(self, card_indices: torch.Tensor) -> torch.Tensor:
        """Compute permutation-invariant deck embedding.

        Args:
            card_indices: (batch, num_cards) int64 vocabulary indices.

        Returns:
            (batch, output_dim) deck embedding.
        """
        # (batch, num_cards, card_embed_dim)
        emb = self.card_embedding(card_indices)

        # Per-card transform: (batch, num_cards, hidden_dim)
        phi_out = self.phi(emb)

        # Mask padding (card_index == 0)
        mask = (card_indices != 0).unsqueeze(-1).float()  # (batch, num_cards, 1)
        phi_out = phi_out * mask

        # Aggregation: mean + max pooling over cards
        card_count = mask.sum(dim=1).clamp(min=1)  # (batch, 1)
        mean_pool = phi_out.sum(dim=1) / card_count  # (batch, hidden_dim)

        # Max pool — set padding to -inf
        phi_for_max = phi_out.masked_fill(mask == 0, float("-inf"))
        max_pool = phi_for_max.max(dim=1).values  # (batch, hidden_dim)

        # Concatenate and project
        aggregated = torch.cat([mean_pool, max_pool], dim=1)  # (batch, 2*hidden_dim)
        return self.rho(aggregated)  # (batch, output_dim)

    def load_card_embeddings_from(self, source_embedding: nn.Embedding) -> int:
        """Transfer card embedding weights from another model (e.g. WP model).

        Copies weights for indices that fit in both vocabularies.

        Args:
            source_embedding: Source nn.Embedding to copy from.

        Returns:
            Number of embedding rows transferred.
        """
        src_weight = source_embedding.weight.data
        tgt_weight = self.card_embedding.weight.data

        n_copy = min(src_weight.size(0), tgt_weight.size(0))
        src_dim = src_weight.size(1)
        tgt_dim = tgt_weight.size(1)

        # Copy the overlapping dimensions
        dim_copy = min(src_dim, tgt_dim)
        tgt_weight[:n_copy, :dim_copy] = src_weight[:n_copy, :dim_copy]

        return n_copy
