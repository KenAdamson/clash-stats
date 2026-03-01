"""Temporal Convolutional Network for game state embeddings (ADR-003 Phase 1).

Architecture:
  Input (batch, seq_len, 17) hand-crafted features
  → Card Embedding: nn.Embedding(vocab_size, 16)
  → Concatenate → (batch, seq_len, 33)
  → Transpose → (batch, 33, seq_len) for Conv1d
  → TCN Encoder: 6 TemporalBlocks with dilated causal convolutions
    channels: [33→64, 64→64, 64→128, 128→128, 128→256, 256→256]
    dilations: [1, 2, 4, 8, 16, 32]
  → Global Pooling: concat [mean_pool, max_pool, last_hidden] → 768
  → Projection: Linear(768→256) → ReLU → Linear(256→128)
  → Game Embedding: 128-dim
  → Classification Head: Linear(128→1) → Sigmoid [for training]
"""

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import weight_norm


class TemporalBlock(nn.Module):
    """Dilated causal convolution block with residual connection.

    Each block: conv1d → batchnorm → relu → dropout → conv1d → batchnorm → relu → dropout
    Plus a residual skip connection (1x1 conv if channels change).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float = 0.2,
    ):
        super().__init__()
        padding = (kernel_size - 1) * dilation  # causal padding on left

        self.conv1 = weight_norm(nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=padding,
        ))
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = weight_norm(nn.Conv1d(
            out_channels, out_channels, kernel_size,
            dilation=dilation, padding=padding,
        ))
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # Residual connection — 1x1 conv if channel dimensions differ
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

        self._padding = padding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, channels, seq_len)

        Returns:
            (batch, out_channels, seq_len)
        """
        residual = self.residual(x)

        # First conv block
        out = self.conv1(x)
        out = out[:, :, : x.size(2)]  # trim causal padding from right
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        # Second conv block
        out = self.conv2(out)
        out = out[:, :, : x.size(2)]  # trim causal padding from right
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout(out)

        return self.relu(out + residual)


class TCNEncoder(nn.Module):
    """Stack of TemporalBlocks with increasing dilation.

    Args:
        input_channels: Input feature dimension.
        channel_sizes: List of output channels per block.
        kernel_size: Convolution kernel size.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        input_channels: int,
        channel_sizes: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        if channel_sizes is None:
            channel_sizes = [64, 64, 128, 128, 256, 256]

        layers = []
        num_levels = len(channel_sizes)
        for i in range(num_levels):
            dilation = 2 ** i
            in_ch = input_channels if i == 0 else channel_sizes[i - 1]
            out_ch = channel_sizes[i]
            layers.append(TemporalBlock(
                in_ch, out_ch, kernel_size, dilation, dropout,
            ))

        self.network = nn.Sequential(*layers)
        self.output_channels = channel_sizes[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all temporal blocks.

        Args:
            x: (batch, input_channels, seq_len)

        Returns:
            (batch, output_channels, seq_len)
        """
        return self.network(x)


class GameEmbeddingModel(nn.Module):
    """Full TCN model: card embedding + TCN encoder + pooling + projection.

    Args:
        vocab_size: Number of unique cards (including special tokens).
        card_embed_dim: Card embedding dimension.
        feature_dim: Hand-crafted feature dimension per event (excluding card_id).
        tcn_channels: Channel sizes for TCN blocks.
        kernel_size: TCN kernel size.
        dropout: Dropout rate.
        embedding_dim: Output embedding dimension.
    """

    def __init__(
        self,
        vocab_size: int,
        card_embed_dim: int = 16,
        feature_dim: int = 17,
        tcn_channels: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
        embedding_dim: int = 128,
    ):
        super().__init__()
        self.card_embedding = nn.Embedding(vocab_size, card_embed_dim, padding_idx=0)
        self.feature_dim = feature_dim

        input_channels = card_embed_dim + feature_dim  # 16 + 17 = 33

        self.tcn = TCNEncoder(
            input_channels=input_channels,
            channel_sizes=tcn_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        # Global pooling output: mean + max + last → 3 * output_channels
        pool_dim = 3 * self.tcn.output_channels

        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(pool_dim, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim),
        )

        # Classification head (for training)
        self.classifier = nn.Linear(embedding_dim, 1)

    def _masked_pool(
        self,
        tcn_out: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Global pooling with mask for padding positions.

        Args:
            tcn_out: (batch, channels, seq_len)
            lengths: (batch,) original sequence lengths

        Returns:
            (batch, 3 * channels) — concat of mean, max, last
        """
        batch_size, channels, seq_len = tcn_out.shape
        device = tcn_out.device

        # Build mask: (batch, 1, seq_len)
        arange = torch.arange(seq_len, device=device).unsqueeze(0)
        mask = arange < lengths.unsqueeze(1)
        mask = mask.unsqueeze(1).float()  # (batch, 1, seq_len)

        # Masked mean pooling
        masked = tcn_out * mask
        mean_pool = masked.sum(dim=2) / lengths.unsqueeze(1).float().clamp(min=1)

        # Masked max pooling — set padding to -inf
        masked_for_max = tcn_out.masked_fill(mask == 0, float("-inf"))
        max_pool = masked_for_max.max(dim=2).values

        # Last valid hidden state
        last_indices = (lengths - 1).clamp(min=0).long()
        last_hidden = tcn_out[
            torch.arange(batch_size, device=device), :, last_indices
        ]

        return torch.cat([mean_pool, max_pool, last_hidden], dim=1)

    def forward(
        self,
        card_ids: torch.Tensor,
        features: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            card_ids: (batch, seq_len) int64 — card vocabulary indices
            features: (batch, seq_len, feature_dim) float32 — hand-crafted features
            lengths: (batch,) int64 — original sequence lengths

        Returns:
            embeddings: (batch, embedding_dim) — 128-dim game embeddings
            logits: (batch, 1) — raw logits for win probability
        """
        # Card embedding: (batch, seq_len) → (batch, seq_len, card_embed_dim)
        card_emb = self.card_embedding(card_ids)

        # Concatenate: (batch, seq_len, card_embed_dim + feature_dim)
        combined = torch.cat([card_emb, features], dim=2)

        # Transpose to channels-first for Conv1d: (batch, channels, seq_len)
        combined = combined.transpose(1, 2)

        # TCN encoder
        tcn_out = self.tcn(combined)

        # Global pooling with mask
        pooled = self._masked_pool(tcn_out, lengths)

        # Projection → embedding
        embeddings = self.projection(pooled)

        # Classification head
        logits = self.classifier(embeddings)

        return embeddings, logits
