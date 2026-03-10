"""Win Probability Model (ADR-004).

Causal TCN that produces P(win) at every game tick. Reuses the TCN
encoder architecture from ADR-003 but replaces global pooling with
a per-tick classification head.

Architecture:
  Input: Event sequence (card_id + 17-dim features per event)
    → Card Embedding: nn.Embedding(vocab_size, 16)
    → Concatenate → (batch, seq_len, 33)
    → Transpose → (batch, 33, seq_len)
    → TCN Encoder: 6 TemporalBlocks (causal dilated convolutions)
      channels: [33→64, 64→64, 64→128, 128→128, 128→256, 256→256]
    → Per-tick head: Linear(256→64) → ReLU → Dropout → Linear(64→1)
    → Output: P(win) at each tick
"""

import torch
import torch.nn as nn

from tracker.ml.tcn import TCNEncoder


class WinProbabilityModel(nn.Module):
    """Causal TCN with per-tick win probability head.

    Args:
        vocab_size: Number of unique cards (including special tokens).
        card_embed_dim: Card embedding dimension.
        feature_dim: Hand-crafted feature dimension per event.
        tcn_channels: Channel sizes for TCN blocks.
        kernel_size: TCN kernel size.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        vocab_size: int,
        card_embed_dim: int = 16,
        feature_dim: int = 17,
        tcn_channels: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
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

        # Per-tick classification head: (batch, 256, seq_len) → (batch, 1, seq_len)
        out_ch = self.tcn.output_channels  # 256
        self.head = nn.Sequential(
            nn.Conv1d(out_ch, 64, 1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(64, 1, 1),
        )

    def forward(
        self,
        card_ids: torch.Tensor,
        features: torch.Tensor,
        lengths: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass producing per-tick logits.

        Args:
            card_ids: (batch, seq_len) int64 — card vocabulary indices.
            features: (batch, seq_len, feature_dim) float32.
            lengths: (batch,) int64 — original sequence lengths.

        Returns:
            logits: (batch, seq_len) — raw logits per tick (apply sigmoid for P(win)).
        """
        card_emb = self.card_embedding(card_ids)
        combined = torch.cat([card_emb, features], dim=2)
        combined = combined.transpose(1, 2)  # (batch, channels, seq_len)

        tcn_out = self.tcn(combined)  # (batch, 256, seq_len)
        logits = self.head(tcn_out).squeeze(1)  # (batch, seq_len)

        return logits

    @classmethod
    def from_pretrained_tcn(
        cls,
        tcn_checkpoint_path: str,
        vocab_size: int,
        device: torch.device,
        freeze_encoder: bool = True,
        dropout: float = 0.2,
    ) -> "WinProbabilityModel":
        """Initialize from a trained ADR-003 TCN checkpoint.

        Loads card embedding and TCN encoder weights from GameEmbeddingModel,
        initializes a fresh per-tick head.

        Args:
            tcn_checkpoint_path: Path to tcn_v1.pt checkpoint.
            vocab_size: Card vocabulary size.
            device: Target device.
            freeze_encoder: Whether to freeze card embedding + TCN encoder weights.
            dropout: Dropout for the head.

        Returns:
            WinProbabilityModel with pretrained encoder weights.
        """
        checkpoint = torch.load(tcn_checkpoint_path, map_location=device, weights_only=True)
        saved_vocab = checkpoint.get("vocab_size", vocab_size)

        model = cls(vocab_size=saved_vocab, dropout=dropout)

        # Load matching weights from GameEmbeddingModel state dict
        source_state = checkpoint["model_state_dict"]
        target_state = model.state_dict()

        transferred = 0
        for key in target_state:
            # Match card_embedding and tcn weights
            if key in source_state and target_state[key].shape == source_state[key].shape:
                target_state[key] = source_state[key]
                transferred += 1

        model.load_state_dict(target_state)

        if freeze_encoder:
            for name, param in model.named_parameters():
                if name.startswith("card_embedding.") or name.startswith("tcn."):
                    param.requires_grad = False

        model.to(device)
        return model
