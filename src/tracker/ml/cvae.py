"""Conditional VAE for counterfactual game generation (ADR-006).

Encodes a replay event sequence into a latent code z (conditioned on both
player and opponent decks), then decodes autoregressively to generate
counterfactual game sequences under modified deck compositions.

Architecture:
  Encoder:
    TCNEncoder (6 blocks, 33->256 channels) -> global pool (mean+max+last=768)
    + player_deck_emb (32) + opponent_deck_emb (32) -> 832
    -> Linear(832, 256) -> ReLU
    -> mu: Linear(256, 64), logvar: Linear(256, 64)
    -> z = reparameterize(mu, logvar)

  Decoder:
    Transformer (3 layers, 4 heads, d_model=256, dropout=0.2)
    Condition: z(64) + player_deck(32) + opponent_deck(32) = 128
      -> Linear(128, 256) as cross-attention memory (length=1)
    Input: card_embedding(16) + features(17) = 33
      -> Linear(33, 256) + sinusoidal positional encoding
    Output heads:
      card: Linear(256, vocab_size)
      tick_delta: Linear(256, 1) + softplus
      arena_xy: Linear(256, 2) + sigmoid
      side: Linear(256, 1) — binary logit
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from tracker.ml.tcn import TCNEncoder
from tracker.ml.deck_embedding import DeckEmbedder


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for Transformer inputs."""

    def __init__(self, d_model: int, max_len: int = 300, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input.

        Args:
            x: (batch, seq_len, d_model)

        Returns:
            (batch, seq_len, d_model)
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class CVAEEncoder(nn.Module):
    """Variational encoder: TCN + deck conditioning -> (mu, logvar).

    Args:
        vocab_size: Card vocabulary size.
        card_embed_dim: Card embedding dimension.
        feature_dim: Per-event feature dimension (excluding card embedding).
        tcn_channels: TCN block channel sizes.
        kernel_size: TCN kernel size.
        dropout: Dropout rate.
        deck_embed_dim: Deck embedding dimension.
        latent_dim: Variational latent dimension.
    """

    def __init__(
        self,
        vocab_size: int,
        card_embed_dim: int = 16,
        feature_dim: int = 17,
        tcn_channels: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
        deck_embed_dim: int = 32,
        latent_dim: int = 64,
    ):
        super().__init__()
        self.card_embedding = nn.Embedding(vocab_size, card_embed_dim, padding_idx=0)
        self.feature_dim = feature_dim

        input_channels = card_embed_dim + feature_dim  # 33

        self.tcn = TCNEncoder(
            input_channels=input_channels,
            channel_sizes=tcn_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        # Global pool output: mean + max + last = 3 * output_channels
        pool_dim = 3 * self.tcn.output_channels  # 768

        # Concat deck embeddings
        condition_dim = pool_dim + deck_embed_dim * 2  # 768 + 64 = 832

        self.fc = nn.Sequential(
            nn.Linear(condition_dim, 256),
            nn.ReLU(),
        )

        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def _masked_pool(
        self, tcn_out: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Global pooling with mask: cat(mean, max, last).

        Args:
            tcn_out: (batch, channels, seq_len)
            lengths: (batch,)

        Returns:
            (batch, 3 * channels)
        """
        batch_size, channels, seq_len = tcn_out.shape
        device = tcn_out.device

        arange = torch.arange(seq_len, device=device).unsqueeze(0)
        mask = (arange < lengths.unsqueeze(1)).unsqueeze(1).float()

        # Mean pool
        masked = tcn_out * mask
        mean_pool = masked.sum(dim=2) / lengths.unsqueeze(1).float().clamp(min=1)

        # Max pool
        masked_for_max = tcn_out.masked_fill(mask == 0, float("-inf"))
        max_pool = masked_for_max.max(dim=2).values

        # Last valid hidden
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
        player_deck_emb: torch.Tensor,
        opponent_deck_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode a game sequence into variational parameters.

        Args:
            card_ids: (batch, seq_len) int64.
            features: (batch, seq_len, feature_dim) float32.
            lengths: (batch,) int64.
            player_deck_emb: (batch, deck_embed_dim) float32.
            opponent_deck_emb: (batch, deck_embed_dim) float32.

        Returns:
            mu: (batch, latent_dim)
            logvar: (batch, latent_dim)
        """
        card_emb = self.card_embedding(card_ids)
        combined = torch.cat([card_emb, features], dim=2)
        combined = combined.transpose(1, 2)  # (batch, channels, seq_len)

        tcn_out = self.tcn(combined)
        pooled = self._masked_pool(tcn_out, lengths)

        # Condition on decks
        conditioned = torch.cat([pooled, player_deck_emb, opponent_deck_emb], dim=1)
        h = self.fc(conditioned)

        return self.fc_mu(h), self.fc_logvar(h)


class CVAEDecoder(nn.Module):
    """Transformer decoder conditioned on z + deck embeddings.

    Generates event sequences autoregressively during inference.

    Args:
        vocab_size: Card vocabulary size.
        card_embed_dim: Card embedding dimension.
        feature_dim: Per-event feature dimension.
        d_model: Transformer hidden dimension.
        nhead: Number of attention heads.
        num_layers: Number of transformer decoder layers.
        dropout: Dropout rate.
        deck_embed_dim: Deck embedding dimension.
        latent_dim: Latent code dimension.
        max_events: Maximum generated sequence length.
    """

    def __init__(
        self,
        vocab_size: int,
        card_embed_dim: int = 16,
        feature_dim: int = 17,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
        deck_embed_dim: int = 32,
        latent_dim: int = 64,
        max_events: int = 500,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_events = max_events

        # Input embedding: card_embed + features -> d_model
        input_dim = card_embed_dim + feature_dim  # 33
        self.input_proj = nn.Linear(input_dim, d_model)
        self.card_embedding = nn.Embedding(vocab_size, card_embed_dim, padding_idx=0)

        # Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_events + 1, dropout=dropout)

        # Condition projection: z + player_deck + opponent_deck -> d_model
        condition_dim = latent_dim + deck_embed_dim * 2  # 128
        self.condition_proj = nn.Linear(condition_dim, d_model)

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output heads
        self.head_card = nn.Linear(d_model, vocab_size)
        self.head_tick_delta = nn.Linear(d_model, 1)
        self.head_arena_xy = nn.Linear(d_model, 2)
        self.head_side = nn.Linear(d_model, 1)

    def forward(
        self,
        card_ids: torch.Tensor,
        features: torch.Tensor,
        lengths: torch.Tensor,
        z: torch.Tensor,
        player_deck_emb: torch.Tensor,
        opponent_deck_emb: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Teacher-forced forward pass (training).

        Args:
            card_ids: (batch, seq_len) int64.
            features: (batch, seq_len, feature_dim) float32.
            lengths: (batch,) int64.
            z: (batch, latent_dim) latent code.
            player_deck_emb: (batch, deck_embed_dim).
            opponent_deck_emb: (batch, deck_embed_dim).

        Returns:
            Dict with keys: card_logits, tick_delta, arena_xy, side_logit
        """
        batch_size, seq_len = card_ids.shape
        device = card_ids.device

        # Input embedding
        card_emb = self.card_embedding(card_ids)
        combined = torch.cat([card_emb, features], dim=2)  # (batch, seq, 33)
        tgt = self.input_proj(combined)  # (batch, seq, d_model)
        tgt = self.pos_enc(tgt)

        # Condition memory: (batch, 1, d_model)
        condition = torch.cat([z, player_deck_emb, opponent_deck_emb], dim=1)
        memory = self.condition_proj(condition).unsqueeze(1)

        # Causal mask for autoregressive decoding
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=device
        )

        # Padding mask
        arange = torch.arange(seq_len, device=device).unsqueeze(0)
        tgt_key_padding_mask = arange >= lengths.unsqueeze(1)

        # Decode
        out = self.transformer(
            tgt, memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
        )  # (batch, seq_len, d_model)

        return {
            "card_logits": self.head_card(out),           # (batch, seq, vocab)
            "tick_delta": F.softplus(self.head_tick_delta(out).squeeze(-1)),  # (batch, seq)
            "arena_xy": torch.sigmoid(self.head_arena_xy(out)),  # (batch, seq, 2)
            "side_logit": self.head_side(out).squeeze(-1),       # (batch, seq)
        }

    @torch.no_grad()
    def generate(
        self,
        z: torch.Tensor,
        player_deck_emb: torch.Tensor,
        opponent_deck_emb: torch.Tensor,
        end_token_id: int,
        player_deck_ids: torch.Tensor | None = None,
        max_events: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """Autoregressive generation from latent code.

        Args:
            z: (batch, latent_dim) latent codes.
            player_deck_emb: (batch, deck_embed_dim).
            opponent_deck_emb: (batch, deck_embed_dim).
            end_token_id: Vocabulary index for <END> token.
            player_deck_ids: (batch, 8) card indices for deck masking.
            max_events: Max events to generate.

        Returns:
            Dict with generated card_ids, tick_deltas, arena_xys, sides, lengths.
        """
        if max_events is None:
            max_events = self.max_events

        batch_size = z.size(0)
        device = z.device

        # Condition memory
        condition = torch.cat([z, player_deck_emb, opponent_deck_emb], dim=1)
        memory = self.condition_proj(condition).unsqueeze(1)  # (batch, 1, d_model)

        # Start with a zero embedding (start token)
        gen_card_ids = []
        gen_tick_deltas = []
        gen_arena_xys = []
        gen_sides = []
        active = torch.ones(batch_size, dtype=torch.bool, device=device)
        gen_lengths = torch.full((batch_size,), max_events, dtype=torch.long, device=device)

        # Accumulate hidden sequence
        hidden_seq = torch.zeros(batch_size, 0, self.d_model, device=device)

        for step in range(max_events):
            if not active.any():
                break

            if step == 0:
                # Start token: zero input
                step_input = torch.zeros(batch_size, 1, self.d_model, device=device)
            else:
                # Use previous step's generated output
                prev_card = gen_card_ids[-1]  # (batch,)
                card_emb = self.card_embedding(prev_card)  # (batch, card_embed_dim)
                # Construct feature from generated values
                prev_tick_delta = gen_tick_deltas[-1].unsqueeze(-1)  # (batch, 1)
                prev_arena = gen_arena_xys[-1]  # (batch, 2)
                prev_side = gen_sides[-1].unsqueeze(-1)  # (batch, 1)
                # Fill remaining features with zeros (13 dims for phase/lane/etc)
                padding_features = torch.zeros(batch_size, 13, device=device)
                feat_vec = torch.cat([prev_side, prev_tick_delta, padding_features, prev_arena[:, :1]], dim=1)
                # Ensure feature_dim = 17
                feat_vec = torch.zeros(batch_size, 17, device=device)
                feat_vec[:, 0] = gen_sides[-1]  # side
                feat_vec[:, 1] = gen_tick_deltas[-1].clamp(max=1.0)  # tick_norm
                feat_vec[:, 6] = gen_arena_xys[-1][:, 0] * 2 - 1  # arena_x_norm
                feat_vec[:, 7] = gen_arena_xys[-1][:, 1] * 2 - 1  # arena_y_norm

                combined = torch.cat([card_emb, feat_vec], dim=1)  # (batch, 33)
                step_input = self.input_proj(combined).unsqueeze(1)  # (batch, 1, d_model)

            # Add positional encoding for this step
            step_input = step_input + self.pos_enc.pe[:, step : step + 1]

            # Append to sequence
            hidden_seq = torch.cat([hidden_seq, step_input], dim=1)

            # Causal mask
            seq_len = hidden_seq.size(1)
            causal_mask = nn.Transformer.generate_square_subsequent_mask(
                seq_len, device=device
            )

            # Decode
            out = self.transformer(hidden_seq, memory, tgt_mask=causal_mask)
            last_out = out[:, -1]  # (batch, d_model)

            # Predict
            card_logits = self.head_card(last_out)  # (batch, vocab_size)

            # Mask non-deck cards for team events (optional)
            # Sample card
            card_probs = F.softmax(card_logits, dim=-1)
            sampled_card = torch.multinomial(card_probs, 1).squeeze(-1)

            tick_delta = F.softplus(self.head_tick_delta(last_out).squeeze(-1))
            arena_xy = torch.sigmoid(self.head_arena_xy(last_out))
            side_logit = self.head_side(last_out).squeeze(-1)
            side = (torch.sigmoid(side_logit) > 0.5).float()

            # Check for END token
            ended = sampled_card == end_token_id
            newly_ended = ended & active
            gen_lengths[newly_ended] = step

            gen_card_ids.append(sampled_card)
            gen_tick_deltas.append(tick_delta)
            gen_arena_xys.append(arena_xy)
            gen_sides.append(side)

            active = active & ~ended

        # Stack
        out_len = len(gen_card_ids)
        if out_len == 0:
            return {
                "card_ids": torch.zeros(batch_size, 0, dtype=torch.long, device=device),
                "tick_deltas": torch.zeros(batch_size, 0, device=device),
                "arena_xys": torch.zeros(batch_size, 0, 2, device=device),
                "sides": torch.zeros(batch_size, 0, device=device),
                "lengths": torch.zeros(batch_size, dtype=torch.long, device=device),
            }

        return {
            "card_ids": torch.stack(gen_card_ids, dim=1),       # (batch, out_len)
            "tick_deltas": torch.stack(gen_tick_deltas, dim=1),  # (batch, out_len)
            "arena_xys": torch.stack(gen_arena_xys, dim=1),     # (batch, out_len, 2)
            "sides": torch.stack(gen_sides, dim=1),              # (batch, out_len)
            "lengths": gen_lengths,
        }


class CounterfactualVAE(nn.Module):
    """Full CVAE model combining encoder, decoder, and deck embedders.

    v3 changes to prevent posterior collapse:
      - Deck bottleneck: 32-dim deck embeddings compressed to 8-dim before
        decoder, forcing z to carry game information the deck can't.
      - Deck dropout: 30% dropout on bottlenecked deck embeddings during
        training, ensuring decoder learns to rely on z.
      - KL targeting: loss uses |KL - target| instead of beta*KL, explicitly
        controlling information content of z (configured in training loop).

    The encoder sees full 32-dim deck embeddings (no bottleneck) so it can
    produce informative z. Only the decoder's view of the deck is restricted.

    Args:
        vocab_size: Card vocabulary size.
        latent_dim: Variational latent dimension.
        deck_embed_dim: Deck embedding dimension (before bottleneck).
        deck_bottleneck_dim: Deck dimension seen by decoder (after bottleneck).
        deck_dropout: Dropout rate on bottlenecked deck embeddings.
        dropout: General dropout rate.
    """

    def __init__(
        self,
        vocab_size: int,
        latent_dim: int = 64,
        deck_embed_dim: int = 32,
        deck_bottleneck_dim: int = 8,
        deck_dropout: float = 0.3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim
        self.deck_bottleneck_dim = deck_bottleneck_dim

        self.player_deck_embedder = DeckEmbedder(
            vocab_size, output_dim=deck_embed_dim,
        )
        self.opponent_deck_embedder = DeckEmbedder(
            vocab_size, output_dim=deck_embed_dim,
        )

        # Bottleneck: compress deck embeddings before decoder sees them
        self.deck_bottleneck = nn.Linear(deck_embed_dim, deck_bottleneck_dim)
        self.deck_dropout = nn.Dropout(deck_dropout)

        # Encoder sees full deck embeddings (no bottleneck)
        self.encoder = CVAEEncoder(
            vocab_size=vocab_size,
            dropout=dropout,
            deck_embed_dim=deck_embed_dim,
            latent_dim=latent_dim,
        )

        # Decoder sees bottlenecked deck embeddings
        self.decoder = CVAEDecoder(
            vocab_size=vocab_size,
            dropout=dropout,
            deck_embed_dim=deck_bottleneck_dim,
            latent_dim=latent_dim,
        )

    def bottleneck_deck(self, deck_emb: torch.Tensor) -> torch.Tensor:
        """Apply deck bottleneck (+ dropout if training) for decoder input.

        Args:
            deck_emb: (batch, deck_embed_dim) full deck embedding.

        Returns:
            (batch, deck_bottleneck_dim) compressed deck embedding.
        """
        return self.deck_dropout(self.deck_bottleneck(deck_emb))

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z from N(mu, sigma) via reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(
        self,
        card_ids: torch.Tensor,
        features: torch.Tensor,
        lengths: torch.Tensor,
        player_deck_ids: torch.Tensor,
        opponent_deck_ids: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Full forward pass: encode -> reparameterize -> decode.

        Args:
            card_ids: (batch, seq_len) event card indices.
            features: (batch, seq_len, 17) event features.
            lengths: (batch,) sequence lengths.
            player_deck_ids: (batch, 8) player deck card indices.
            opponent_deck_ids: (batch, 8) opponent deck card indices.

        Returns:
            Dict with decoder outputs + mu + logvar.
        """
        player_deck_emb = self.player_deck_embedder(player_deck_ids)
        opponent_deck_emb = self.opponent_deck_embedder(opponent_deck_ids)

        # Encoder sees full deck embeddings
        mu, logvar = self.encoder(
            card_ids, features, lengths,
            player_deck_emb, opponent_deck_emb,
        )
        z = self.reparameterize(mu, logvar)

        # Decoder sees bottlenecked + dropped-out deck embeddings
        p_deck_bn = self.deck_dropout(self.deck_bottleneck(player_deck_emb))
        o_deck_bn = self.deck_dropout(self.deck_bottleneck(opponent_deck_emb))

        decoder_out = self.decoder(
            card_ids, features, lengths,
            z, p_deck_bn, o_deck_bn,
        )

        decoder_out["mu"] = mu
        decoder_out["logvar"] = logvar
        return decoder_out

    @classmethod
    def from_pretrained_wp(
        cls,
        wp_checkpoint_path: str,
        vocab_size: int,
        device: torch.device,
        freeze_encoder: bool = True,
        dropout: float = 0.2,
        **kwargs,
    ) -> "CounterfactualVAE":
        """Initialize CVAE with encoder weights from a trained WP model.

        Transfers card_embedding and TCN weights from wp_v1.pt.

        Args:
            wp_checkpoint_path: Path to wp_v1.pt.
            vocab_size: Vocabulary size.
            device: Target device.
            freeze_encoder: Freeze TCN weights initially.
            dropout: Dropout rate.
            **kwargs: Additional args passed to CounterfactualVAE (e.g.
                deck_bottleneck_dim, deck_dropout).

        Returns:
            CounterfactualVAE with transferred encoder weights.
        """
        checkpoint = torch.load(wp_checkpoint_path, map_location=device, weights_only=True)
        saved_vocab = checkpoint.get("vocab_size", vocab_size)

        model = cls(vocab_size=saved_vocab, dropout=dropout, **kwargs)

        # Transfer card_embedding and TCN weights to encoder
        source_state = checkpoint["model_state_dict"]

        # Map WP model keys -> CVAE encoder keys
        encoder_state = model.encoder.state_dict()
        transferred = 0
        for key in encoder_state:
            if key in source_state and encoder_state[key].shape == source_state[key].shape:
                encoder_state[key] = source_state[key]
                transferred += 1

        model.encoder.load_state_dict(encoder_state)

        # Also transfer card embeddings to decoder
        decoder_state = model.decoder.state_dict()
        for key in decoder_state:
            src_key = key
            if src_key in source_state and decoder_state[key].shape == source_state[src_key].shape:
                decoder_state[key] = source_state[src_key]
        model.decoder.load_state_dict(decoder_state)

        # Transfer to deck embedders
        if "card_embedding.weight" in source_state:
            src_emb_weight = source_state["card_embedding.weight"]
            for embedder in [model.player_deck_embedder, model.opponent_deck_embedder]:
                emb = embedder.card_embedding.weight.data
                n_copy = min(src_emb_weight.size(0), emb.size(0))
                dim_copy = min(src_emb_weight.size(1), emb.size(1))
                emb[:n_copy, :dim_copy] = src_emb_weight[:n_copy, :dim_copy]

        if freeze_encoder:
            for name, param in model.encoder.named_parameters():
                if name.startswith("card_embedding.") or name.startswith("tcn."):
                    param.requires_grad = False

        model.to(device)
        return model
