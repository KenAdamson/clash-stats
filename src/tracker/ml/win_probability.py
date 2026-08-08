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
    → Concat deck prior (v10): mean-pooled card+variant embeddings for BOTH
      decks, broadcast to every tick so it is present at tick 0
    → Per-tick head: Linear(256[+extra][+deck]→64) → ReLU → Dropout → Linear(64→1)
    → Output: P(win) at each tick
"""

import torch
import torch.nn as nn

from tracker.ml.tcn import TCNEncoder

# base / evo / hero (deck_cards.card_variant)
N_CARD_VARIANTS = 3


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
        extra_feature_dim: int = 0,
        tcn_channels: list[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
        deck_features: bool = False,
        deck_interaction: bool = False,
        deck_antisym: bool = False,
    ):
        super().__init__()
        self.card_embedding = nn.Embedding(vocab_size, card_embed_dim, padding_idx=0)
        # base / evo / hero. The card vocabulary is keyed on card_name alone, so
        # Evo Witch and Witch share an index; this offset is what separates them.
        # Variant is per (battle, side, card) — the same Wizard is evo in one of
        # Ken's games and hero in the other — so it cannot live in the vocabulary.
        #
        # Created ONLY with the deck prior. Building it unconditionally adds a
        # tensor that pre-v10 checkpoints do not carry, so a strict load of v9
        # fails with a missing key — which would have broken production
        # inference the moment the paused wp_infer_new cron came back.
        self.variant_embedding = (
            nn.Embedding(N_CARD_VARIANTS, card_embed_dim) if deck_features else None
        )
        self.feature_dim = feature_dim
        # Record the architecture shape so a checkpoint can be reconstructed at
        # inference without hardcoding sizes (capacity experiments vary these).
        self.card_embed_dim = card_embed_dim
        self.tcn_channels = list(tcn_channels) if tcn_channels else [64, 64, 128, 128, 256, 256]
        # Board-truth features injected at the HEAD rather than the encoder input.
        # The encoder input width (and thus the pretrained TCN weights) stays at
        # 16+feature_dim, so a frozen pretrained encoder transfers unchanged; the
        # extra features are concatenated onto the TCN output before the head.
        # extra_feature_dim=0 reproduces the original (encoder-input) behaviour.
        self.extra_feature_dim = extra_feature_dim

        # Deck composition, injected at the head alongside extra_feature_dim.
        #
        # This is the v10 change. Until now the model had NO deck input at all:
        # it learned a deck only by watching cards get played, so at the first
        # event it had seen one card and the opponent had revealed at most one.
        # A structural matchup — a PEKKA against a deck holding no tank killer
        # and no cheap kiting unit — was unrepresentable, and pre_game_wp sat at
        # a near-uninformative median 0.467 across 518K games.
        #
        # Both decks are known from the API before a game is ever scored, so the
        # prior is free. Deck vectors reach the head directly, which means they
        # are available at EVERY tick including the first, rather than diffusing
        # in through causal convolutions.
        self.deck_features = deck_features
        # [own ; opp ; own*opp ; own-opp].
        #
        # own and opp alone enter the head additively — W_own·own + W_opp·opp —
        # which can express "this deck is strong" but NOT "this deck beats that
        # one". A hard counter is an interaction, and measurement bore that out:
        # against empirical archetype-matchup win rates spanning 0.076, v9 (no
        # deck input) spread 0.0100 and the additive v10 spread 0.0202 — twice
        # as much differentiation, but still only ~27% of the truth. The model
        # had the direction of Goblin Barrel Bait over Goblin Giant roughly
        # right and the magnitude off by nearly 4x, which is what shrinkage
        # toward the mean looks like when marginals are asked to approximate an
        # interaction.
        #
        # own*opp is a rank-1 bilinear term — the cheapest thing that lets a
        # specific pairing be special rather than the sum of two reputations.
        # own-opp carries relative strength, which the sum cannot isolate.
        self.deck_interaction = deck_interaction and deck_features
        # v11.1: constrain the deck term to be ANTISYMMETRIC.
        #
        # A matchup must satisfy P(A beats B) = 1 - P(B beats A), so the deck
        # contribution to the logit has to flip sign when the decks swap. The
        # v11 own*opp term does the opposite: elementwise product is
        # commutative, so it is invariant under the swap and can only express
        # "this pairing is unusual", never "who wins it". Measured on v11, half
        # the deck signal was symmetric (sym 0.161 vs anti 0.173) and the model
        # spent that capacity shifting its baseline down a near-uniform 0.0405
        # -- global pessimism, exactly what a symmetric term can buy.
        #
        # Antisym mode keeps only terms that negate under the swap:
        #   own - opp                    linear, antisymmetric by construction
        #   own^T (A - A^T) opp          skew-symmetric bilinear, the canonical
        #                                "who beats whom" operator and strictly
        #                                more expressive than the difference
        # It also makes mirror matches exactly 0.5, killing the +0.029 own-side
        # bias measured on v10/v11.
        self.deck_antisym = deck_antisym and deck_features
        if self.deck_antisym:
            n_blocks = 2                       # [own-opp ; bilinear]
        elif self.deck_interaction:
            n_blocks = 4
        else:
            n_blocks = 2
        self.deck_dim = n_blocks * card_embed_dim if deck_features else 0
        if self.deck_antisym:
            # K skew-symmetric DxD forms; M_k = A_k - A_k^T is skew for any A_k.
            self.deck_bilinear = nn.Parameter(
                torch.randn(card_embed_dim, card_embed_dim, card_embed_dim) * 0.02
            )

        input_channels = card_embed_dim + feature_dim  # 16 + 17 = 33
        # The deck prior is ADDED to the head's first-layer output rather than
        # concatenated onto the TCN output. Algebraically identical — a linear
        # layer over [tcn ; deck] is W_tcn·tcn + W_deck·deck — but concatenating
        # a time-broadcast deck materialises a (batch, 512+deck, L) tensor: 453MB
        # at batch 1024, against this box's 256MB small-BAR host-visible window.
        # That is what killed the first v10 run at epoch 1 with
        # UR_RESULT_ERROR_OUT_OF_HOST_MEMORY. As a projection the deck costs
        # (batch, 64, 1), broadcast over time for free, so v10's head is no
        # heavier than v9's — and every v9 head tensor keeps its name and shape,
        # so the warm start still transfers it.

        self.tcn = TCNEncoder(
            input_channels=input_channels,
            channel_sizes=tcn_channels,
            kernel_size=kernel_size,
            dropout=dropout,
        )

        # Per-tick classification head: (batch, 256[+extra], seq_len) → (batch, 1, seq_len)
        out_ch = self.tcn.output_channels  # 256
        # bias=False in antisym mode is load-bearing: with a bias,
        # f(-x) = -Wx + b, which is NOT -f(x), and the antisymmetry the whole
        # design rests on would be broken by a constant.
        self.deck_proj = (nn.Conv1d(self.deck_dim, 64, 1, bias=not self.deck_antisym)
                          if deck_features else None)
        self.head = nn.Sequential(
            nn.Conv1d(out_ch + extra_feature_dim, 64, 1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(64, 1, 1),
        )

    def encode_decks(
        self, deck_ids: torch.Tensor, deck_variants: torch.Tensor
    ) -> torch.Tensor:
        """(batch, 2, 8) card ids + variants -> (batch, 2*card_embed_dim).

        Reuses the played-card embedding table, so a deck vector lives in the
        same space the model already learned from 1.67M games of placements —
        the deck prior costs one small variant table, not a second encoder.
        Mean-pooling over the 8 cards keeps it order-invariant, which a deck is.
        """
        emb = self.card_embedding(deck_ids) + self.variant_embedding(deck_variants)
        pooled = emb.mean(dim=2)                       # (batch, 2, D)
        own, opp = pooled[:, 0], pooled[:, 1]
        if self.deck_antisym:
            skew = self.deck_bilinear - self.deck_bilinear.transpose(-1, -2)
            bilin = torch.einsum("bi,kij,bj->bk", own, skew, opp)   # (batch, D)
            return torch.cat([own - opp, bilin], dim=1)             # (batch, 2D)
        if not self.deck_interaction:
            return pooled.flatten(1)                   # (batch, 2D)
        return torch.cat([own, opp, own * opp, own - opp], dim=1)  # (batch, 4D)

    def forward(
        self,
        card_ids: torch.Tensor,
        features: torch.Tensor,
        lengths: torch.Tensor,
        deck_ids: torch.Tensor | None = None,
        deck_variants: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass producing per-tick logits.

        Args:
            card_ids: (batch, seq_len) int64 — card vocabulary indices.
            features: (batch, seq_len, feature_dim) float32.
            lengths: (batch,) int64 — original sequence lengths.
            deck_ids: (batch, 2, 8) int64 — own/opponent deck card indices.
            deck_variants: (batch, 2, 8) int64 — 0 base, 1 evo, 2 hero.

        Returns:
            logits: (batch, seq_len) — raw logits per tick (apply sigmoid for P(win)).
        """
        card_emb = self.card_embedding(card_ids)
        if self.extra_feature_dim > 0:
            # Encoder sees only the base features; board-truth features are held
            # out and injected at the head (see __init__).
            base = features[:, :, :self.feature_dim]
            extra = features[:, :, self.feature_dim:self.feature_dim + self.extra_feature_dim]
        else:
            base = features
            extra = None
        combined = torch.cat([card_emb, base], dim=2)
        combined = combined.transpose(1, 2)  # (batch, channels, seq_len)

        tcn_out = self.tcn(combined)  # (batch, 256, seq_len)
        if extra is not None:
            tcn_out = torch.cat([tcn_out, extra.transpose(1, 2)], dim=1)  # (batch, 256+extra, seq_len)
        h = self.head[0](tcn_out)  # (batch, 64, seq_len)
        if self.deck_features:
            if deck_ids is None or deck_variants is None:
                raise ValueError(
                    "model was built with deck_features=True but forward() got no "
                    "deck tensors — the dataset and checkpoint are out of sync"
                )
            # (batch, 2D, 1) -> (batch, 64, 1), broadcast-added across every tick.
            # Constant in time by construction, so the prior is present at tick 0
            # — the whole point — without a per-tick copy of it existing anywhere.
            deck = self.encode_decks(deck_ids, deck_variants).unsqueeze(2)
            h = h + self.deck_proj(deck)
        for layer in self.head[1:]:
            h = layer(h)
        logits = h.squeeze(1)  # (batch, seq_len)

        return logits

    @classmethod
    def from_pretrained_tcn(
        cls,
        tcn_checkpoint_path: str,
        vocab_size: int,
        device: torch.device,
        freeze_encoder: bool = True,
        dropout: float = 0.2,
        extra_feature_dim: int = 0,
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

        model = cls(vocab_size=saved_vocab, dropout=dropout,
                    extra_feature_dim=extra_feature_dim)

        # Load matching weights from the source state dict. card_embedding + tcn
        # match by shape and transfer; the head does NOT match when
        # extra_feature_dim>0 (its input width grew by the injected features), so
        # it stays freshly initialized and is the only thing trained.
        source_state = checkpoint["model_state_dict"]
        target_state = model.state_dict()

        transferred = 0
        for key in target_state:
            # Match card_embedding and tcn weights
            if key in source_state and target_state[key].shape == source_state[key].shape:
                v = source_state[key]
                # Older WP checkpoints (v1/v2/v3) carry a handful of NaN weights
                # from pre-grad-clip training. The inference path sanitizes them;
                # transferring them raw would poison a frozen encoder (NaN output).
                # Only a few entries per large tensor, so nan->0 keeps norms finite.
                if torch.is_tensor(v) and v.dtype.is_floating_point and not torch.isfinite(v).all():
                    v = torch.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
                target_state[key] = v
                transferred += 1

        model.load_state_dict(target_state)

        if freeze_encoder:
            for name, param in model.named_parameters():
                if name.startswith("card_embedding.") or name.startswith("tcn."):
                    param.requires_grad = False

        model.to(device)
        return model
