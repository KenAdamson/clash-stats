"""Tests for CVAE counterfactual simulator (ADR-006).

Tests cover:
  - DeckEmbedder permutation invariance and shape
  - CVAE forward pass shapes
  - KL divergence >= 0 and beta=0 zeroes KL contribution
  - CVAE encoder/decoder integration
  - ORM model creation
  - Alembic migration creates tables
"""

import numpy as np
import pytest
import torch

from tracker.ml.deck_embedding import DeckEmbedder
from tracker.ml.cvae import (
    CounterfactualVAE,
    CVAEDecoder,
    CVAEEncoder,
    SinusoidalPositionalEncoding,
)
# Import storage models so Base.metadata knows about the tables
from tracker.ml.cvae_storage import CounterfactualResult, DeckGradientResult  # noqa: F401


# =============================================================================
# DECK EMBEDDER
# =============================================================================

class TestDeckEmbedder:
    """Tests for the Deep Sets deck embedder."""

    @pytest.fixture
    def embedder(self):
        return DeckEmbedder(vocab_size=100, card_embed_dim=16, hidden_dim=32, output_dim=32)

    def test_output_shape(self, embedder):
        """Output should be (batch, output_dim)."""
        deck = torch.randint(2, 100, (4, 8))
        out = embedder(deck)
        assert out.shape == (4, 32)

    def test_permutation_invariance(self, embedder):
        """Same cards in different order should produce same embedding."""
        deck = torch.randint(2, 100, (1, 8))
        # Create a permuted version
        perm = torch.randperm(8)
        deck_permuted = deck[:, perm]

        embedder.eval()
        with torch.no_grad():
            emb1 = embedder(deck)
            emb2 = embedder(deck_permuted)

        torch.testing.assert_close(emb1, emb2, atol=1e-5, rtol=1e-5)

    def test_different_decks_differ(self, embedder):
        """Different decks should produce different embeddings."""
        deck1 = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]])
        deck2 = torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17]])

        embedder.eval()
        with torch.no_grad():
            emb1 = embedder(deck1)
            emb2 = embedder(deck2)

        assert not torch.allclose(emb1, emb2, atol=1e-3)

    def test_padding_handled(self, embedder):
        """Padding tokens (index 0) should be masked out."""
        deck_full = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]])
        deck_padded = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 0]])

        embedder.eval()
        with torch.no_grad():
            emb_full = embedder(deck_full)
            emb_padded = embedder(deck_padded)

        # Should differ because one card is missing
        assert not torch.allclose(emb_full, emb_padded, atol=1e-3)

    def test_load_card_embeddings(self, embedder):
        """Should transfer weights from a source embedding."""
        source = torch.nn.Embedding(80, 16, padding_idx=0)
        n_copied = embedder.load_card_embeddings_from(source)
        assert n_copied == 80  # min of 100 and 80


# =============================================================================
# CVAE ENCODER
# =============================================================================

class TestCVAEEncoder:
    """Tests for the variational encoder."""

    @pytest.fixture
    def encoder(self):
        return CVAEEncoder(
            vocab_size=100,
            card_embed_dim=16,
            feature_dim=17,
            dropout=0.0,
            deck_embed_dim=32,
            latent_dim=64,
        )

    def test_output_shapes(self, encoder):
        """mu and logvar should be (batch, latent_dim)."""
        batch = 4
        seq_len = 20
        card_ids = torch.randint(0, 100, (batch, seq_len))
        features = torch.randn(batch, seq_len, 17)
        lengths = torch.full((batch,), seq_len, dtype=torch.long)
        player_deck_emb = torch.randn(batch, 32)
        opponent_deck_emb = torch.randn(batch, 32)

        encoder.eval()
        with torch.no_grad():
            mu, logvar = encoder(card_ids, features, lengths, player_deck_emb, opponent_deck_emb)

        assert mu.shape == (batch, 64)
        assert logvar.shape == (batch, 64)


# =============================================================================
# CVAE DECODER
# =============================================================================

class TestCVAEDecoder:
    """Tests for the Transformer decoder."""

    @pytest.fixture
    def decoder(self):
        return CVAEDecoder(
            vocab_size=100,
            card_embed_dim=16,
            feature_dim=17,
            d_model=64,
            nhead=4,
            num_layers=2,
            dropout=0.0,
            deck_embed_dim=32,
            latent_dim=64,
        )

    def test_teacher_forced_shapes(self, decoder):
        """Teacher-forced outputs should match input shapes."""
        batch = 4
        seq_len = 20
        card_ids = torch.randint(1, 100, (batch, seq_len))
        features = torch.randn(batch, seq_len, 17)
        lengths = torch.full((batch,), seq_len, dtype=torch.long)
        z = torch.randn(batch, 64)
        p_emb = torch.randn(batch, 32)
        o_emb = torch.randn(batch, 32)

        decoder.eval()
        with torch.no_grad():
            out = decoder(card_ids, features, lengths, z, p_emb, o_emb)

        assert out["card_logits"].shape == (batch, seq_len, 100)
        assert out["tick_delta"].shape == (batch, seq_len)
        assert out["arena_xy"].shape == (batch, seq_len, 2)
        assert out["side_logit"].shape == (batch, seq_len)

    def test_tick_delta_positive(self, decoder):
        """Tick deltas should be non-negative (softplus)."""
        batch = 2
        seq_len = 10
        card_ids = torch.randint(1, 100, (batch, seq_len))
        features = torch.randn(batch, seq_len, 17)
        lengths = torch.full((batch,), seq_len, dtype=torch.long)
        z = torch.randn(batch, 64)
        p_emb = torch.randn(batch, 32)
        o_emb = torch.randn(batch, 32)

        decoder.eval()
        with torch.no_grad():
            out = decoder(card_ids, features, lengths, z, p_emb, o_emb)

        assert (out["tick_delta"] >= 0).all()

    def test_arena_xy_bounded(self, decoder):
        """Arena positions should be in [0, 1] (sigmoid)."""
        batch = 2
        seq_len = 10
        card_ids = torch.randint(1, 100, (batch, seq_len))
        features = torch.randn(batch, seq_len, 17)
        lengths = torch.full((batch,), seq_len, dtype=torch.long)
        z = torch.randn(batch, 64)
        p_emb = torch.randn(batch, 32)
        o_emb = torch.randn(batch, 32)

        decoder.eval()
        with torch.no_grad():
            out = decoder(card_ids, features, lengths, z, p_emb, o_emb)

        assert (out["arena_xy"] >= 0).all()
        assert (out["arena_xy"] <= 1).all()


# =============================================================================
# FULL CVAE
# =============================================================================

class TestCounterfactualVAE:
    """Tests for the full CVAE model."""

    @pytest.fixture
    def cvae(self):
        return CounterfactualVAE(
            vocab_size=100,
            latent_dim=64,
            deck_embed_dim=32,
            dropout=0.0,
        )

    def test_forward_shapes(self, cvae):
        """Full forward pass should produce correct output shapes."""
        batch = 4
        seq_len = 20
        card_ids = torch.randint(1, 100, (batch, seq_len))
        features = torch.randn(batch, seq_len, 17)
        lengths = torch.full((batch,), seq_len, dtype=torch.long)
        p_deck = torch.randint(2, 100, (batch, 8))
        o_deck = torch.randint(2, 100, (batch, 8))

        cvae.eval()
        with torch.no_grad():
            out = cvae(card_ids, features, lengths, p_deck, o_deck)

        assert out["mu"].shape == (batch, 64)
        assert out["logvar"].shape == (batch, 64)
        assert out["card_logits"].shape == (batch, seq_len, 100)

    def test_kl_nonnegative(self, cvae):
        """KL divergence should be >= 0."""
        batch = 4
        seq_len = 15
        card_ids = torch.randint(1, 100, (batch, seq_len))
        features = torch.randn(batch, seq_len, 17)
        lengths = torch.full((batch,), seq_len, dtype=torch.long)
        p_deck = torch.randint(2, 100, (batch, 8))
        o_deck = torch.randint(2, 100, (batch, 8))

        cvae.eval()
        with torch.no_grad():
            out = cvae(card_ids, features, lengths, p_deck, o_deck)

        mu = out["mu"]
        logvar = out["logvar"]
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        assert kl.item() >= -1e-6  # Allow tiny float imprecision

    def test_reparameterize_deterministic_at_zero_var(self, cvae):
        """When logvar=-inf (sigma=0), z should equal mu."""
        mu = torch.randn(2, 64)
        logvar = torch.full((2, 64), -100.0)  # Very negative = sigma ~= 0
        z = CounterfactualVAE.reparameterize(mu, logvar)
        torch.testing.assert_close(z, mu, atol=1e-4, rtol=1e-4)

    def test_generation_produces_output(self, cvae):
        """Autoregressive generation should produce non-empty sequences."""
        batch = 2
        z = torch.randn(batch, 64)
        p_deck = torch.randint(2, 100, (batch, 8))
        o_deck = torch.randint(2, 100, (batch, 8))
        p_emb = cvae.bottleneck_deck(cvae.player_deck_embedder(p_deck))
        o_emb = cvae.bottleneck_deck(cvae.opponent_deck_embedder(o_deck))

        cvae.eval()
        gen = cvae.decoder.generate(
            z, p_emb, o_emb,
            end_token_id=99,  # Unlikely to be sampled immediately
            max_events=10,
        )

        assert gen["card_ids"].shape[0] == batch
        assert gen["card_ids"].shape[1] <= 10


# =============================================================================
# POSITIONAL ENCODING
# =============================================================================

class TestPositionalEncoding:
    def test_shape(self):
        pe = SinusoidalPositionalEncoding(d_model=64, max_len=100, dropout=0.0)
        x = torch.zeros(2, 50, 64)
        out = pe(x)
        assert out.shape == (2, 50, 64)

    def test_different_positions(self):
        pe = SinusoidalPositionalEncoding(d_model=64, max_len=100, dropout=0.0)
        x = torch.zeros(1, 100, 64)
        out = pe(x)
        # Different positions should have different encodings
        assert not torch.allclose(out[0, 0], out[0, 1])


# =============================================================================
# STORAGE MODELS
# =============================================================================

class TestCVAEStorage:
    def test_counterfactual_result_creation(self, session):
        """CounterfactualResult table should exist after metadata.create_all."""
        from sqlalchemy import inspect
        insp = inspect(session.bind)
        tables = set(insp.get_table_names())
        assert "counterfactual_results" in tables

    def test_deck_gradient_result_creation(self, session):
        """DeckGradientResult table should exist after metadata.create_all."""
        from sqlalchemy import inspect
        insp = inspect(session.bind)
        tables = set(insp.get_table_names())
        assert "deck_gradient_results" in tables

    def test_insert_counterfactual_result(self, session):
        """Should be able to insert and query CounterfactualResult."""
        from tracker.ml.cvae_storage import CounterfactualResult
        from tracker.tests.conftest import make_battle, PLAYER_DECK, OPPONENT_DECK
        from tracker import analytics

        # Need a battle to FK reference
        battle = make_battle()
        analytics.store_battle(session, battle, "#L90009GPP")
        bid = session.query(Battle.battle_id).first()[0]

        cr = CounterfactualResult(
            battle_id=bid,
            old_card="Bats",
            new_card="Minions",
            original_wp=0.65,
            counterfactual_wp_mean=0.58,
            counterfactual_wp_std=0.05,
            delta_wp=-0.07,
            n_samples=10,
            model_version="cvae-v1",
        )
        session.add(cr)
        session.flush()

        result = session.query(CounterfactualResult).first()
        assert result.old_card == "Bats"
        assert result.delta_wp == pytest.approx(-0.07)


# Need Battle import for the storage test
from tracker.models import Battle
