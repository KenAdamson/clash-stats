"""Counterfactual generation and deck gradient analysis (ADR-006).

CounterfactualGenerator:
  1. Load real game -> encode -> (mu, logvar)
  2. Swap card in player deck -> new deck embedding
  3. Sample z from N(mu, sigma) x n_samples, decode autoregressively
  4. Post-process: clamp positions, enforce positive tick deltas
  5. Run WP model on each generated sequence -> P(win) distribution
  6. Report: original P(win), mean counterfactual P(win), delta, std

DeckGradient:
  For each card in player's deck x each valid substitute:
    Run counterfactual on N recent personal games
    Average delta P(win)
  Return ranked: (old_card, new_card, delta_wr, confidence_interval)
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sqlalchemy import select, text as sa_text
from sqlalchemy.orm import Session

from tracker.ml.card_metadata import CardVocabulary, kebab_to_title
from tracker.ml.cvae import CounterfactualVAE
from tracker.ml.cvae_dataset import CVAEDataset, DECK_SIZE
from tracker.ml.cvae_storage import CounterfactualResult, DeckGradientResult
from tracker.ml.cvae_training import CVAE_MODEL_VERSION
from tracker.ml.sequence_dataset import MIN_EVENTS
from tracker.ml.win_probability import WinProbabilityModel
from tracker.ml.wp_training import WP_MODEL_VERSION
from tracker.models import Battle, DeckCard

logger = logging.getLogger(__name__)


END_TOKEN = "<END>"


def _detect_device() -> torch.device:
    """Detect best available device for CVAE inference."""
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_cvae(
    model_dir: Path, device: torch.device,
) -> Optional[CounterfactualVAE]:
    """Load trained CVAE from checkpoint."""
    # Prefer v3 -> v2 -> v1
    for name in ["cvae_v3.pt", "cvae_v2.pt", "cvae_v1.pt"]:
        cvae_path = model_dir / name
        if cvae_path.exists():
            break
    else:
        return None

    checkpoint = torch.load(cvae_path, map_location=device, weights_only=True)
    model = CounterfactualVAE(
        vocab_size=checkpoint["vocab_size"],
        latent_dim=checkpoint.get("latent_dim", 64),
        deck_bottleneck_dim=checkpoint.get("deck_bottleneck_dim", 32),
        deck_dropout=0.0,  # no dropout at inference
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    logger.info(
        "Loaded CVAE from epoch %d (val_loss=%.4f)",
        checkpoint["epoch"], checkpoint["val_loss"],
    )
    return model


def _load_wp_model(
    model_dir: Path, device: torch.device,
) -> Optional[WinProbabilityModel]:
    """Load trained WP model from checkpoint."""
    wp_path = model_dir / "wp_v1.pt"
    if not wp_path.exists():
        return None

    checkpoint = torch.load(wp_path, map_location=device, weights_only=True)
    model = WinProbabilityModel(
        vocab_size=checkpoint["vocab_size"], dropout=0.0,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def _get_deck_ids(
    session: Session, battle_id: str, vocab: CardVocabulary,
) -> tuple[np.ndarray, np.ndarray]:
    """Get player and opponent deck card IDs for a battle.

    Returns:
        (player_deck_ids, opponent_deck_ids) each shape (8,) int64.
    """
    rows = session.execute(
        select(DeckCard.card_name, DeckCard.is_player_deck)
        .where(DeckCard.battle_id == battle_id)
    ).all()

    player_ids = np.zeros(DECK_SIZE, dtype=np.int64)
    opponent_ids = np.zeros(DECK_SIZE, dtype=np.int64)
    pi, oi = 0, 0

    for card_name, is_player in rows:
        idx = vocab.encode(card_name)
        if is_player and pi < DECK_SIZE:
            player_ids[pi] = idx
            pi += 1
        elif not is_player and oi < DECK_SIZE:
            opponent_ids[oi] = idx
            oi += 1

    return player_ids, opponent_ids


def _evaluate_sequence_wp(
    wp_model: WinProbabilityModel,
    card_ids: torch.Tensor,
    features: torch.Tensor,
    lengths: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """Run WP model on sequences to get final P(win).

    Args:
        card_ids: (batch, seq_len) int64.
        features: (batch, seq_len, 17) float32.
        lengths: (batch,) int64.

    Returns:
        (batch,) final P(win) values.
    """
    with torch.no_grad():
        logits = wp_model(card_ids, features, lengths)  # (batch, seq_len)

    # Get last valid tick logit
    batch_size = logits.size(0)
    last_indices = (lengths - 1).clamp(min=0).long()
    last_logits = logits[torch.arange(batch_size, device=device), last_indices]
    probs = torch.sigmoid(last_logits).cpu().numpy()
    return probs


def _build_wp_features_from_generated(
    generated: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Convert generated sequence into WP model features.

    Builds the 17-dim feature vector from generated card_ids, tick_deltas,
    arena_xys, and sides.

    Returns:
        features: (batch, seq_len, 17) float32.
    """
    batch_size, seq_len = generated["card_ids"].shape
    device = generated["card_ids"].device
    features = torch.zeros(batch_size, seq_len, 17, device=device)

    # side (index 0)
    features[:, :, 0] = generated["sides"]

    # game_tick_norm (index 1) — cumulative sum of tick_deltas
    cum_ticks = torch.cumsum(generated["tick_deltas"], dim=1)
    features[:, :, 1] = cum_ticks.clamp(max=1.0)

    # arena_x_norm (index 6): generated is [0,1], convert to [-1,1]
    features[:, :, 6] = generated["arena_xys"][:, :, 0] * 2 - 1

    # arena_y_norm (index 7): generated is [0,1], convert to [-1,1]
    features[:, :, 7] = generated["arena_xys"][:, :, 1] * 2 - 1

    # play_number (index 11) — simple increment
    play_nums = torch.arange(1, seq_len + 1, device=device).float() / 20.0
    features[:, :, 11] = play_nums.unsqueeze(0).expand(batch_size, -1).clamp(max=1.0)

    return features


def replay_swap(
    session: Session,
    battle_id: str,
    old_card: str,
    new_card: str,
    model_dir: Optional[Path] = None,
    ticks: list[int] | None = None,
) -> Optional[dict]:
    """Counterfactual by swapping card plays in a real replay.

    Takes a real game's event sequence, replaces plays of old_card with
    new_card (which must already be in the player's deck), and reruns
    the WP model to compare P(win) curves.

    No CVAE needed — pure replay modification + WP evaluation.

    Args:
        session: Database session.
        battle_id: Source battle.
        old_card: Card to replace (Title Case, e.g. "Bats").
        new_card: Card to substitute (must be in player's deck).
        model_dir: Directory containing wp_v1.pt.
        ticks: Optional list of specific game ticks to swap at.
               If None, swaps all occurrences.

    Returns:
        Dict with original/modified P(win) curves and delta, or None on error.
    """
    if model_dir is None:
        model_dir = Path("data/ml_models")

    device = _detect_device()
    wp_model = _load_wp_model(model_dir, device)
    if wp_model is None:
        logger.error("No WP model found")
        return None

    vocab = CardVocabulary(session)

    # Load the real game sequence via SequenceDataset
    from tracker.ml.sequence_dataset import SequenceDataset
    dataset = SequenceDataset(session, vocab, battle_ids=[battle_id])
    if len(dataset) == 0:
        logger.warning("No replay data for battle %s", battle_id)
        return None

    card_ids_t, features_t, label = dataset[0]

    # Load replay events to map indices to card names and ticks
    from tracker.models import ReplayEvent
    events = session.execute(
        select(ReplayEvent)
        .where(ReplayEvent.battle_id == battle_id, ReplayEvent.card_name != "_invalid")
        .order_by(ReplayEvent.game_tick)
    ).scalars().all()

    old_card_kebab = old_card.lower().replace(" ", "-").replace(".", "")
    new_idx = vocab.encode(new_card)
    if new_idx == 1:  # UNK
        logger.warning("Card '%s' not in vocabulary", new_card)
        return None

    # Verify new_card is in player's deck
    player_deck, _ = _get_deck_ids(session, battle_id, vocab)
    deck_names = [vocab.decode(int(idx)) for idx in player_deck if idx != 0]
    if new_card not in deck_names:
        logger.warning("'%s' is not in the player's deck: %s", new_card, deck_names)
        return None

    # Find which event indices to swap
    swap_indices = []
    for i, ev in enumerate(events):
        if i >= len(card_ids_t):
            break
        if ev.side == "team" and ev.card_name == old_card_kebab:
            if ticks is None or ev.game_tick in ticks:
                swap_indices.append(i)

    if not swap_indices:
        logger.warning("No plays of '%s' (team side) found in battle %s", old_card, battle_id)
        return None

    # Build modified sequence
    card_ids_orig = card_ids_t.clone()
    card_ids_mod = card_ids_t.clone()

    # Also update the elixir cost feature (index 13) for swapped events
    features_mod = features_t.clone()
    new_elixir = vocab.elixir(new_card)

    for i in swap_indices:
        card_ids_mod[i] = new_idx
        if new_elixir is not None:
            features_mod[i, 13] = new_elixir / 10.0

    # Run WP model on both sequences
    seq_len = len(card_ids_t)
    lengths = torch.tensor([seq_len], dtype=torch.long, device=device)

    orig_cids = card_ids_orig.unsqueeze(0).to(device)
    orig_feats = features_t.unsqueeze(0).to(device)
    mod_cids = card_ids_mod.unsqueeze(0).to(device)
    mod_feats = features_mod.unsqueeze(0).to(device)

    with torch.no_grad():
        orig_logits = wp_model(orig_cids, orig_feats, lengths)
        mod_logits = wp_model(mod_cids, mod_feats, lengths)

    orig_probs = torch.sigmoid(orig_logits).squeeze(0).cpu().numpy()
    mod_probs = torch.sigmoid(mod_logits).squeeze(0).cpu().numpy()

    # Build tick list from events
    event_ticks = [ev.game_tick for ev in events[:seq_len]]

    # Compute deltas at swap points
    swap_deltas = []
    for i in swap_indices:
        swap_deltas.append({
            "tick": events[i].game_tick,
            "original_wp": float(orig_probs[i]),
            "modified_wp": float(mod_probs[i]),
            "delta": float(mod_probs[i] - orig_probs[i]),
        })

    result = {
        "battle_id": battle_id,
        "old_card": old_card,
        "new_card": new_card,
        "swaps": len(swap_indices),
        "original_final_wp": float(orig_probs[-1]),
        "modified_final_wp": float(mod_probs[-1]),
        "delta_final_wp": float(mod_probs[-1] - orig_probs[-1]),
        "swap_details": swap_deltas,
        "original_curve": orig_probs.tolist(),
        "modified_curve": mod_probs.tolist(),
        "ticks": event_ticks,
    }

    return result



class CounterfactualGenerator:
    """Generates counterfactual game sequences with modified decks.

    Args:
        session: Database session.
        model_dir: Directory containing model checkpoints.
    """

    def __init__(
        self,
        session: Session,
        model_dir: Optional[Path] = None,
    ):
        if model_dir is None:
            model_dir = Path("data/ml_models")

        self.session = session
        self.model_dir = model_dir
        self.device = _detect_device()

        self.vocab = CardVocabulary(session)

        # Add END token to vocabulary if not present
        if END_TOKEN not in self.vocab._card_to_idx:
            end_idx = self.vocab.size
            self.vocab._card_to_idx[END_TOKEN] = end_idx
            self.vocab._idx_to_card[end_idx] = END_TOKEN
        self.end_token_id = self.vocab.encode(END_TOKEN)

        self.cvae = _load_cvae(model_dir, self.device)
        self.wp_model = _load_wp_model(model_dir, self.device)

    def run_counterfactual(
        self,
        battle_id: str,
        old_card: str,
        new_card: str,
        n_samples: int = 10,
    ) -> Optional[dict]:
        """Generate counterfactual for swapping one card.

        Args:
            battle_id: Source battle to base counterfactual on.
            old_card: Card to remove from player's deck.
            new_card: Card to add in its place.
            n_samples: Number of counterfactual samples.

        Returns:
            Dict with original_wp, cf_wp_mean, cf_wp_std, delta, or None on error.
        """
        if self.cvae is None:
            logger.error("No CVAE model loaded")
            return None
        if self.wp_model is None:
            logger.error("No WP model loaded")
            return None

        # Load original game data
        dataset = CVAEDataset(self.session, self.vocab, battle_ids=[battle_id])
        if len(dataset) == 0:
            logger.warning("No data found for battle %s", battle_id)
            return None

        card_ids, features, label, player_deck, opponent_deck = dataset[0]
        card_ids = card_ids.unsqueeze(0).to(self.device)
        features = features.unsqueeze(0).to(self.device)
        lengths = torch.tensor([card_ids.size(1)], dtype=torch.long, device=self.device)
        player_deck = player_deck.unsqueeze(0).to(self.device)
        opponent_deck = opponent_deck.unsqueeze(0).to(self.device)

        # Get original P(win)
        original_wp = _evaluate_sequence_wp(
            self.wp_model, card_ids, features, lengths, self.device,
        )[0]

        # Encode original game
        player_deck_emb = self.cvae.player_deck_embedder(player_deck)
        opponent_deck_emb = self.cvae.opponent_deck_embedder(opponent_deck)

        with torch.no_grad():
            mu, logvar = self.cvae.encoder(
                card_ids, features, lengths,
                player_deck_emb, opponent_deck_emb,
            )

        # Swap card in player deck
        old_idx = self.vocab.encode(old_card)
        new_idx = self.vocab.encode(new_card)

        modified_deck = player_deck.clone()
        # Find and replace the old card
        replaced = False
        for i in range(DECK_SIZE):
            if modified_deck[0, i].item() == old_idx:
                modified_deck[0, i] = new_idx
                replaced = True
                break

        if not replaced:
            logger.warning("Card '%s' not found in player deck for battle %s", old_card, battle_id)
            return None

        # New deck embedding (bottlenecked for decoder)
        modified_deck_emb = self.cvae.bottleneck_deck(
            self.cvae.player_deck_embedder(modified_deck)
        )
        opponent_deck_emb_bn = self.cvae.bottleneck_deck(opponent_deck_emb)

        # Sample and decode
        cf_wps = []
        for _ in range(n_samples):
            z = CounterfactualVAE.reparameterize(mu, logvar)

            generated = self.cvae.decoder.generate(
                z, modified_deck_emb, opponent_deck_emb_bn,
                end_token_id=self.end_token_id,
                player_deck_ids=modified_deck,
            )

            gen_len = generated["lengths"]
            if gen_len[0] == 0:
                continue

            # Build WP features from generated sequence
            gen_features = _build_wp_features_from_generated(generated)
            gen_card_ids = generated["card_ids"]

            # Evaluate P(win)
            wp = _evaluate_sequence_wp(
                self.wp_model, gen_card_ids, gen_features, gen_len, self.device,
            )[0]
            cf_wps.append(float(wp))

        if not cf_wps:
            logger.warning("All counterfactual samples produced empty sequences")
            return None

        cf_mean = float(np.mean(cf_wps))
        cf_std = float(np.std(cf_wps))
        delta = cf_mean - float(original_wp)

        result = {
            "battle_id": battle_id,
            "old_card": old_card,
            "new_card": new_card,
            "original_wp": float(original_wp),
            "counterfactual_wp_mean": cf_mean,
            "counterfactual_wp_std": cf_std,
            "delta_wp": delta,
            "n_samples": len(cf_wps),
            "sample_wps": cf_wps,
        }

        # Store result
        self.session.add(CounterfactualResult(
            battle_id=battle_id,
            old_card=old_card,
            new_card=new_card,
            original_wp=float(original_wp),
            counterfactual_wp_mean=cf_mean,
            counterfactual_wp_std=cf_std,
            delta_wp=delta,
            n_samples=len(cf_wps),
            model_version=CVAE_MODEL_VERSION,
            raw_json=result,
        ))
        self.session.commit()

        return result

    def compute_deck_gradient(
        self,
        n_games: int = 20,
        n_samples: int = 5,
    ) -> list[dict]:
        """Rank single-card swaps by expected WR delta.

        Tests each card in the player's deck against all other known cards,
        averaged over recent personal games.

        Args:
            n_games: Number of recent personal games to average over.
            n_samples: Counterfactual samples per game per swap.

        Returns:
            List of dicts sorted by mean_delta_wp descending.
        """
        if self.cvae is None or self.wp_model is None:
            logger.error("CVAE or WP model not loaded")
            return []

        # Get recent personal battles with replay data
        battle_ids = self.session.execute(
            sa_text("""
                SELECT b.battle_id
                FROM battles b
                JOIN (
                    SELECT battle_id FROM replay_events
                    WHERE card_name != '_invalid'
                    GROUP BY battle_id HAVING COUNT(*) >= :min_events
                ) re ON re.battle_id = b.battle_id
                WHERE b.corpus = 'personal'
                  AND b.battle_type = 'PvP'
                  AND b.result IN ('win', 'loss')
                ORDER BY b.battle_time DESC
                LIMIT :limit
            """),
            {"min_events": MIN_EVENTS, "limit": n_games},
        ).scalars().all()
        battle_ids = list(battle_ids)

        if not battle_ids:
            logger.warning("No personal games with replay data")
            return []

        # Get player's current deck from most recent battle
        player_deck_ids, _ = _get_deck_ids(
            self.session, battle_ids[0], self.vocab,
        )
        player_cards = [
            self.vocab.decode(int(idx)) for idx in player_deck_ids if idx != 0
        ]

        # Get all known cards as potential substitutes
        all_cards = self.vocab.card_names()
        substitutes = [c for c in all_cards if c not in player_cards and c != END_TOKEN]

        logger.info(
            "Computing deck gradient: %d deck cards x %d substitutes x %d games",
            len(player_cards), len(substitutes), len(battle_ids),
        )

        results = []
        for old_card in player_cards:
            for new_card in substitutes:
                deltas = []
                for bid in battle_ids:
                    result = self.run_counterfactual(
                        bid, old_card, new_card, n_samples=n_samples,
                    )
                    if result is not None:
                        deltas.append(result["delta_wp"])

                if not deltas:
                    continue

                mean_delta = float(np.mean(deltas))
                std_delta = float(np.std(deltas))
                n = len(deltas)
                # 95% CI
                ci_half = 1.96 * std_delta / max(np.sqrt(n), 1)

                entry = {
                    "old_card": old_card,
                    "new_card": new_card,
                    "mean_delta_wp": mean_delta,
                    "ci_low": mean_delta - ci_half,
                    "ci_high": mean_delta + ci_half,
                    "n_games": n,
                }
                results.append(entry)

                # Store
                self.session.add(DeckGradientResult(
                    old_card=old_card,
                    new_card=new_card,
                    mean_delta_wp=mean_delta,
                    ci_low=mean_delta - ci_half,
                    ci_high=mean_delta + ci_half,
                    n_games=n,
                    model_version=CVAE_MODEL_VERSION,
                ))

        self.session.commit()

        # Sort by mean delta descending
        results.sort(key=lambda r: r["mean_delta_wp"], reverse=True)
        return results
