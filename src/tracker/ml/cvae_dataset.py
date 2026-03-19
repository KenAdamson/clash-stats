"""Extended dataset for CVAE training (ADR-006).

Extends SequenceDataset to also load player and opponent deck card IDs
for each battle, enabling deck-conditioned generation.
"""

import logging
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from tracker.models import Battle, DeckCard, ReplayEvent
from tracker.ml.card_metadata import CardVocabulary, kebab_to_title
from tracker.ml.sequence_dataset import (
    SequenceDataset,
    MIN_EVENTS,
    GAME_TICK_MAX,
    PLAY_NUMBER_CAP,
    _game_phase_onehot,
    _lane_onehot,
    _card_type_onehot,
    ARENA_X_MID,
    ARENA_Y_MID,
)

logger = logging.getLogger(__name__)

# Number of cards in a deck
DECK_SIZE = 8

# Truncate sequences longer than this in the collate function.
# Attention is O(n^2) — a single 450-event game pads the whole batch.
# P99 is 140, so 200 covers 99%+ of games with minimal data loss.
MAX_SEQ_LEN = 200


class CVAEDataset(Dataset):
    """Dataset that yields event sequences + deck card IDs for CVAE training.

    Each sample returns:
        (card_ids, features, label, player_deck_ids, opponent_deck_ids)

    Args:
        session: SQLAlchemy session.
        vocab: CardVocabulary for card->index mapping.
        battle_ids: Optional list of specific battle IDs to include.
    """

    def __init__(
        self,
        session: Session,
        vocab: CardVocabulary,
        battle_ids: list[str] | None = None,
    ):
        self.vocab = vocab

        # Reuse SequenceDataset for event sequences
        self._base = SequenceDataset(session, vocab, battle_ids=battle_ids)

        # Load deck card IDs for each battle in the same order
        # Re-derive battle_ids from the query (same as SequenceDataset)
        if battle_ids is not None:
            battle_rows = session.execute(
                text("""
                    SELECT b.battle_id
                    FROM battles b
                    JOIN (
                        SELECT battle_id, COUNT(*) as event_count
                        FROM replay_events
                        WHERE card_name != '_invalid'
                        GROUP BY battle_id
                        HAVING COUNT(*) >= :min_events
                    ) re_counts ON re_counts.battle_id = b.battle_id
                    WHERE b.battle_type = 'PvP'
                      AND b.result IN ('win', 'loss')
                      AND b.battle_id IN :bids
                    ORDER BY b.battle_time
                """),
                {"min_events": MIN_EVENTS, "bids": tuple(battle_ids)},
            ).scalars().all()
        else:
            battle_rows = session.execute(
                text("""
                    SELECT b.battle_id
                    FROM battles b
                    JOIN (
                        SELECT battle_id, COUNT(*) as event_count
                        FROM replay_events
                        WHERE card_name != '_invalid'
                        GROUP BY battle_id
                        HAVING COUNT(*) >= :min_events
                    ) re_counts ON re_counts.battle_id = b.battle_id
                    WHERE b.battle_type = 'PvP'
                      AND b.result IN ('win', 'loss')
                    ORDER BY b.battle_time
                """),
                {"min_events": MIN_EVENTS},
            ).scalars().all()

        all_battle_ids = list(battle_rows)

        # Load deck_cards for all relevant battles
        self._deck_data: list[tuple[np.ndarray, np.ndarray]] = []

        # Batch load deck cards
        chunk_size = 500
        deck_by_battle: dict[str, dict[str, list[str]]] = {
            bid: {"player": [], "opponent": []} for bid in all_battle_ids
        }

        for i in range(0, len(all_battle_ids), chunk_size):
            chunk = all_battle_ids[i : i + chunk_size]
            rows = session.execute(
                select(DeckCard.battle_id, DeckCard.card_name, DeckCard.is_player_deck)
                .where(DeckCard.battle_id.in_(chunk))
            ).all()

            for bid, card_name, is_player in rows:
                side = "player" if is_player else "opponent"
                if bid in deck_by_battle:
                    deck_by_battle[bid][side].append(card_name)

        # Build deck index arrays, matching _base._samples order
        skipped_decks = 0
        valid_indices = []
        for idx, bid in enumerate(all_battle_ids):
            if idx >= len(self._base):
                break

            player_cards = deck_by_battle[bid]["player"]
            opponent_cards = deck_by_battle[bid]["opponent"]

            if len(player_cards) < 1 or len(opponent_cards) < 1:
                skipped_decks += 1
                continue

            # Encode deck cards to vocab indices, pad to DECK_SIZE
            p_ids = np.zeros(DECK_SIZE, dtype=np.int64)
            o_ids = np.zeros(DECK_SIZE, dtype=np.int64)

            for j, card in enumerate(player_cards[:DECK_SIZE]):
                p_ids[j] = vocab.encode(card)
            for j, card in enumerate(opponent_cards[:DECK_SIZE]):
                o_ids[j] = vocab.encode(card)

            self._deck_data.append((p_ids, o_ids))
            valid_indices.append(idx)

        # Filter base samples to only those with valid deck data
        if len(valid_indices) < len(self._base):
            self._base._samples = [self._base._samples[i] for i in valid_indices]

        logger.info(
            "CVAEDataset: %d games loaded (%d skipped for missing decks)",
            len(self._deck_data), skipped_decks,
        )

    def __len__(self) -> int:
        return len(self._deck_data)

    def __getitem__(
        self, idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, float, torch.Tensor, torch.Tensor]:
        """Return (card_ids, features, label, player_deck_ids, opponent_deck_ids)."""
        card_ids, features, label = self._base[idx]
        player_deck = torch.from_numpy(self._deck_data[idx][0])
        opponent_deck = torch.from_numpy(self._deck_data[idx][1])
        return card_ids, features, label, player_deck, opponent_deck


def cvae_collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor, float, torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad event sequences and stack deck tensors.

    Returns:
        card_ids: (batch, max_len) int64
        features: (batch, max_len, 17) float32
        lengths: (batch,) int64
        labels: (batch,) float32
        player_deck_ids: (batch, 8) int64
        opponent_deck_ids: (batch, 8) int64
        mask: (batch, max_len) float32
    """
    card_ids_list, features_list, labels_list, p_decks, o_decks = zip(*batch)

    # Truncate sequences to MAX_SEQ_LEN to bound attention cost
    lengths = torch.tensor(
        [min(len(c), MAX_SEQ_LEN) for c in card_ids_list], dtype=torch.int64,
    )
    max_len = int(lengths.max())

    batch_size = len(batch)
    padded_card_ids = torch.zeros(batch_size, max_len, dtype=torch.int64)
    padded_features = torch.zeros(batch_size, max_len, 17, dtype=torch.float32)

    for i, (cids, feats) in enumerate(zip(card_ids_list, features_list)):
        seq_len = int(lengths[i])
        padded_card_ids[i, :seq_len] = cids[:seq_len]
        padded_features[i, :seq_len] = feats[:seq_len]

    labels = torch.tensor(labels_list, dtype=torch.float32)
    player_deck_ids = torch.stack(list(p_decks))
    opponent_deck_ids = torch.stack(list(o_decks))

    # Mask
    arange = torch.arange(max_len).unsqueeze(0)
    mask = (arange < lengths.unsqueeze(1)).float()

    return padded_card_ids, padded_features, lengths, labels, player_deck_ids, opponent_deck_ids, mask
