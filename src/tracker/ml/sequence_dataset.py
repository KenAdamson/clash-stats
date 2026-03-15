"""PyTorch Dataset for replay event sequences.

Converts raw replay events from the database into padded tensor batches
for TCN training. Each event becomes an 18-dim feature vector (card index
is embedded to 16-dim at forward time in the model → 34 total).

Per-event features (18 hand-crafted):
  card_id (1), side (1), game_tick_norm (1), game_phase one-hot (4),
  arena_x_norm (1), arena_y_norm (1), lane one-hot (3), play_number (1),
  ability_used (1), elixir_cost (1), card_type one-hot (3), is_evo (1)
"""

import logging
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from tracker.models import Battle, ReplayEvent, DeckCard
from tracker.ml.card_metadata import CardVocabulary, kebab_to_title

logger = logging.getLogger(__name__)

# Arena midpoints (from features.py)
ARENA_X_MID = 8750
ARENA_Y_MID = 15750

# Max arena bounds for normalization
ARENA_X_MAX = 17500
ARENA_Y_MAX = 31500

# Game tick phase boundaries
PHASE_REGULAR_END = 3360
PHASE_DOUBLE_END = 5280
PHASE_OT_END = 7920

# Cap play_number to avoid outlier influence
PLAY_NUMBER_CAP = 20

# Max game tick for normalization (OT double elixir max)
GAME_TICK_MAX = 10000

# Minimum events per game to include
MIN_EVENTS = 4

# Card type to one-hot index
CARD_TYPE_IDX = {"troop": 0, "spell": 1, "building": 2}


def _game_phase_onehot(tick: int) -> list[float]:
    """One-hot encode game phase from tick value."""
    if tick < PHASE_REGULAR_END:
        return [1.0, 0.0, 0.0, 0.0]
    elif tick < PHASE_DOUBLE_END:
        return [0.0, 1.0, 0.0, 0.0]
    elif tick < PHASE_OT_END:
        return [0.0, 0.0, 1.0, 0.0]
    else:
        return [0.0, 0.0, 0.0, 1.0]


def _lane_onehot(arena_x: int) -> list[float]:
    """One-hot encode lane from arena_x position."""
    margin = 2000
    if arena_x < ARENA_X_MID - margin:
        return [1.0, 0.0, 0.0]  # left
    elif arena_x > ARENA_X_MID + margin:
        return [0.0, 1.0, 0.0]  # right
    else:
        return [0.0, 0.0, 1.0]  # center


def _card_type_onehot(card_type: str) -> list[float]:
    """One-hot encode card type."""
    vec = [0.0, 0.0, 0.0]
    idx = CARD_TYPE_IDX.get(card_type, 0)
    vec[idx] = 1.0
    return vec


class SequenceDataset(Dataset):
    """Dataset of replay event sequences for TCN training.

    Each sample is a variable-length sequence of per-event feature vectors
    plus a win/loss label.

    Args:
        session: SQLAlchemy session.
        vocab: CardVocabulary for card→index mapping.
    """

    def __init__(self, session: Session, vocab: CardVocabulary):
        self.vocab = vocab

        # Build evo set: cards that have ability_used=1 in replay_events
        evo_cards = set(
            session.execute(
                text("SELECT DISTINCT card_name FROM replay_events WHERE ability_used = 1")
            ).scalars().all()
        )
        self._evo_cards = evo_cards

        # Find all PvP battles with sufficient replay events.
        # Use a pre-aggregated JOIN instead of a correlated subquery — the correlated
        # version scans 13M replay_events rows for every battle (O(n*m)), whereas
        # the JOIN aggregates once then filters (O(m) + index lookup).
        battle_rows = session.execute(
            text("""
                SELECT b.battle_id, b.result
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
        ).all()

        logger.info("Loading %d games with replay data", len(battle_rows))

        self._samples: list[tuple[np.ndarray, np.ndarray, float]] = []
        # samples: list of (card_ids, features, label)

        # Batch-load all replay events grouped by battle_id
        battle_ids = [r[0] for r in battle_rows]
        result_map = {r[0]: 1.0 if r[1] == "win" else 0.0 for r in battle_rows}

        # Load events in chunks to avoid massive queries
        chunk_size = 500
        events_by_battle: dict[str, list] = {bid: [] for bid in battle_ids}

        for i in range(0, len(battle_ids), chunk_size):
            chunk = battle_ids[i : i + chunk_size]
            events = session.execute(
                select(ReplayEvent)
                .where(
                    ReplayEvent.battle_id.in_(chunk),
                    ReplayEvent.card_name != "_invalid",
                )
                .order_by(ReplayEvent.battle_id, ReplayEvent.game_tick)
            ).scalars().all()

            for ev in events:
                events_by_battle[ev.battle_id].append(ev)

        # Build feature sequences
        skipped = 0
        for battle_id in battle_ids:
            evts = events_by_battle[battle_id]
            if len(evts) < MIN_EVENTS:
                skipped += 1
                continue

            card_ids = np.zeros(len(evts), dtype=np.int64)
            features = np.zeros((len(evts), 17), dtype=np.float32)

            for j, ev in enumerate(evts):
                title_name = kebab_to_title(ev.card_name)
                card_ids[j] = self.vocab.encode(title_name)

                # side: 1.0 for team, 0.0 for opponent
                features[j, 0] = 1.0 if ev.side == "team" else 0.0

                # game_tick normalized
                features[j, 1] = min(ev.game_tick / GAME_TICK_MAX, 1.0)

                # game_phase one-hot (4)
                features[j, 2:6] = _game_phase_onehot(ev.game_tick)

                # arena_x normalized [-1, 1]
                features[j, 6] = (ev.arena_x - ARENA_X_MID) / ARENA_X_MID

                # arena_y normalized [-1, 1]
                features[j, 7] = (ev.arena_y - ARENA_Y_MID) / ARENA_Y_MID

                # lane one-hot (3)
                features[j, 8:11] = _lane_onehot(ev.arena_x)

                # play_number capped and normalized
                features[j, 11] = min(ev.play_number, PLAY_NUMBER_CAP) / PLAY_NUMBER_CAP

                # ability_used
                features[j, 12] = float(ev.ability_used)

                # elixir_cost normalized (1-10 range)
                elixir = self.vocab.elixir(title_name)
                features[j, 13] = (elixir or 4) / 10.0

                # card_type one-hot (3)
                card_type = self.vocab.card_type(title_name)
                features[j, 14:17] = _card_type_onehot(card_type)

            label = result_map[battle_id]
            self._samples.append((card_ids, features, label))

        logger.info(
            "SequenceDataset: %d games loaded, %d skipped, avg %.1f events/game",
            len(self._samples),
            skipped,
            np.mean([s[1].shape[0] for s in self._samples]) if self._samples else 0,
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Return (card_ids, features, label) for a single game."""
        card_ids, features, label = self._samples[idx]
        return (
            torch.from_numpy(card_ids),
            torch.from_numpy(features),
            label,
        )


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor, float]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad sequences to max length in batch.

    Returns:
        card_ids: (batch, max_len) int64
        features: (batch, max_len, 17) float32
        lengths: (batch,) int64 — original sequence lengths
        labels: (batch,) float32
    """
    card_ids_list, features_list, labels = zip(*batch)
    lengths = torch.tensor([len(c) for c in card_ids_list], dtype=torch.int64)
    max_len = int(lengths.max())

    batch_size = len(batch)
    padded_card_ids = torch.zeros(batch_size, max_len, dtype=torch.int64)
    padded_features = torch.zeros(batch_size, max_len, 17, dtype=torch.float32)

    for i, (cids, feats) in enumerate(zip(card_ids_list, features_list)):
        seq_len = len(cids)
        padded_card_ids[i, :seq_len] = cids
        padded_features[i, :seq_len] = feats

    labels_tensor = torch.tensor(labels, dtype=torch.float32)

    return padded_card_ids, padded_features, lengths, labels_tensor
