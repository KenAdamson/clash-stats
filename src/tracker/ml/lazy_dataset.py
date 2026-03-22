"""Lazy-loading dataset for replay event sequences.

Unlike SequenceDataset which loads all games into memory at init,
LazySequenceDataset stores only battle IDs and queries replay events
from PostgreSQL on demand in __getitem__. Memory usage is constant
regardless of dataset size.

Uses keyset pagination for efficient ordered access.
"""

import logging
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from tracker.ml.card_metadata import CardVocabulary, kebab_to_title, CARD_TYPES
from tracker.ml.sequence_dataset import (
    MIN_EVENTS,
    GAME_TICK_MAX,
    PLAY_NUMBER_CAP,
    ARENA_X_MID,
    ARENA_Y_MID,
    _game_phase_onehot,
    _lane_onehot,
    _card_type_onehot,
    collate_fn,  # re-export for convenience
)

logger = logging.getLogger(__name__)


class LazyBatchLoader:
    """Drop-in replacement for DataLoader that uses bulk DB queries.

    Instead of DataLoader's per-sample __getitem__ + collate pattern,
    this fetches an entire batch of games in one PostgreSQL query via
    LazySequenceDataset.load_batch(), then collates.

    Compatible with WP training loop — yields the same tuple format
    as DataLoader with wp_collate_fn.

    Args:
        dataset: LazySequenceDataset instance.
        indices: Which dataset indices to iterate over.
        batch_size: Samples per batch.
        shuffle: Shuffle order each iteration.
        collate_fn: Function to collate samples into tensors.
    """

    def __init__(
        self,
        dataset: "LazySequenceDataset",
        indices: list[int],
        batch_size: int,
        shuffle: bool = True,
        collate_fn=None,
    ):
        self.dataset = dataset
        self.indices = indices
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.collate_fn = collate_fn or (lambda x: x)

    def __iter__(self):
        if self.shuffle:
            order = torch.randperm(len(self.indices)).tolist()
            indices = [self.indices[i] for i in order]
        else:
            indices = self.indices

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            samples = self.dataset.load_batch(batch_indices)
            if samples:
                yield self.collate_fn(samples)

    def __len__(self):
        return (len(self.indices) + self.batch_size - 1) // self.batch_size


class BatchQuerySampler:
    """Sampler that yields batch-sized index lists for bulk DB queries.

    Used with LazySequenceDataset.load_batch() to fetch an entire batch
    in one PostgreSQL query instead of one query per sample.
    """

    def __init__(self, n: int, batch_size: int, shuffle: bool = True):
        self._n = n
        self._batch_size = batch_size
        self._shuffle = shuffle

    def __iter__(self):
        if self._shuffle:
            indices = torch.randperm(self._n).tolist()
        else:
            indices = list(range(self._n))
        for i in range(0, self._n, self._batch_size):
            yield indices[i:i + self._batch_size]

    def __len__(self):
        return (self._n + self._batch_size - 1) // self._batch_size


class LazySequenceDataset(Dataset):
    """Lazy-loading dataset that queries PostgreSQL per-sample.

    Memory usage: ~10 bytes per game (battle_id + label) vs ~4KB per
    game for SequenceDataset. At 74K games: ~740KB vs ~300MB.

    Args:
        session: SQLAlchemy session (used for init query only).
        vocab: CardVocabulary for card→index mapping.
        battle_ids: Optional list of specific battle IDs.
        db_url: Database URL for creating per-query connections.
                If None, extracted from session's engine.
    """

    def __init__(
        self,
        session: Session,
        vocab: CardVocabulary,
        battle_ids: list[str] | None = None,
        db_url: str | None = None,
    ):
        self.vocab = vocab

        # Store DB URL for per-query connections in __getitem__
        if db_url:
            self._db_url = db_url
        else:
            self._db_url = str(session.bind.url)

        # Query battle IDs and results — lightweight, just strings
        if battle_ids is not None:
            rows = session.execute(
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
                      AND b.battle_id = ANY(:bids)
                    ORDER BY b.battle_time
                """),
                {"min_events": MIN_EVENTS, "bids": list(battle_ids)},
            ).all()
        else:
            rows = session.execute(
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

        # Store only IDs and labels — ~10 bytes per game
        self._battle_ids = [r[0] for r in rows]
        self._labels = [1.0 if r[1] == "win" else 0.0 for r in rows]

        logger.info(
            "LazySequenceDataset: %d games indexed (no events loaded yet)",
            len(self._battle_ids),
        )

    def __len__(self) -> int:
        return len(self._battle_ids)

    @property
    def battle_ids(self) -> list[str]:
        """Ordered list of battle IDs in this dataset."""
        return self._battle_ids

    def _get_engine(self):
        """Get or create a shared engine for lazy queries."""
        if not hasattr(self, '_engine') or self._engine is None:
            self._engine = create_engine(
                self._db_url, pool_size=5, max_overflow=5, pool_recycle=300,
            )
        return self._engine

    def _build_sample(self, rows) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert event rows to (card_ids, features) tensors."""
        n_events = len(rows)
        card_ids = np.zeros(n_events, dtype=np.int64)
        features = np.zeros((n_events, 17), dtype=np.float32)

        for j, (card_name, side, game_tick, arena_x, arena_y,
                play_number, ability_used) in enumerate(rows):
            title_name = kebab_to_title(card_name)
            card_ids[j] = self.vocab.encode(title_name)

            features[j, 0] = 1.0 if side == "team" else 0.0
            features[j, 1] = min(game_tick / GAME_TICK_MAX, 1.0)
            features[j, 2:6] = _game_phase_onehot(game_tick)
            features[j, 6] = (arena_x - ARENA_X_MID) / ARENA_X_MID
            features[j, 7] = (arena_y - ARENA_Y_MID) / ARENA_Y_MID
            features[j, 8:11] = _lane_onehot(arena_x)
            features[j, 11] = min(play_number, PLAY_NUMBER_CAP) / PLAY_NUMBER_CAP
            features[j, 12] = float(ability_used)
            elixir = self.vocab.elixir(title_name)
            features[j, 13] = (elixir or 4) / 10.0
            card_type = CARD_TYPES.get(title_name, "troop")
            features[j, 14:17] = _card_type_onehot(card_type)

        return torch.from_numpy(card_ids), torch.from_numpy(features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Load a single game's events from PostgreSQL on demand."""
        battle_id = self._battle_ids[idx]
        label = self._labels[idx]

        engine = self._get_engine()
        with engine.connect() as conn:
            rows = conn.execute(
                text("""
                    SELECT card_name, side, game_tick, arena_x, arena_y,
                           play_number, ability_used
                    FROM replay_events
                    WHERE battle_id = :bid AND card_name != '_invalid'
                    ORDER BY game_tick
                """),
                {"bid": battle_id},
            ).all()

        card_ids, features = self._build_sample(rows)
        return card_ids, features, label

    def load_batch(
        self, indices: list[int],
    ) -> list[tuple[torch.Tensor, torch.Tensor, float]]:
        """Load a batch of games in a single DB query.

        One round-trip for the entire batch instead of one per sample.
        Call this from a custom collate_fn or batch sampler.

        Args:
            indices: Dataset indices to load.

        Returns:
            List of (card_ids, features, label) tuples.
        """
        batch_bids = [self._battle_ids[i] for i in indices]
        batch_labels = {self._battle_ids[i]: self._labels[i] for i in indices}

        engine = self._get_engine()
        with engine.connect() as conn:
            rows = conn.execute(
                text("""
                    SELECT battle_id, card_name, side, game_tick, arena_x,
                           arena_y, play_number, ability_used
                    FROM replay_events
                    WHERE battle_id = ANY(:bids) AND card_name != '_invalid'
                    ORDER BY battle_id, game_tick
                """),
                {"bids": batch_bids},
            ).all()

        # Group by battle_id
        from collections import defaultdict
        events_by_bid: dict[str, list] = defaultdict(list)
        for row in rows:
            events_by_bid[row[0]].append(row[1:])  # strip battle_id

        # Build samples in the requested order
        samples = []
        for bid in batch_bids:
            evts = events_by_bid.get(bid, [])
            if not evts:
                continue
            card_ids, features = self._build_sample(evts)
            samples.append((card_ids, features, batch_labels[bid]))

        return samples
