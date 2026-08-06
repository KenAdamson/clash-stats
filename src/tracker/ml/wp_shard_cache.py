"""Pre-extracted, memory-mapped WP training shards — the "firehose" data path.

The in-memory pipeline feeds training from per-game Python objects assembled out
of PostgreSQL, with a single-threaded pad-per-batch collate. At 1.7M games that
collate is the bottleneck: the XPU sits idle while one CPU thread pads sequences.

This module extracts the dataset ONCE to fixed-shape memory-mapped numpy files.
Training then reads batches by fancy-indexing the memmaps — no per-event Python,
no per-batch padding; the OS page cache and NVMe do the feeding. (At petabyte
scale the same idea is sharded per-worker over a network FS; at our ~16GB scale a
single set of memmaps on local NVMe saturates the trainer without workers.)

Shard directory layout:
  card_ids.npy   int16   [N, L]     (vocab is ~124 — int16 is plenty)
  features.npy   float16 [N, L, F]  (normalized features are small — fp16 safe)
  lengths.npy    int32   [N]
  labels.npy     uint8   [N]
  battle_ids.txt         N lines, row-aligned
  meta.json              {n, max_len, feature_dim, extra_features, built_at_utc,
                          truncated_games, source}

Rows are ordered by battle_time ascending, so the time-ordered train/val split is
an index range — identical semantics to the in-memory path.
"""

import json
import logging
import os
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

MAX_LEN = 192  # P99.9 of events-per-game (P50=54, P99=140, max~376); tail truncates


def build_shards(
    session,
    out_dir: str,
    vocab=None,
    extra_features: bool = True,
    max_games: int | None = None,
    max_len: int = MAX_LEN,
    battle_rows: list | None = None,
) -> dict:
    """Extract the full training set to memory-mapped shard files.

    Streams the exact same feature pipeline as SequenceDataset (via sample_sink,
    so feature logic cannot drift) but writes rows straight into preallocated
    memmaps instead of accumulating Python objects.

    Args:
        session: SQLAlchemy session.
        out_dir: Directory to write shard files into (created if needed).
        vocab: CardVocabulary (built from session if None).
        extra_features: Include the board-truth feature dims.
        max_games: Optional cap (most-recent N), mirrors WP_MAX_GAMES.
        max_len: Fixed sequence length; longer games truncate (masked anyway).

    Returns:
        meta dict (also written to meta.json).
    """
    from tracker.ml.card_metadata import CardVocabulary
    from tracker.ml.sequence_dataset import (
        SequenceDataset, select_training_battles,
        BASE_FEATURE_DIM, EXTRA_FEATURE_DIM, DECK_SIZE,
    )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if vocab is None:
        vocab = CardVocabulary(session)

    # battle_rows may be injected to pin an exact battle set (reproducible builds;
    # the corpus grows continuously, so two select calls minutes apart differ).
    if battle_rows is None:
        battle_rows = select_training_battles(session, max_games=max_games)
    n = len(battle_rows)
    feature_dim = BASE_FEATURE_DIM + (EXTRA_FEATURE_DIM if extra_features else 0)
    logger.info("Building WP shards: %d games, L=%d, F=%d -> %s", n, max_len, feature_dim, out)

    card_ids_mm = np.lib.format.open_memmap(
        out / "card_ids.npy", mode="w+", dtype=np.int16, shape=(n, max_len))
    features_mm = np.lib.format.open_memmap(
        out / "features.npy", mode="w+", dtype=np.float16, shape=(n, max_len, feature_dim))
    lengths_mm = np.lib.format.open_memmap(
        out / "lengths.npy", mode="w+", dtype=np.int32, shape=(n,))
    labels_mm = np.lib.format.open_memmap(
        out / "labels.npy", mode="w+", dtype=np.uint8, shape=(n,))
    # Deck prior: per GAME, not per tick. 16 int16 + 16 int8 per game is ~50MB
    # across the full 1.7M pool, versus ~55GB had it been stored per timestep —
    # and it keeps the block-shuffled sequential read the firehose was built for.
    deck_ids_mm = np.lib.format.open_memmap(
        out / "deck_ids.npy", mode="w+", dtype=np.int16, shape=(n, 2, DECK_SIZE))
    deck_vars_mm = np.lib.format.open_memmap(
        out / "deck_vars.npy", mode="w+", dtype=np.int8, shape=(n, 2, DECK_SIZE))

    state = {"row": 0, "truncated": 0}
    bid_file = open(out / "battle_ids.txt", "w")

    def sink(battle_id: str, card_ids: np.ndarray, features: np.ndarray, label: float,
             deck_ids: np.ndarray, deck_vars: np.ndarray):
        i = state["row"]
        ln = len(card_ids)
        if ln > max_len:
            state["truncated"] += 1
            card_ids = card_ids[:max_len]
            features = features[:max_len]
            ln = max_len
        card_ids_mm[i, :ln] = card_ids
        features_mm[i, :ln] = features.astype(np.float16)
        lengths_mm[i] = ln
        labels_mm[i] = int(label)
        deck_ids_mm[i] = deck_ids
        deck_vars_mm[i] = deck_vars
        bid_file.write(battle_id + "\n")
        state["row"] += 1

    # Drives the exact SequenceDataset streaming build; samples go to the sink
    # instead of RAM. MIN_EVENTS-skipped games never reach the sink.
    SequenceDataset(session, vocab, extra_features=extra_features,
                    battle_rows=battle_rows, sample_sink=sink)
    bid_file.close()

    n_written = state["row"]
    # MIN_EVENTS skips mean n_written <= n; trim by rewriting lengths of the tail
    # to 0 is wasteful — instead record the true count in meta and have readers
    # honor it. (Rows are written densely, so [0, n_written) is valid.)
    from datetime import datetime, timezone
    meta = {
        "n": n_written,
        "n_allocated": n,
        "max_len": max_len,
        "feature_dim": feature_dim,
        "extra_features": extra_features,
        "deck_features": True,
        "deck_size": DECK_SIZE,
        "truncated_games": state["truncated"],
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "battles PvP+pathOfLegend win/loss, time-ascending",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    for mm in (card_ids_mm, features_mm, lengths_mm, labels_mm,
               deck_ids_mm, deck_vars_mm):
        mm.flush()
    logger.info("WP shards built: %d rows (%d truncated to L=%d)",
                n_written, state["truncated"], max_len)
    return meta


class ShardDataset:
    """Memory-mapped shard-backed dataset (read side).

    Not a torch Dataset — batches come from ShardBatchLoader via vectorized
    fancy-indexing, not per-item __getitem__.
    """

    def __init__(self, shard_dir: str):
        d = Path(shard_dir)
        self.meta = json.loads((d / "meta.json").read_text())
        self.n = self.meta["n"]
        self.max_len = self.meta["max_len"]
        self.feature_dim = self.meta["feature_dim"]
        self.card_ids = np.load(d / "card_ids.npy", mmap_mode="r")
        self.features = np.load(d / "features.npy", mmap_mode="r")
        self.lengths = np.load(d / "lengths.npy", mmap_mode="r")
        self.labels = np.load(d / "labels.npy", mmap_mode="r")
        # Older shard dirs predate the deck prior; absent files mean the loader
        # yields zeros, which the PAD embedding maps to a zero vector — so a v9
        # shard set still trains a v9-shaped model without a rebuild.
        # Per-game rarity weights (optional). Applied by scaling the loss mask,
        # which the trainer already normalises by — so a weight is a weighted
        # mean, with no change to the loss code or the batch signature.
        self.has_weights = (d / "deck_weights.npy").exists()
        self.deck_weights = (np.load(d / "deck_weights.npy", mmap_mode="r")
                             if self.has_weights else None)
        self.has_decks = (d / "deck_ids.npy").exists()
        self.deck_ids = np.load(d / "deck_ids.npy", mmap_mode="r") if self.has_decks else None
        self.deck_vars = np.load(d / "deck_vars.npy", mmap_mode="r") if self.has_decks else None
        self.battle_ids_in_order = (d / "battle_ids.txt").read_text().splitlines()
        logger.info("ShardDataset: %d games (L=%d, F=%d) memmapped from %s",
                    self.n, self.max_len, self.feature_dim, shard_dir)

    def __len__(self) -> int:
        return self.n


class ShardBatchLoader:
    """Firehose batch iterator over a ShardDataset — block-shuffled sequential IO.

    Uniform-random row gathers thrash the page cache when the shard file rivals
    free RAM (measured: 103GB read for a 16GB file, one core pinned on page
    faults). So randomness is applied WebDataset-style at two levels that keep
    the IO sequential: blocks of contiguous rows are visited in random order,
    each block is read with ONE sequential slice (NVMe-friendly, ~GB/s), and rows
    are permuted within the block before batching. Fresh block order + fresh
    permutations every epoch. Emits the same (card_ids, features, lengths,
    labels, mask, deck_ids, deck_vars) tuples as wp_collate_fn.
    """

    BLOCK_ROWS = 65536  # ~600MB of fp16 features per sequential read, ~128 batches

    # Fraction of TRAINING rows whose deck prior is blanked to the unknown state.
    #
    # Not a regulariser — a distribution fix. Replay-link stub battles
    # (_ensure_link_battle) carry replay events but no deck_cards by design, so
    # ~1.8% of the pool has no deck at all. Corpus replay scraping ramped in
    # mid-June, so every stub sits in the recent tail, and the time-ordered split
    # puts 100% of them in validation: 9.01% of val, 0.00% of train. Training on
    # decks that are always present and then validating on a zero vector 9% of
    # the time feeds the model an input it has never seen, which would penalise
    # the deck prior for a data artefact rather than measure it. Masking teaches
    # the unknown state, and it matches production, where stubs keep arriving.
    DECK_MASK_RATE = float(os.environ.get("WP_DECK_MASK_RATE", "0.10"))

    # Weight the loss by how unusual a deck's SHAPE is. Off by default: it
    # deliberately distorts the training distribution, so it is a decision, not
    # a default. Motivation is measured -- 510 spell-heavy games in 1.68M carry
    # 0.03% of the gradient, and the model opens them at 47% when they really
    # win 21%, an error the aggregate loss cannot see.
    RARITY_WEIGHTS = os.environ.get("WP_RARITY_WEIGHTS", "0") == "1"

    def __init__(self, dataset: ShardDataset, indices, batch_size: int, shuffle: bool):
        self.ds = dataset
        self.indices = np.asarray(indices, dtype=np.int64)
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self) -> int:
        return (len(self.indices) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        idx = self.indices
        n_blocks = (len(idx) + self.BLOCK_ROWS - 1) // self.BLOCK_ROWS
        block_order = np.random.permutation(n_blocks) if self.shuffle else np.arange(n_blocks)
        for bi in block_order:
            rows = idx[bi * self.BLOCK_ROWS:(bi + 1) * self.BLOCK_ROWS]
            a, b = int(rows[0]), int(rows[-1]) + 1
            if len(rows) == b - a:
                # Contiguous (the normal case — train/val indices are ranges):
                # one sequential slice per array, materialized into RAM.
                cid_blk = np.asarray(self.ds.card_ids[a:b])
                feat_blk = np.asarray(self.ds.features[a:b])
                len_blk = np.asarray(self.ds.lengths[a:b])
                lab_blk = np.asarray(self.ds.labels[a:b])
                dck_blk = np.asarray(self.ds.deck_ids[a:b]) if self.ds.has_decks else None
                dvr_blk = np.asarray(self.ds.deck_vars[a:b]) if self.ds.has_decks else None
                wgt_blk = np.asarray(self.ds.deck_weights[a:b]) if self.ds.has_weights else None
            else:
                cid_blk = self.ds.card_ids[rows]
                feat_blk = self.ds.features[rows]
                len_blk = self.ds.lengths[rows]
                lab_blk = self.ds.labels[rows]
                dck_blk = self.ds.deck_ids[rows] if self.ds.has_decks else None
                dvr_blk = self.ds.deck_vars[rows] if self.ds.has_decks else None
                wgt_blk = self.ds.deck_weights[rows] if self.ds.has_weights else None
            perm = (np.random.permutation(len(rows)) if self.shuffle
                    else np.arange(len(rows)))
            for s in range(0, len(rows), self.batch_size):
                pb = perm[s:s + self.batch_size]
                lengths = len_blk[pb].astype(np.int64)
                L = max(int(lengths.max()), 1)
                lengths_t = torch.from_numpy(lengths)
                card_ids = torch.from_numpy(cid_blk[pb, :L].astype(np.int64))
                features = torch.from_numpy(feat_blk[pb, :L].astype(np.float32))
                game_labels = torch.from_numpy(lab_blk[pb].astype(np.float32))
                labels = game_labels.unsqueeze(1).expand(len(pb), L)
                mask = (torch.arange(L).unsqueeze(0) < lengths_t.unsqueeze(1)).float()
                # Training only. Never during eval, or metrics stop being
                # comparable between models.
                if self.shuffle and self.RARITY_WEIGHTS and wgt_blk is not None:
                    mask = mask * torch.from_numpy(
                        wgt_blk[pb].astype(np.float32)).unsqueeze(1)
                if dck_blk is not None:
                    deck_ids = torch.from_numpy(dck_blk[pb].astype(np.int64))
                    deck_vars = torch.from_numpy(dvr_blk[pb].astype(np.int64))
                    # shuffle=True marks the training pass; never mask during eval.
                    if self.shuffle and self.DECK_MASK_RATE > 0:
                        drop = torch.rand(deck_ids.size(0)) < self.DECK_MASK_RATE
                        deck_ids[drop] = 0
                        deck_vars[drop] = 0
                else:
                    deck_ids = torch.zeros(len(pb), 2, 8, dtype=torch.int64)
                    deck_vars = torch.zeros(len(pb), 2, 8, dtype=torch.int64)
                yield card_ids, features, lengths_t, labels, mask, deck_ids, deck_vars
            del cid_blk, feat_blk, len_blk, lab_blk, dck_blk, dvr_blk, wgt_blk
