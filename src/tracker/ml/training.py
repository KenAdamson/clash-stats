"""TCN training loop and embedding generation (ADR-003 Phase 1).

Handles device detection (XPU/CUDA/CPU), training with early stopping,
inference for all games, UMAP 3D projection, HDBSCAN clustering, and
storage of 128-dim TCN embeddings.
"""

import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from sqlalchemy.orm import Session

from tracker.ml.card_metadata import CardVocabulary
from tracker.ml.sequence_dataset import SequenceDataset, collate_fn
from tracker.ml.tcn import GameEmbeddingModel
from tracker.ml.storage import GameEmbedding, to_blob

logger = logging.getLogger(__name__)

TCN_MODEL_VERSION = "tcn-v1"

# Training hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
DROPOUT = 0.2
EMBEDDING_DIM = 128
VAL_FRACTION = 0.2

# Cap on games embedded per incremental run. Bounds the scoped dataset's IN
# clause and per-run wall time; a large backlog (e.g. after the pipeline was
# down) drains over consecutive cron runs, oldest-first, instead of choking a
# single run. Steady-state new-game counts are far below this, so the cap is
# inert in normal operation. Override via EMBED_MAX_PER_RUN.
MAX_EMBED_PER_RUN = int(os.environ.get("EMBED_MAX_PER_RUN", "5000"))

# Above this NaN fraction, treat the saved UMAP reducer as broken (version
# incompatibility) and abort the run storing nothing, rather than dribbling in
# the unrepresentative minority that happened to project. Below it, a few NaN
# rows are treated as genuinely degenerate inputs and skipped individually.
BROKEN_REDUCER_NAN_FRACTION = 0.25

# DataLoader parallelism for TCN training. num_workers>0 overlaps host-side
# collate with XPU compute (the A770 was ~98% idle starved by num_workers=0).
# Batch size stays at BATCH_SIZE to preserve training dynamics; workers alone
# give the speedup. Both env-overridable. num_workers respects the container's
# low CPU priority (cpu_shares 256, nice +15) so training still yields to Plex.
DATALOADER_NUM_WORKERS = int(os.environ.get("TCN_DATALOADER_WORKERS", "4"))
# Default 512, not BATCH_SIZE (64). On the Arc A770 the bottleneck is per-launch
# XPU dispatch latency, not compute or data loading: batch 64 → ~4,485 tiny
# launches/epoch and ~80 min/epoch with the XPU computing in brief bursts
# (~10% CCS). Batch 512 amortizes the dispatch overhead → ~14 min/epoch (~6×)
# measured 06-11, with the XPU far better utilized. Env-overridable for other
# hardware. (Larger batch slightly slows per-epoch convergence but each epoch is
# ~6× cheaper; val accuracy tracks the same within a couple epochs.)
DATALOADER_BATCH_SIZE = int(os.environ.get("TCN_BATCH_SIZE", "512"))


def _detect_device() -> torch.device:
    """Detect the best available device: XPU → CUDA → CPU."""
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        device = torch.device("xpu")
        logger.info("Using Intel XPU: %s", torch.xpu.get_device_name(0))
        return device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA: %s", torch.cuda.get_device_name(0))
        return device
    logger.info("Using CPU")
    return torch.device("cpu")


class TCNTrainer:
    """Handles TCN model training with early stopping.

    Args:
        model: GameEmbeddingModel instance.
        dataset: SequenceDataset.
        device: Torch device.
        model_dir: Directory to save the best model.
    """

    def __init__(
        self,
        model: GameEmbeddingModel,
        dataset: SequenceDataset,
        device: torch.device,
        model_dir: Path,
    ):
        self.model = model.to(device)
        self.device = device
        self.model_dir = model_dir

        # Train/val split
        n = len(dataset)
        n_val = int(n * VAL_FRACTION)
        n_train = n - n_val

        # Deterministic split (last 20% = val, ordered by battle_time)
        train_indices = list(range(n_train))
        val_indices = list(range(n_train, n))

        # The dataset is fully in-memory (numpy in _samples, no DB I/O in
        # __getitem__), so DataLoader workers parallelize the CPU-side collate/
        # padding and keep the XPU fed. With num_workers=0 the A770 sat ~98%
        # idle (CCS ~1.5%) while a single host thread loaded batches → ~94
        # min/epoch on 358K games. Workers don't change the data or batch order
        # (the sampler lives in the main process), so training math is identical.
        # persistent_workers avoids re-forking the worker pool every epoch.
        loader_kwargs = dict(
            batch_size=DATALOADER_BATCH_SIZE,
            collate_fn=collate_fn,
            num_workers=DATALOADER_NUM_WORKERS,
            persistent_workers=DATALOADER_NUM_WORKERS > 0,
            prefetch_factor=4 if DATALOADER_NUM_WORKERS > 0 else None,
        )
        self.train_loader = DataLoader(
            Subset(dataset, train_indices), shuffle=True, **loader_kwargs,
        )
        self.val_loader = DataLoader(
            Subset(dataset, val_indices), shuffle=False, **loader_kwargs,
        )
        self.full_loader = DataLoader(
            dataset, shuffle=False, **loader_kwargs,
        )

        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = AdamW(
            model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY,
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=EPOCHS)

        logger.info(
            "Train/val split: %d / %d games (%.0f%% / %.0f%%)",
            n_train, n_val, 100 * n_train / n, 100 * n_val / n,
        )

    def train(self) -> Path:
        """Run training loop with early stopping.

        Returns:
            Path to saved best model checkpoint.
        """
        self.model_dir.mkdir(parents=True, exist_ok=True)
        best_path = self.model_dir / "tcn_v1.pt"

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, EPOCHS + 1):
            t0 = time.time()

            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for card_ids, features, lengths, labels in self.train_loader:
                card_ids = card_ids.to(self.device)
                features = features.to(self.device)
                lengths = lengths.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                _, logits = self.model(card_ids, features, lengths)
                logits = logits.squeeze(1)
                loss = self.criterion(logits, labels)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item() * labels.size(0)
                preds = (logits > 0).float()
                train_correct += (preds == labels).sum().item()
                train_total += labels.size(0)

            self.scheduler.step()

            train_loss /= max(train_total, 1)
            train_acc = train_correct / max(train_total, 1)

            # Validation phase
            val_loss, val_acc = self._evaluate(self.val_loader)

            elapsed = time.time() - t0
            logger.info(
                "Epoch %d/%d [%.1fs] — train loss: %.4f acc: %.3f | val loss: %.4f acc: %.3f",
                epoch, EPOCHS, elapsed, train_loss, train_acc, val_loss, val_acc,
            )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    "model_state_dict": self.model.state_dict(),
                    "vocab_size": self.model.card_embedding.num_embeddings,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                }, best_path)
                logger.info("  → New best model saved (val_loss=%.4f)", val_loss)
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    logger.info(
                        "Early stopping at epoch %d (patience=%d)",
                        epoch, EARLY_STOPPING_PATIENCE,
                    )
                    break

        logger.info("Training complete. Best val loss: %.4f", best_val_loss)
        return best_path

    def _evaluate(self, loader: DataLoader) -> tuple[float, float]:
        """Evaluate on a DataLoader, returning (loss, accuracy)."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for card_ids, features, lengths, labels in loader:
                card_ids = card_ids.to(self.device)
                features = features.to(self.device)
                lengths = lengths.to(self.device)
                labels = labels.to(self.device)

                _, logits = self.model(card_ids, features, lengths)
                logits = logits.squeeze(1)
                loss = self.criterion(logits, labels)

                total_loss += loss.item() * labels.size(0)
                preds = (logits > 0).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        return total_loss / max(total, 1), correct / max(total, 1)

    @torch.no_grad()
    def extract_embeddings(self) -> tuple[list[int], np.ndarray]:
        """Run inference on all games and return 128-dim embeddings.

        Returns:
            Tuple of (dataset_indices, embeddings) where embeddings is
            shape (n_games, 128).
        """
        self.model.eval()
        all_embeddings = []

        for card_ids, features, lengths, labels in self.full_loader:
            card_ids = card_ids.to(self.device)
            features = features.to(self.device)
            lengths = lengths.to(self.device)

            embeddings, _ = self.model(card_ids, features, lengths)
            all_embeddings.append(embeddings.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)


def train_tcn(session: Session, model_dir: Optional[Path] = None) -> None:
    """Full TCN pipeline: train, embed, cluster, store.

    Args:
        session: Database session.
        model_dir: Directory for model files.
    """
    from tracker.ml.clustering import label_clusters
    from tracker.ml.umap_embeddings import EmbeddingPipeline

    if model_dir is None:
        model_dir = Path("data/ml_models")

    device = _detect_device()

    # 1. Build vocabulary
    vocab = CardVocabulary(session)
    logger.info("Vocabulary size: %d", vocab.size)

    # 2. Create dataset
    dataset = SequenceDataset(session, vocab)
    if len(dataset) < 50:
        logger.error("Need at least 50 games with replay data (have %d)", len(dataset))
        print(f"  ✗ Need at least 50 games with replay data (have {len(dataset)})")
        print("    Run --fetch-replays first")
        return

    # 3. Initialize model
    model = GameEmbeddingModel(
        vocab_size=vocab.size,
        dropout=DROPOUT,
        embedding_dim=EMBEDDING_DIM,
    )
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model parameters: %s", f"{n_params:,}")

    # 4. Train
    trainer = TCNTrainer(model, dataset, device, model_dir)
    best_path = trainer.train()

    # 5. Load best model for inference
    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    logger.info(
        "Loaded best model from epoch %d (val_loss=%.4f, val_acc=%.3f)",
        checkpoint["epoch"], checkpoint["val_loss"], checkpoint["val_acc"],
    )

    # 6. Extract 128-dim embeddings for all games
    trainer.model = model
    embeddings_128d = trainer.extract_embeddings()
    logger.info("Extracted %d embeddings of dim %d", *embeddings_128d.shape)

    # 7. Get battle_ids in dataset order
    # Dataset samples are ordered by battle_time, matching the DB query order
    from sqlalchemy import text as sa_text
    from tracker.ml.sequence_dataset import MIN_EVENTS

    # Use JOIN instead of correlated subquery to avoid O(n*m) scan on 13M rows
    battle_rows = session.execute(
        sa_text("""
            SELECT b.battle_id
            FROM battles b
            JOIN (
                SELECT battle_id FROM replay_events
                WHERE card_name != '_invalid'
                GROUP BY battle_id HAVING COUNT(*) >= :min_events
            ) re_counts ON re_counts.battle_id = b.battle_id
            WHERE b.battle_type = 'PvP' AND b.result IN ('win', 'loss')
            ORDER BY b.battle_time
        """),
        {"min_events": MIN_EVENTS},
    ).scalars().all()

    battle_ids = list(battle_rows)[:len(embeddings_128d)]

    # 8. UMAP 128d → 3d for visualization
    logger.info("Fitting UMAP 128d → 3d for visualization")
    pipeline = EmbeddingPipeline(model_dir=model_dir)
    embeddings_3d = pipeline.reduce_to_3d(embeddings_128d)

    # 9. HDBSCAN clustering on 128-dim space
    logger.info("Clustering on 128-dim embeddings")
    cluster_ids = label_clusters(embeddings_128d)

    # 10. Store in DB
    logger.info("Storing %d TCN embeddings", len(battle_ids))
    for i, battle_id in enumerate(battle_ids):
        session.merge(GameEmbedding(
            battle_id=battle_id,
            embedding_15d=to_blob(embeddings_128d[i]),  # legacy BLOB
            embedding_3d=to_blob(embeddings_3d[i]),      # legacy BLOB
            embedding_tcn_128d=embeddings_128d[i].tolist(),  # native vector
            embedding_vec_3d=embeddings_3d[i].tolist(),      # native vector
            cluster_id=int(cluster_ids[i]) if cluster_ids[i] >= 0 else None,
            model_version=TCN_MODEL_VERSION,
        ))

        if (i + 1) % 500 == 0:
            session.flush()

    session.commit()

    print(f"  ✓ TCN training complete: {len(battle_ids)} games embedded")
    print(f"  ✓ Model: {n_params:,} params, best val_loss={checkpoint['val_loss']:.4f}, "
          f"val_acc={checkpoint['val_acc']:.3f}")
    print(f"  ✓ Embeddings: {EMBEDDING_DIM}d → UMAP 3d, HDBSCAN clustered")
    print(f"  ✓ Saved to {best_path}")


def embed_new(session: Session, model_dir: Optional[Path] = None) -> int:
    """Inference-only: embed games that don't have embeddings yet.

    Loads the trained TCN and saved UMAP 3D reducer, runs forward pass
    on new games only, and stores embeddings. No retraining.

    Args:
        session: Database session.
        model_dir: Directory containing tcn_v1.pt and umap_3d_standalone.pkl.

    Returns:
        Number of newly embedded games.
    """
    import pickle
    from sqlalchemy import text as sa_text
    from tracker.ml.sequence_dataset import MIN_EVENTS

    if model_dir is None:
        model_dir = Path("data/ml_models")

    tcn_path = model_dir / "tcn_v1.pt"
    umap_path = model_dir / "umap_3d_standalone.pkl"

    if not tcn_path.exists():
        print("  ✗ No trained TCN model found. Run --train-tcn first.")
        return 0
    if not umap_path.exists():
        print("  ✗ No fitted UMAP reducer found. Run --train-tcn first.")
        return 0

    # 1. Find battles with replay data but no embedding (JOIN avoids correlated
    #    subquery), oldest-first, capped at MAX_EMBED_PER_RUN. The cap bounds the
    #    scoped dataset below and lets a large backlog drain over several runs in
    #    chronological order rather than choking a single run.
    new_rows = session.execute(
        sa_text("""
            SELECT b.battle_id
            FROM battles b
            JOIN (
                SELECT battle_id FROM replay_events
                WHERE card_name != '_invalid'
                GROUP BY battle_id HAVING COUNT(*) >= :min_events
            ) re_counts ON re_counts.battle_id = b.battle_id
            WHERE b.battle_type = 'PvP' AND b.result IN ('win', 'loss')
              AND b.battle_id NOT IN (
                  SELECT ge.battle_id FROM game_embeddings ge
                  WHERE ge.model_version = :model_version
              )
            ORDER BY b.battle_time
            LIMIT :max_per_run
        """),
        {
            "min_events": MIN_EVENTS,
            "model_version": TCN_MODEL_VERSION,
            "max_per_run": MAX_EMBED_PER_RUN,
        },
    ).scalars().all()

    if not new_rows:
        print("  · All games already embedded — nothing to do")
        return 0

    capped = len(new_rows) == MAX_EMBED_PER_RUN
    logger.info(
        "Embedding %d new games this run%s",
        len(new_rows),
        f" (capped at {MAX_EMBED_PER_RUN}; more remain, will drain next run)" if capped else "",
    )
    print(f"  → {len(new_rows)} new games to embed"
          + (f" (per-run cap {MAX_EMBED_PER_RUN}; backlog drains over consecutive runs)" if capped else ""))

    # 1a. Reducer pre-flight. The expensive steps below (scoped dataset build +
    #     TCN inference) are wasted if the saved reducer is broken and NaNs on
    #     output. Probe it against a random sample of already-stored 128d
    #     embeddings first; a high NaN rate means the reducer is version-
    #     incompatible — abort before doing the work. Loaded reducer is reused
    #     in step 6. Skipped when nothing is stored yet (nothing to probe with).
    import json as _json
    umap_reducer = None
    probe = session.execute(sa_text(
        "SELECT embedding_tcn_128d FROM game_embeddings "
        "WHERE embedding_tcn_128d IS NOT NULL ORDER BY random() LIMIT 64"
    )).scalars().all()
    if probe:
        with open(umap_path, "rb") as f:
            umap_reducer = pickle.load(f)
        P = np.array([_json.loads(r) if isinstance(r, str) else list(r) for r in probe],
                     dtype=np.float32)
        probe_nan = float(np.isnan(umap_reducer.transform(P)).any(axis=1).mean())
        if probe_nan > BROKEN_REDUCER_NAN_FRACTION:
            logger.error(
                "UMAP reducer pre-flight: NaN on %.0f%% of %d probe embeddings — "
                "umap_3d_standalone.pkl is incompatible with the current "
                "umap/numba/numpy versions. Aborting before TCN inference. "
                "Refit the 128d→3d reducer to resume.",
                100 * probe_nan, len(P),
            )
            print(f"  ✗ UMAP reducer broken (pre-flight NaN {100*probe_nan:.0f}%) — "
                  "aborting before inference. Refit umap_3d_standalone.pkl.")
            return 0

    # 2. Vocabulary + checkpoint
    device = _detect_device()
    vocab = CardVocabulary(session)
    checkpoint = torch.load(tcn_path, map_location=device, weights_only=True)
    saved_vocab_size = checkpoint["vocab_size"]
    if vocab.size > saved_vocab_size:
        logger.warning(
            "Vocabulary grew (%d → %d). New cards will use index 0 (unknown).",
            saved_vocab_size, vocab.size,
        )

    # 3. Build a SCOPED dataset over only the new games — not the full corpus.
    #    The dataset exposes battle_ids_in_order (aligned with its samples and
    #    skipping any that fail the MIN_EVENTS load check), so embeddings map
    #    straight back to battles without a second full-table query.
    dataset = SequenceDataset(session, vocab, battle_ids=new_rows)
    new_battle_ids = dataset.battle_ids_in_order

    if len(dataset) == 0:
        print("  · New games didn't survive dataset filtering — no events?")
        return 0

    logger.info("Embedding %d new games (scoped dataset)", len(dataset))

    # 4. Load trained model
    model = GameEmbeddingModel(
        vocab_size=saved_vocab_size,
        dropout=0.0,  # inference mode — no dropout
        embedding_dim=EMBEDDING_DIM,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # 5. Run inference. The dataset already contains only the new games, so we
    #    iterate it directly (no Subset). DataLoader preserves order
    #    (shuffle=False), keeping embeddings aligned with battle_ids_in_order.
    new_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    all_embeddings = []
    with torch.no_grad():
        for card_ids, features, lengths, labels in new_loader:
            card_ids = card_ids.to(device)
            features = features.to(device)
            lengths = lengths.to(device)
            embeddings, _ = model(card_ids, features, lengths)
            all_embeddings.append(embeddings.cpu().numpy())

    embeddings_128d = np.concatenate(all_embeddings, axis=0)
    logger.info("Extracted %d embeddings of dim %d", *embeddings_128d.shape)

    # 6. UMAP transform (not fit!) using saved reducer. Reuse the reducer loaded
    #    in the pre-flight; load now only if there was nothing to probe with.
    if umap_reducer is None:
        with open(umap_path, "rb") as f:
            umap_reducer = pickle.load(f)

    embeddings_3d = umap_reducer.transform(embeddings_128d)
    logger.info("Projected to 3D via saved UMAP reducer")

    # 6a. Guard against a version-incompatible reducer. A UMAP/numba/numpy
    #     version bump can leave umap_3d_standalone.pkl loadable but broken —
    #     transform() then returns NaN for valid input (observed 06-10 after the
    #     1.8→1.9 stack bump: 97% NaN on a random 1000-game sample, deterministic).
    #     Storing NaN fails the pgvector insert ("NaN not allowed in vector").
    #     A HIGH NaN rate means the reducer itself is broken: abort and store
    #     nothing (the lucky few that survive are a biased subset — dribbling
    #     them in would pollute the manifold). Only a LOW rate is treated as
    #     genuine degenerate inputs and skipped row-wise. Either way the skipped
    #     games stay un-embedded and get picked up once the reducer is refit.
    nan_rows = np.isnan(embeddings_3d).any(axis=1)
    nan_frac = float(nan_rows.mean()) if len(nan_rows) else 0.0
    if nan_frac > BROKEN_REDUCER_NAN_FRACTION:
        logger.error(
            "UMAP reducer produced NaN for %.0f%% of %d games — "
            "umap_3d_standalone.pkl is incompatible with the current "
            "umap/numba/numpy versions. Aborting (no rows stored). Refit the "
            "128d→3d reducer before incremental embedding can resume.",
            100 * nan_frac, len(embeddings_3d),
        )
        print(f"  ✗ UMAP reducer producing NaN for {100*nan_frac:.0f}% of games "
              "(version-incompatible) — aborting, no rows stored. "
              "Refit umap_3d_standalone.pkl.")
        return 0
    if nan_rows.any():
        keep = ~nan_rows
        logger.warning(
            "Skipping %d/%d games with a NaN 3D projection (degenerate input)",
            int(nan_rows.sum()), len(embeddings_3d),
        )
        embeddings_128d = embeddings_128d[keep]
        embeddings_3d = embeddings_3d[keep]
        new_battle_ids = [b for b, k in zip(new_battle_ids, keep) if k]

    # 7. Store (no cluster assignment — would need full re-clustering)
    for i, battle_id in enumerate(new_battle_ids):
        session.merge(GameEmbedding(
            battle_id=battle_id,
            embedding_15d=to_blob(embeddings_128d[i]),  # legacy BLOB
            embedding_3d=to_blob(embeddings_3d[i]),      # legacy BLOB
            embedding_tcn_128d=embeddings_128d[i].tolist(),  # native vector
            embedding_vec_3d=embeddings_3d[i].tolist(),      # native vector
            cluster_id=None,
            model_version=TCN_MODEL_VERSION,
        ))

    session.commit()

    print(f"  ✓ Embedded {len(new_battle_ids)} new games (128d → 3d)")
    print(f"  ✓ Inference only — no retraining")
    return len(new_battle_ids)
