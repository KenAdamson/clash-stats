"""Win probability training loop (ADR-004).

Handles training the causal TCN win probability model with per-tick
BCE loss, optional transfer learning from ADR-003, Platt scaling
calibration, and per-game WPA inference + storage.
"""

import contextlib
import logging
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Subset
from sqlalchemy import text as sa_text
from sqlalchemy.orm import Session

from tracker.ml.calibration import PlattCalibrator
from tracker.ml.card_metadata import CardVocabulary, kebab_to_title
from tracker.ml.sequence_dataset import SequenceDataset, MIN_EVENTS
from tracker.ml.wp_dataset import wp_collate_fn
from tracker.ml.win_probability import WinProbabilityModel
from tracker.ml.wp_storage import WinProbability, GameWPSummary

logger = logging.getLogger(__name__)

WP_MODEL_VERSION = "wp-v1"

# Opt into the +5 board-truth features (exact-elixir + opponent-skill). When
# True, WP trains on the 22-dim vector and CANNOT transfer from the 17-dim
# production/TCN encoders (input width differs) — it trains from scratch.
WP_EXTRA_FEATURES = True

# Watermark for incremental WP inference: highest replay_events.id already
# examined. Keyed on event-row ARRIVAL (autoincrement id), not battle_time —
# replays land hours/days after their battle, so a time-based watermark would
# skip late arrivals. Overlap re-examines a tail slice each run to absorb
# out-of-order commits near the previous max; the game_wp_summary anti-join
# makes reprocessing idempotent. Delete the file to force one full-scan run.
WP_WATERMARK_PATH = Path("data/wp_infer_watermark.txt")
WP_WATERMARK_OVERLAP = 50_000


def _read_wp_watermark(path: Path = WP_WATERMARK_PATH) -> Optional[int]:
    """Return the stored watermark id, or None for a full-scan run."""
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _write_wp_watermark(value: int, path: Path = WP_WATERMARK_PATH) -> None:
    """Persist the watermark; failure is non-fatal (next run rescans more)."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(str(value))
    except OSError:
        logger.warning("Could not persist WP inference watermark to %s", path)


def _resolve_wp_path(session: Session, model_dir: Path) -> Optional[Path]:
    """Resolve the production WP model path.

    Checks the model registry first, falls back to wp_v1.pt for
    backwards compatibility with pre-registry checkpoints.
    """
    try:
        from tracker.ml.model_registry import get_production_filename
        prod_filename = get_production_filename(session, "wp")
        if prod_filename:
            path = model_dir / prod_filename
            if path.exists():
                return path
    except Exception:
        pass  # Registry table may not exist yet

    # Fallback: scan for latest wp_vN.pt
    candidates = sorted(model_dir.glob("wp_v*.pt"), reverse=True)
    if candidates:
        return candidates[0]
    return None


def _sanitize_state_dict(state_dict: dict) -> int:
    """Replace NaN entries in float tensors with 0.0, in place.

    Some saved WP checkpoints contain a handful of NaN weights (likely a
    training-time gradient blowup that wasn't clipped). Without this, ~0.1%
    of games produce all-NaN per-tick output because the NaN kernel positions
    propagate forward through the TCN. Replacing NaN with 0 degrades those
    channels slightly but yields valid inference; ~0.08% of games is the
    observed affected population, recovered fully by this patch.

    Returns the number of NaN values replaced (0 if checkpoint is clean).
    """
    total = 0
    for k, v in state_dict.items():
        if torch.is_tensor(v) and v.dtype.is_floating_point:
            n = int(torch.isnan(v).sum())
            if n:
                state_dict[k] = torch.where(torch.isnan(v), torch.zeros_like(v), v)
                total += n
    return total


# Training hyperparameters
BATCH_SIZE = 4096
LEARNING_RATE = 5e-4
# From-scratch (full-encoder) training is far more unstable than the frozen-
# encoder transfer path: a lower peak LR + LR warmup lets Adam's second-moment
# estimates settle before the full rate kicks in. Without warmup, a parameter
# that sits at ~zero grad for many steps then receives a real grad yields a
# giant update (lr·m/(√v+eps)) that overflows fp32 a few epochs in — this is
# what NaN-corrupted the wp_v4 run at epoch 11 (guards catch a bad batch but
# can't undo an already-corrupted weight).
FROM_SCRATCH_LR = 3e-4
WARMUP_EPOCHS = 3
# Adam eps 1e-8 -> 1e-6: enlarges the denominator floor so a tiny second-moment
# estimate can't blow the update up. Cheap insurance, safe for both paths.
ADAM_EPS = 1e-6
WEIGHT_DECAY = 1e-4
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
DROPOUT = 0.2
VAL_FRACTION = 0.2


def _arch_env():
    """Capacity-experiment overrides: WP_TCN_CHANNELS (comma-separated ints, e.g.
    '128,128,256,256,512,512') and WP_CARD_EMBED (int). Returns (tcn_channels or
    None, card_embed_dim). Applies to the from-scratch path only."""
    ch = os.environ.get("WP_TCN_CHANNELS")
    tcn = [int(x) for x in ch.split(",")] if ch else None
    emb = int(os.environ.get("WP_CARD_EMBED", "16"))
    return tcn, emb


def _detect_device() -> torch.device:
    """Detect the best available device: XPU → CUDA → CPU.

    WP_DEVICE overrides the search. The reason it exists: this box has ONE
    small-BAR A770 that cannot hold two GPU jobs -- that collision killed a
    training run -- so while a train is live the 5-minute inference cron has to
    be paused, and P(win) curves go stale for as long as training takes. The
    WP model is 3.8M parameters and inference is a handful of forward passes
    per game, so CPU is entirely adequate for keeping the dashboard current;
    it is only training that needs the GPU.
    """
    forced = os.environ.get("WP_DEVICE", "").strip().lower()
    if forced:
        logger.info("WP_DEVICE=%s — overriding device detection", forced)
        return torch.device(forced)
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


class WPTrainer:
    """Trains the win probability model with per-tick BCE loss.

    Args:
        model: WinProbabilityModel instance.
        dataset: SequenceDataset.
        device: Torch device.
        model_dir: Directory to save the best model.
        class_weight: Optional weight for positive class (win) to handle imbalance.
    """

    def __init__(
        self,
        model: WinProbabilityModel,
        dataset,
        device: torch.device,
        model_dir: Path,
        class_weight: Optional[float] = None,
        amp: bool = True,
        batch_size: int = BATCH_SIZE,
        learning_rate: float = LEARNING_RATE,
        warmup_epochs: int = 0,
    ):
        self.model = model.to(device)
        self.device = device
        self.model_dir = model_dir
        # batch_size defaults to BATCH_SIZE (fine for the frozen-encoder base
        # model), but the from-scratch full-encoder train retains all TCN
        # activations for backward and blows the A770 Level-Zero allocation
        # ceiling at 4096 (OUT_OF_HOST_MEMORY, which poisons the context and
        # surfaces as a NaN loss). Callers drop it to 512 for from-scratch.
        self.batch_size = batch_size
        # bf16 autocast is safe when transferring from a pretrained encoder
        # (moderate logits) but a from-scratch random init produces extreme
        # early logits that overflow bf16 -> NaN loss at epoch 1. Callers train
        # from scratch in fp32.
        self.amp = amp

        # Train/val split — last 20% as validation (ordered by battle_time)
        n = len(dataset)
        n_val = int(n * VAL_FRACTION)
        n_train = n - n_val

        train_indices = list(range(n_train))
        val_indices = list(range(n_train, n))

        # Loader dispatch: memmap shards (firehose) > lazy > in-memory DataLoader
        from tracker.ml.lazy_dataset import LazySequenceDataset, LazyBatchLoader
        from tracker.ml.wp_shard_cache import ShardDataset, ShardBatchLoader
        if isinstance(dataset, ShardDataset):
            # Vectorized gathers from memory-mapped shards — no per-batch Python
            # padding, no collate bottleneck (the fix for CPU-bound, XPU-idle
            # training at 1.7M games).
            self.train_loader = ShardBatchLoader(
                dataset, train_indices, batch_size=self.batch_size, shuffle=True)
            self.val_loader = ShardBatchLoader(
                dataset, val_indices, batch_size=self.batch_size, shuffle=False)
            self.full_loader = ShardBatchLoader(
                dataset, list(range(n)), batch_size=self.batch_size, shuffle=False)
        elif isinstance(dataset, LazySequenceDataset):
            self.train_loader = LazyBatchLoader(
                dataset, train_indices, batch_size=self.batch_size,
                shuffle=True, collate_fn=wp_collate_fn,
            )
            self.val_loader = LazyBatchLoader(
                dataset, val_indices, batch_size=self.batch_size,
                shuffle=False, collate_fn=wp_collate_fn,
            )
            self.full_loader = LazyBatchLoader(
                dataset, list(range(n)), batch_size=self.batch_size,
                shuffle=False, collate_fn=wp_collate_fn,
            )
        else:
            self.train_loader = DataLoader(
                Subset(dataset, train_indices),
                batch_size=self.batch_size,
                shuffle=True,
                collate_fn=wp_collate_fn,
                num_workers=0,
            )
            self.val_loader = DataLoader(
                Subset(dataset, val_indices),
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=wp_collate_fn,
                num_workers=0,
            )
            self.full_loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=wp_collate_fn,
                num_workers=0,
            )

        # Class-weighted BCE for imbalanced win/loss
        pos_weight = torch.tensor([class_weight], device=device) if class_weight else None
        self.criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)

        # Only optimize parameters that require gradients (frozen encoder excluded)
        trainable = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = AdamW(trainable, lr=learning_rate,
                               weight_decay=WEIGHT_DECAY, eps=ADAM_EPS)
        # Linear warmup for the first `warmup_epochs` (from-scratch stability),
        # then cosine anneal over the remainder. warmup_epochs=0 -> plain cosine
        # (unchanged transfer path).
        if warmup_epochs > 0:
            warm = LinearLR(self.optimizer, start_factor=0.1,
                            total_iters=warmup_epochs)
            cos = CosineAnnealingLR(self.optimizer,
                                    T_max=max(1, EPOCHS - warmup_epochs))
            self.scheduler = SequentialLR(self.optimizer, [warm, cos],
                                          milestones=[warmup_epochs])
        else:
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=EPOCHS)

        n_trainable = sum(p.numel() for p in trainable)
        n_total = sum(p.numel() for p in model.parameters())
        logger.info(
            "Train/val split: %d / %d games | %s/%s params trainable",
            n_train, n_val, f"{n_trainable:,}", f"{n_total:,}",
        )

    def train(self, checkpoint_path: Optional[Path] = None) -> Path:
        """Run training loop with early stopping.

        Args:
            checkpoint_path: Override path for saving checkpoint.
                If None, uses model_dir / "wp_v1.pt" for backwards compat.

        Returns:
            Path to saved best model checkpoint.
        """
        self.model_dir.mkdir(parents=True, exist_ok=True)
        best_path = checkpoint_path or (self.model_dir / "wp_v1.pt")

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, EPOCHS + 1):
            t0 = time.time()

            # Training phase
            self.model.train()
            train_loss = 0.0
            train_ticks = 0

            for card_ids, features, lengths, labels, mask, deck_ids, deck_vars in self.train_loader:
                card_ids = card_ids.to(self.device)
                features = features.to(self.device)
                lengths = lengths.to(self.device)
                labels = labels.to(self.device)
                mask = mask.to(self.device)
                deck_ids = deck_ids.to(self.device)
                deck_vars = deck_vars.to(self.device)

                self.optimizer.zero_grad()
                _amp = (torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
                        if self.amp else contextlib.nullcontext())
                with _amp:
                    logits = self.model(card_ids, features, lengths, deck_ids, deck_vars)
                    loss_per_tick = self.criterion(logits, labels)
                    loss = (loss_per_tick * mask).sum() / mask.sum().clamp(min=1)

                # Non-finite loss (occasional, e.g. a rare long-sequence batch
                # overflowing the from-scratch TCN in fp32): the check is BEFORE
                # backward, so the weights are still clean — SKIP this batch
                # rather than aborting the whole run. A run-wide skip counter
                # guards against a systematic problem (abort if it never trains).
                if not torch.isfinite(loss):
                    self._nan_skips = getattr(self, "_nan_skips", 0) + 1
                    self._nan_run_skips = getattr(self, "_nan_run_skips", 0) + 1
                    if self._nan_run_skips <= 5 or self._nan_run_skips % 50 == 0:
                        logger.warning("Non-finite loss at epoch %d — skipping batch "
                                       "(run skips: %d)", epoch, self._nan_run_skips)
                    if self._nan_run_skips > 2000:
                        raise RuntimeError(
                            f"Too many non-finite batches ({self._nan_run_skips}) — "
                            "training is not converging. Aborting."
                        )
                    continue

                loss.backward()
                # Clip gradients to prevent the runaway-update path that
                # introduced NaN weights in v1/v2/v3 checkpoints. clip_grad_norm_
                # returns the pre-clip total norm; if it's non-finite the grads
                # would NaN-poison the weights (clip_coef becomes nan), so skip
                # the step instead — keeps weights clean across a bad batch.
                total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                if not torch.isfinite(total_norm):
                    self._nan_run_skips = getattr(self, "_nan_run_skips", 0) + 1
                    self.optimizer.zero_grad()
                    continue
                self.optimizer.step()

                train_loss += (loss_per_tick * mask).sum().item()
                train_ticks += mask.sum().item()

            self.scheduler.step()
            train_loss /= max(train_ticks, 1)

            # Validation phase
            val_loss, val_acc = self._evaluate(self.val_loader)

            elapsed = time.time() - t0
            logger.info(
                "Epoch %d/%d [%.1fs] — train loss: %.4f | val loss: %.4f acc: %.3f",
                epoch, EPOCHS, elapsed, train_loss, val_loss, val_acc,
            )

            # Self-healing: if the epoch ended non-finite (val_loss NaN/Inf or
            # any live weight non-finite), the model has been corrupted mid-run.
            # Rather than burn the remaining epochs emitting NaN (what wasted
            # wp_v4's epochs 11-20), roll back to the last clean checkpoint, drop
            # the corrupted Adam moments, and permanently halve the LR to escape
            # the instability. Capped at 3 recoveries before giving up.
            weights_finite = all(
                torch.isfinite(v).all() for v in self.model.state_dict().values()
                if torch.is_tensor(v) and v.dtype.is_floating_point
            )
            if not math.isfinite(val_loss) or not weights_finite:
                self._recoveries = getattr(self, "_recoveries", 0) + 1
                if best_path.exists() and self._recoveries <= 3:
                    logger.warning(
                        "Epoch %d non-finite (val_loss=%s, weights_finite=%s) — "
                        "rollback to best + halve LR (recovery %d/3)",
                        epoch, val_loss, weights_finite, self._recoveries,
                    )
                    ckpt = torch.load(best_path, map_location=self.device,
                                      weights_only=True)
                    self.model.load_state_dict(ckpt["model_state_dict"])
                    self.optimizer.state.clear()  # drop corrupted Adam m/v
                    for sub in getattr(self.scheduler, "_schedulers", [self.scheduler]):
                        sub.base_lrs = [b * 0.5 for b in sub.base_lrs]
                    patience_counter += 1
                    if patience_counter >= EARLY_STOPPING_PATIENCE:
                        logger.info("Early stopping at epoch %d (post-recovery)", epoch)
                        break
                    continue
                raise RuntimeError(
                    f"Non-finite training state at epoch {epoch} "
                    f"(recoveries={self._recoveries}, best exists={best_path.exists()})"
                )

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                state_dict = self.model.state_dict()
                # Refuse to persist a checkpoint with NaN/Inf weights — that's
                # the bug that bit us across v1/v2/v3 (see commit 8c160e0).
                bad = [k for k, v in state_dict.items()
                       if torch.is_tensor(v) and v.dtype.is_floating_point
                       and not torch.isfinite(v).all()]
                if bad:
                    raise RuntimeError(
                        f"Refusing to save checkpoint with non-finite weights in "
                        f"{len(bad)} tensors (first few: {bad[:5]}). "
                        "Loss converged to a corrupt state."
                    )
                torch.save({
                    "model_state_dict": state_dict,
                    "vocab_size": self.model.card_embedding.num_embeddings,
                    "feature_dim": self.model.feature_dim,
                    "extra_feature_dim": getattr(self.model, "extra_feature_dim", 0),
                    "tcn_channels": getattr(self.model, "tcn_channels", None),
                    "card_embed_dim": getattr(self.model, "card_embed_dim", 16),
                    "deck_features": getattr(self.model, "deck_features", False),
                    "deck_interaction": getattr(self.model, "deck_interaction", False),
                    "deck_antisym": getattr(self.model, "deck_antisym", False),
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

    def _evaluate(
        self, loader: DataLoader,
    ) -> tuple[float, float]:
        """Evaluate on a DataLoader, returning (loss, last-tick accuracy)."""
        self.model.eval()
        total_loss = 0.0
        total_ticks = 0
        correct = 0
        total_games = 0

        with torch.no_grad():
            for card_ids, features, lengths, labels, mask, deck_ids, deck_vars in loader:
                card_ids = card_ids.to(self.device)
                features = features.to(self.device)
                lengths = lengths.to(self.device)
                labels = labels.to(self.device)
                mask = mask.to(self.device)
                deck_ids = deck_ids.to(self.device)
                deck_vars = deck_vars.to(self.device)

                _amp = (torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
                        if self.amp else contextlib.nullcontext())
                with _amp:
                    logits = self.model(card_ids, features, lengths, deck_ids, deck_vars)
                    loss_per_tick = self.criterion(logits, labels)
                total_loss += (loss_per_tick * mask).sum().item()
                total_ticks += mask.sum().item()

                # Last-tick accuracy (final prediction vs result)
                batch_size = logits.size(0)
                last_indices = (lengths - 1).clamp(min=0).long()
                last_logits = logits[torch.arange(batch_size, device=self.device), last_indices]
                last_labels = labels[:, 0]  # all ticks have same label
                preds = (last_logits > 0).float()
                correct += (preds == last_labels).sum().item()
                total_games += batch_size

        return total_loss / max(total_ticks, 1), correct / max(total_games, 1)

    @torch.no_grad()
    def collect_val_logits(self) -> tuple[np.ndarray, np.ndarray]:
        """Collect last-tick logits and labels from validation set for calibration.

        Returns:
            Tuple of (logits, labels) arrays, each shape (N,).
        """
        self.model.eval()
        all_logits: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []

        for card_ids, features, lengths, labels, mask, deck_ids, deck_vars in self.val_loader:
            card_ids = card_ids.to(self.device)
            features = features.to(self.device)
            lengths = lengths.to(self.device)
            deck_ids = deck_ids.to(self.device)
            deck_vars = deck_vars.to(self.device)

            logits = self.model(card_ids, features, lengths, deck_ids, deck_vars)

            batch_size = logits.size(0)
            last_indices = (lengths - 1).clamp(min=0).long()
            last_logits = logits[
                torch.arange(batch_size, device=self.device), last_indices
            ]
            last_labels = labels[:, 0]

            all_logits.append(last_logits.cpu().numpy())
            all_labels.append(last_labels.numpy())

        return np.concatenate(all_logits), np.concatenate(all_labels)

    @torch.no_grad()
    def run_inference(
        self,
        session: Session,
        dataset: SequenceDataset,
        battle_ids: list[str],
        vocab: CardVocabulary,
        calibrator: Optional[PlattCalibrator] = None,
    ) -> int:
        """Run WP inference on all games and store results.

        Computes P(win) at each tick, WPA, criticality, and per-game
        summary statistics.

        Args:
            session: Database session.
            dataset: The full SequenceDataset.
            battle_ids: Battle IDs in dataset order.
            vocab: Card vocabulary for reverse lookups.
            calibrator: Optional Platt scaling calibrator for probability correction.

        Returns:
            Number of games processed.
        """
        from tracker.models import ReplayEvent

        self.model.eval()
        processed = 0

        # Load replay events grouped by battle_id for card name lookups
        # Process in batches matching the dataloader
        sample_idx = 0

        for card_ids, features, lengths, labels, mask, deck_ids, deck_vars in self.full_loader:
            card_ids = card_ids.to(self.device)
            features = features.to(self.device)
            lengths = lengths.to(self.device)
            deck_ids = deck_ids.to(self.device)
            deck_vars = deck_vars.to(self.device)

            logits = self.model(card_ids, features, lengths, deck_ids, deck_vars)  # (batch, seq_len)
            logits_np = logits.cpu().numpy()
            if calibrator is not None and calibrator.fitted:
                probs = calibrator.calibrate_logits(logits_np)
            else:
                probs = 1.0 / (1.0 + np.exp(-logits_np))  # sigmoid
            lengths_np = lengths.cpu().numpy()

            batch_size = card_ids.size(0)
            card_ids_np = card_ids.cpu().numpy()

            for i in range(batch_size):
                if sample_idx >= len(battle_ids):
                    break

                bid = battle_ids[sample_idx]
                seq_len = int(lengths_np[i])
                wp_curve = probs[i, :seq_len]
                sample_idx += 1

                # Get card names for this game's events
                events = session.execute(
                    sa_text("""
                        SELECT card_name, game_tick
                        FROM replay_events
                        WHERE battle_id = :bid AND card_name != '_invalid'
                        ORDER BY game_tick
                    """),
                    {"bid": bid},
                ).all()

                # Compute WPA
                wpa = np.zeros(seq_len)
                wpa[0] = wp_curve[0] - 0.5  # delta from prior (0.5 = neutral)
                wpa[1:] = np.diff(wp_curve)
                criticality = np.abs(wpa)

                # Store per-tick records — dialect-aware upsert so re-runs
                # never hit UNIQUE constraint violations.
                dialect = session.bind.dialect.name
                if dialect == "sqlite":
                    _upsert_sql = sa_text("""
                        INSERT OR REPLACE INTO win_probability
                            (battle_id, game_tick, win_prob, wpa, criticality, event_index, model_version)
                        VALUES (:bid, :tick, :wp, :wpa, :crit, :eidx, :ver)
                    """)
                else:
                    # PostgreSQL (and any other dialect with ON CONFLICT support)
                    _upsert_sql = sa_text("""
                        INSERT INTO win_probability
                            (battle_id, game_tick, win_prob, wpa, criticality, event_index, model_version)
                        VALUES (:bid, :tick, :wp, :wpa, :crit, :eidx, :ver)
                        ON CONFLICT (battle_id, game_tick, model_version) DO UPDATE SET
                            win_prob = EXCLUDED.win_prob, wpa = EXCLUDED.wpa,
                            criticality = EXCLUDED.criticality, event_index = EXCLUDED.event_index
                    """)
                for j in range(seq_len):
                    game_tick = events[j][1] if j < len(events) else j
                    session.execute(_upsert_sql, {
                        "bid": bid, "tick": game_tick,
                        "wp": float(wp_curve[j]), "wpa": float(wpa[j]),
                        "crit": float(criticality[j]), "eidx": j,
                        "ver": WP_MODEL_VERSION,
                    })

                # Compute summary
                card_wpa: dict[str, float] = defaultdict(float)
                for j in range(seq_len):
                    if j < len(events):
                        card_name = events[j][0]
                        card_wpa[card_name] += float(wpa[j])

                top_pos = max(card_wpa, key=card_wpa.get) if card_wpa else None
                top_neg = min(card_wpa, key=card_wpa.get) if card_wpa else None

                crit_idx = int(np.argmax(criticality))
                crit_card = events[crit_idx][0] if crit_idx < len(events) else None
                crit_tick = events[crit_idx][1] if crit_idx < len(events) else crit_idx

                volatility = float(np.std(wpa)) if seq_len > 1 else 0.0

                session.merge(GameWPSummary(
                    battle_id=bid,
                    pre_game_wp=float(wp_curve[0]),
                    final_wp=float(wp_curve[-1]),
                    max_wp=float(np.max(wp_curve)),
                    min_wp=float(np.min(wp_curve)),
                    volatility=volatility,
                    top_positive_wpa_card=top_pos,
                    top_negative_wpa_card=top_neg,
                    critical_tick=crit_tick,
                    critical_card=crit_card,
                    model_version=WP_MODEL_VERSION,
                ))

                processed += 1
                if processed % 500 == 0:
                    session.flush()
                    logger.info("  Processed %d games", processed)

        session.commit()
        return processed


def infer_wp(session: Session, model_dir: Optional[Path] = None) -> None:
    """Inference-only: load existing checkpoint and store WP curves for all games.

    Skips training entirely. Requires wp_v1.pt to already exist.

    Args:
        session: Database session.
        model_dir: Directory containing wp_v1.pt.
    """
    if model_dir is None:
        model_dir = Path("data/ml_models")

    wp_path = _resolve_wp_path(session, model_dir)
    if not wp_path:
        print("  ✗ No trained WP model found. Run --train-wp first.")
        return

    device = _detect_device()
    vocab = CardVocabulary(session)

    checkpoint = torch.load(wp_path, map_location=device, weights_only=True)
    saved_vocab_size = checkpoint["vocab_size"]
    sd = checkpoint["model_state_dict"]
    nan_fixed = _sanitize_state_dict(sd)
    if nan_fixed:
        logger.warning("Sanitized %d NaN weights in %s state_dict", nan_fixed, wp_path.name)

    feat_dim = checkpoint.get("feature_dim", 17)
    extra_dim = checkpoint.get("extra_feature_dim", 0)
    model = WinProbabilityModel(vocab_size=saved_vocab_size, feature_dim=feat_dim,
                                extra_feature_dim=extra_dim, dropout=0.0,
                                tcn_channels=checkpoint.get("tcn_channels"),
                                card_embed_dim=checkpoint.get("card_embed_dim", 16),
                                deck_features=checkpoint.get("deck_features", False),
                                deck_interaction=checkpoint.get("deck_interaction", False),
                                deck_antisym=checkpoint.get("deck_antisym", False))
    model.load_state_dict(sd)
    model.to(device)
    model.eval()

    logger.info(
        "Loaded WP checkpoint %s epoch %d (val_loss=%.4f, val_acc=%.3f, feature_dim=%d)",
        wp_path.name, checkpoint["epoch"], checkpoint["val_loss"], checkpoint["val_acc"], feat_dim,
    )

    # Load calibrator — try versioned name first, then generic
    calibrator_stem = wp_path.stem  # e.g. "wp_v2"
    calibrator_path = model_dir / f"{calibrator_stem}_calibrator.json"
    if not calibrator_path.exists():
        calibrator_path = model_dir / "wp_calibrator.json"
    calibrator = None
    if calibrator_path.exists():
        calibrator = PlattCalibrator.load(calibrator_path)
        print(f"  → Platt calibration loaded (a={calibrator.a:.4f}, b={calibrator.b:.4f})")
    else:
        print("  · No calibration file found — using raw sigmoid probabilities")

    dataset = SequenceDataset(session, vocab, extra_features=(feat_dim + extra_dim) > 17)
    if not dataset:
        print("  ✗ No games with replay data found.")
        return

    # Derive battle_ids in dataset order using a JOIN (avoids slow correlated subquery)
    battle_ids = session.execute(
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
    battle_ids = list(battle_ids)[:len(dataset)]

    trainer = WPTrainer.__new__(WPTrainer)
    trainer.model = model
    trainer.device = device
    trainer.full_loader = DataLoader(
        dataset, batch_size=(512 if feat_dim > 17 else BATCH_SIZE), shuffle=False,
        collate_fn=wp_collate_fn, num_workers=0,
    )

    print(f"  → Running WP inference on {len(battle_ids)} games...")
    processed = trainer.run_inference(
        session, dataset, battle_ids, vocab, calibrator=calibrator,
    )
    print(f"  ✓ {processed} games processed with per-tick P(win) + WPA")
    if calibrator:
        print(f"  ✓ Platt-calibrated probabilities")
    print(f"  ✓ val_loss={checkpoint['val_loss']:.4f}, val_acc={checkpoint['val_acc']:.3f}")


def infer_wp_incremental(session: Session, model_dir: Optional[Path] = None) -> int:
    """Run WP inference only on games that have replay events but no WP data.

    Lightweight enough to run on every personal_combined cycle. Loads the
    model once and processes only the delta.

    Args:
        session: Database session.
        model_dir: Directory containing wp_v1.pt.

    Returns:
        Number of new games processed, or -1 if no model available.
    """
    if model_dir is None:
        model_dir = Path("data/ml_models")

    wp_path = _resolve_wp_path(session, model_dir)
    if not wp_path:
        return -1

    # Find games with replay events but no WP summary. With a watermark this
    # only examines battles whose replay events arrived since the last run
    # (id-range scan on the replay_events PK); without one (first run, or
    # after deleting the watermark file) it falls back to the full aggregate.
    watermark = _read_wp_watermark()
    new_watermark = session.execute(
        sa_text("SELECT COALESCE(MAX(id), 0) FROM replay_events")
    ).scalar() or 0

    if watermark is not None:
        # LATERAL keeps the event-count check as per-battle index probes;
        # an IN-subquery form here regresses to a merge join over the whole
        # replay_events table (measured: 68s vs 0.6s on live data).
        missing = session.execute(
            sa_text("""
                WITH new_battles AS (
                    SELECT DISTINCT battle_id FROM replay_events
                    WHERE id > :wm
                )
                SELECT b.battle_id
                FROM new_battles nb
                JOIN battles b ON b.battle_id = nb.battle_id
                LEFT JOIN game_wp_summary gws ON gws.battle_id = b.battle_id
                CROSS JOIN LATERAL (
                    SELECT COUNT(*) AS c FROM replay_events re
                    WHERE re.battle_id = nb.battle_id
                      AND re.card_name != '_invalid'
                ) rc
                WHERE b.battle_type = 'PvP' AND b.result IN ('win', 'loss')
                  AND gws.battle_id IS NULL
                  AND rc.c >= :min_events
                ORDER BY b.battle_time
            """),
            {"min_events": MIN_EVENTS,
             "wm": max(0, watermark - WP_WATERMARK_OVERLAP)},
        ).scalars().all()
    else:
        logger.info("No WP watermark — running full delta scan once")
        missing = session.execute(
            sa_text("""
                SELECT b.battle_id
                FROM battles b
                JOIN (
                    SELECT battle_id FROM replay_events
                    WHERE card_name != '_invalid'
                    GROUP BY battle_id HAVING COUNT(*) >= :min_events
                ) re_counts ON re_counts.battle_id = b.battle_id
                LEFT JOIN game_wp_summary gws ON gws.battle_id = b.battle_id
                WHERE b.battle_type = 'PvP' AND b.result IN ('win', 'loss')
                  AND gws.battle_id IS NULL
                ORDER BY b.battle_time
            """),
            {"min_events": MIN_EVENTS},
        ).scalars().all()

    if not missing:
        _write_wp_watermark(new_watermark)
        return 0

    device = _detect_device()
    vocab = CardVocabulary(session)

    checkpoint = torch.load(wp_path, map_location=device, weights_only=True)
    sd = checkpoint["model_state_dict"]
    nan_fixed = _sanitize_state_dict(sd)
    if nan_fixed:
        logger.warning("Sanitized %d NaN weights in %s state_dict", nan_fixed, wp_path.name)
    feat_dim = checkpoint.get("feature_dim", 17)
    extra_dim = checkpoint.get("extra_feature_dim", 0)
    model = WinProbabilityModel(vocab_size=checkpoint["vocab_size"], feature_dim=feat_dim,
                                extra_feature_dim=extra_dim, dropout=0.0,
                                tcn_channels=checkpoint.get("tcn_channels"),
                                card_embed_dim=checkpoint.get("card_embed_dim", 16),
                                deck_features=checkpoint.get("deck_features", False),
                                deck_interaction=checkpoint.get("deck_interaction", False),
                                deck_antisym=checkpoint.get("deck_antisym", False))
    model.load_state_dict(sd)
    model.to(device)
    model.eval()

    # Load calibrator — try versioned name first, then generic
    calibrator_stem = wp_path.stem
    calibrator_path = model_dir / f"{calibrator_stem}_calibrator.json"
    if not calibrator_path.exists():
        calibrator_path = model_dir / "wp_calibrator.json"
    calibrator = None
    if calibrator_path.exists():
        calibrator = PlattCalibrator.load(calibrator_path)

    # Build dataset for only the missing games
    missing_set = set(missing)
    dataset = SequenceDataset(session, vocab, battle_ids=missing, extra_features=(feat_dim + extra_dim) > 17)
    if not dataset:
        _write_wp_watermark(new_watermark)
        return 0

    battle_ids = list(missing)[:len(dataset)]

    trainer = WPTrainer.__new__(WPTrainer)
    trainer.model = model
    trainer.device = device
    trainer.full_loader = DataLoader(
        dataset, batch_size=(512 if feat_dim > 17 else BATCH_SIZE), shuffle=False,
        collate_fn=wp_collate_fn, num_workers=0,
    )

    processed = trainer.run_inference(
        session, dataset, battle_ids, vocab, calibrator=calibrator,
    )
    # Advance the watermark only after inference committed — a crash before
    # this point just means the next run re-examines the same id range.
    _write_wp_watermark(new_watermark)
    logger.info("Incremental WP inference: %d new games processed", processed)
    return processed


def train_wp(
    session: Session,
    model_dir: Optional[Path] = None,
    unfreeze_encoder: bool = False,
    auto_promote: bool = False,
    lazy: bool = False,
) -> None:
    """Full win probability pipeline: train, register, optionally promote + infer.

    Trains to a versioned checkpoint (wp_vN.pt), registers as a candidate
    in the model registry, and optionally promotes if accuracy improves
    over the current production model.

    Args:
        session: Database session.
        model_dir: Directory for model files.
        unfreeze_encoder: If True, fine-tune the full model including TCN encoder.
        auto_promote: If True, promote and run inference if accuracy improves.
    """
    import time as _time
    from tracker.ml.model_registry import (
        register_model, get_production, promote, next_version,
    )
    from tracker.models import Battle
    from sqlalchemy import func

    if model_dir is None:
        model_dir = Path("data/ml_models")

    t_start = _time.time()
    device = _detect_device()

    # 1. Build vocabulary
    vocab = CardVocabulary(session)
    logger.info("Vocabulary size: %d", vocab.size)

    # 2. Create dataset. WP_SHARD_DIR (pre-extracted memmap shards, see
    # wp_shard_cache.py) takes priority — no DB streaming, no collate bottleneck.
    _shard_dir = os.environ.get("WP_SHARD_DIR")
    if _shard_dir:
        from tracker.ml.wp_shard_cache import ShardDataset
        dataset = ShardDataset(_shard_dir)
        if bool(dataset.meta.get("extra_features")) != WP_EXTRA_FEATURES:
            raise RuntimeError(
                f"Shard extra_features={dataset.meta.get('extra_features')} does not "
                f"match WP_EXTRA_FEATURES={WP_EXTRA_FEATURES} — rebuild shards.")
        print(f"  → Shard dataset: {len(dataset)} games from {_shard_dir} "
              f"(L={dataset.max_len}, F={dataset.feature_dim}, memmap firehose)")
        lazy = False
    elif lazy and WP_EXTRA_FEATURES:
        logger.warning("Lazy dataset does not support extra_features yet — using in-memory SequenceDataset")
        lazy = False
    if _shard_dir:
        pass
    elif lazy:
        from tracker.ml.lazy_dataset import LazySequenceDataset
        db_url = os.environ.get("DATABASE_URL", str(session.bind.url))
        dataset = LazySequenceDataset(session, vocab, db_url=db_url)
        print(f"  → Lazy dataset: {len(dataset)} games (DB-backed, low memory)")
    else:
        dataset = SequenceDataset(session, vocab, extra_features=WP_EXTRA_FEATURES)
        print(f"  → Dataset feature_dim = {dataset.feature_dim}"
              f"{' (base+extra board-truth)' if WP_EXTRA_FEATURES else ''}")
    if len(dataset) < 50:
        logger.error("Need at least 50 games with replay data (have %d)", len(dataset))
        print(f"  ✗ Need at least 50 games with replay data (have {len(dataset)})")
        return

    # 3. Compute class weight for imbalanced data
    if _shard_dir:
        labels = dataset.labels[:]          # uint8 memmap -> in-RAM copy (N bytes)
        labels = [float(v) for v in labels]
    elif lazy:
        # Lazy dataset stores labels directly
        labels = dataset._labels
    else:
        labels = [s[2] for s in dataset._samples]
    n_wins = sum(labels)
    n_losses = len(labels) - n_wins
    if n_losses > 0 and n_wins > 0:
        class_weight = n_losses / n_wins
        logger.info("Class weight: %.3f (%.1f%% wins)", class_weight, 100 * n_wins / len(labels))
    else:
        class_weight = None

    # 4. Determine version number
    version = next_version(session, "wp")
    filename = f"wp_v{version}.pt"
    checkpoint_path = model_dir / filename
    print(f"  → Training WP v{version} ({filename})")

    # 5. Initialize model.
    #  - Base features only (17): transfer the pretrained encoder as before.
    #  - With board-truth features (>17): HEAD-INJECTION — keep the pretrained
    #    encoder (input width 16+17=33) frozen and concatenate the extra features
    #    onto the TCN output before the head. This keeps transfer learning (the
    #    from-scratch retrain lost ~2.7pts acc discarding the pretrained encoder)
    #    while still feeding the model the board-truth signal, and trains only the
    #    small head (fast, low-mem, no from-scratch NaN risk).
    feat_dim = getattr(dataset, "feature_dim", 17)
    prod = get_production(session, "wp")
    BASE_FEAT = 17
    # WP_FROM_SCRATCH=1 forces a full-encoder from-scratch train even for a wide
    # feature vector. Head-injection's frozen encoder is stale on the grown corpus
    # (wp_v6: 0.731 vs from-scratch wp_v5: 0.757), so from-scratch is currently the
    # better config until the TCN encoder itself is retrained on the full corpus.
    _force_scratch = os.environ.get("WP_FROM_SCRATCH") == "1"
    _head_inject = feat_dim > BASE_FEAT and not _force_scratch
    _built_from_scratch = False
    if _force_scratch and feat_dim != 17:
        _tcn_ch, _emb = _arch_env()
        logger.info("WP_FROM_SCRATCH=1 — full-encoder from-scratch (feature_dim=%d, "
                    "tcn_channels=%s, card_embed=%d)", feat_dim, _tcn_ch or "default", _emb)
        print(f"  → Training from scratch (feature_dim={feat_dim}, forced full encoder"
              f"{'' if _tcn_ch is None else f', tcn={_tcn_ch}, embed={_emb}'})")
        _deck = os.environ.get("WP_DECK_FEATURES", "1") == "1"
        _deck_ix = os.environ.get("WP_DECK_INTERACTION", "0") == "1"
        _deck_as = os.environ.get("WP_DECK_ANTISYM", "0") == "1"
        model = WinProbabilityModel(vocab_size=vocab.size, feature_dim=feat_dim, dropout=DROPOUT,
                                    tcn_channels=_tcn_ch, card_embed_dim=_emb,
                                    deck_features=_deck, deck_interaction=_deck_ix,
                                    deck_antisym=_deck_as)
        _built_from_scratch = True
        if _deck:
            logger.info("Deck prior ENABLED: both decks mean-pooled (card+variant "
                        "embeddings) and injected at the head, so P(win) has a "
                        "matchup prior at tick 0%s",
                        " — ANTISYMMETRIC (own-opp + skew-bilinear); mirror==0.5 by "
                        "construction" if _deck_as else
                        " — WITH own*opp interaction term" if _deck_ix else
                        " (additive only, no interaction)")

        # WP_RESUME=<path>: warm-start the weights from an earlier run of the
        # SAME architecture instead of starting cold. A long capacity run that
        # dies to an external cause (GPU contention, host reboot) otherwise
        # throws away days of training, since only the best weights are
        # checkpointed. Optimizer/scheduler state is NOT in the checkpoint, so
        # Adam and the LR schedule restart — this is a warm start, not an exact
        # resume; pair it with a reduced WP_LR to avoid knocking the loaded
        # weights out of the basin they already found.
        _resume = os.environ.get("WP_RESUME", "").strip()
        if _resume:
            if not Path(_resume).exists():
                raise FileNotFoundError(f"WP_RESUME checkpoint not found: {_resume}")
            _ck = torch.load(_resume, map_location="cpu", weights_only=True)
            _ck_tcn = _ck.get("tcn_channels")
            _ck_emb = _ck.get("card_embed_dim", 16)
            _want_tcn = _tcn_ch or model.tcn_channels
            if list(_ck_tcn or model.tcn_channels) != list(_want_tcn) or _ck_emb != _emb:
                raise ValueError(
                    f"WP_RESUME architecture mismatch: checkpoint tcn={_ck_tcn} "
                    f"embed={_ck_emb} vs requested tcn={_want_tcn} embed={_emb}"
                )
            if _ck.get("feature_dim") != feat_dim:
                raise ValueError(
                    f"WP_RESUME feature_dim mismatch: checkpoint "
                    f"{_ck.get('feature_dim')} vs current {feat_dim}"
                )
            # Shape-compatible transfer. Enabling the deck prior grows the head's
            # first conv (out_ch -> out_ch+deck_dim) and adds variant_embedding,
            # so a strict load would refuse a checkpoint whose encoder is exactly
            # what we want to keep. Everything that matches is taken; the rest is
            # initialised fresh and named in the log so a silent partial load can
            # never be mistaken for a full one.
            _src_sd = _ck["model_state_dict"]
            _own_sd = model.state_dict()
            _take = {k: v for k, v in _src_sd.items()
                     if k in _own_sd and _own_sd[k].shape == v.shape}
            _skip = sorted(set(_own_sd) - set(_take))
            model.load_state_dict(_take, strict=False)
            if _skip:
                logger.info("WP_RESUME transferred %d/%d tensors; freshly initialised: %s",
                            len(_take), len(_own_sd), ", ".join(_skip))
                print(f"  → Warm start transferred {len(_take)}/{len(_own_sd)} tensors; "
                      f"fresh: {', '.join(_skip)}")
            logger.info("WP_RESUME — warm-started from %s (epoch %s, val_loss %.4f, "
                        "val_acc %.4f); optimizer/LR schedule restart",
                        _resume, _ck.get("epoch"), _ck.get("val_loss", float("nan")),
                        _ck.get("val_acc", float("nan")))
            print(f"  → Warm start from {_resume} "
                  f"(epoch {_ck.get('epoch')}, val_loss {_ck.get('val_loss'):.4f})")
    elif _head_inject:
        extra = feat_dim - BASE_FEAT
        src = None
        if prod and (model_dir / prod.filename).exists():
            src = str(model_dir / prod.filename)
        elif (model_dir / "tcn_v1.pt").exists():
            src = str(model_dir / "tcn_v1.pt")
        else:
            existing = _resolve_wp_path(session, model_dir)
            src = str(existing) if existing else None
        if src:
            logger.info("Head-injection: frozen encoder from %s + %d board-truth feats at head",
                        src, extra)
            print(f"  → Head-injection (frozen encoder from {Path(src).name} "
                  f"+ {extra} board-truth feats at head)")
            model = WinProbabilityModel.from_pretrained_tcn(
                src, vocab.size, device, freeze_encoder=not unfreeze_encoder,
                dropout=DROPOUT, extra_feature_dim=extra,
            )
        else:
            logger.warning("No pretrained encoder found — head-injection from scratch")
            print("  → Head-injection from scratch (no pretrained encoder)")
            model = WinProbabilityModel(vocab_size=vocab.size, feature_dim=BASE_FEAT,
                                        extra_feature_dim=extra, dropout=DROPOUT)
            _built_from_scratch = True
    elif prod and (model_dir / prod.filename).exists():
        logger.info("Transfer learning from production %s", prod.filename)
        print(f"  → Transfer learning from production {prod.filename}")
        model = WinProbabilityModel.from_pretrained_tcn(
            str(model_dir / prod.filename), vocab.size, device,
            freeze_encoder=not unfreeze_encoder, dropout=DROPOUT,
        )
    else:
        tcn_path = model_dir / "tcn_v1.pt"
        freeze = not unfreeze_encoder
        if tcn_path.exists():
            logger.info("Loading pretrained TCN encoder from %s", tcn_path)
            print("  → Transfer learning from ADR-003 TCN encoder")
            model = WinProbabilityModel.from_pretrained_tcn(
                str(tcn_path), vocab.size, device,
                freeze_encoder=freeze, dropout=DROPOUT,
            )
        else:
            # Fallback: try any existing wp checkpoint
            existing = _resolve_wp_path(session, model_dir)
            if existing:
                logger.info("Transfer learning from %s", existing)
                print(f"  → Transfer learning from {existing.name}")
                model = WinProbabilityModel.from_pretrained_tcn(
                    str(existing), vocab.size, device,
                    freeze_encoder=freeze, dropout=DROPOUT,
                )
            else:
                logger.info("No pretrained model found — training from scratch")
                print("  → Training from scratch")
                model = WinProbabilityModel(vocab_size=vocab.size, dropout=DROPOUT)

    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 6. Train
    # From-scratch training (feature_dim != 17) trains the full encoder, so it
    # runs in fp32 (bf16 NaN-poisons a random init) and at batch 512 (the A770
    # can't hold batch-4096 full-encoder activations — OUT_OF_HOST_MEMORY).
    # Only a genuine full-encoder-from-scratch build needs the slow, stabilized
    # path (fp32, small batch, warmup). Head-injection with a frozen pretrained
    # encoder trains only the small head → fast path (bf16, batch 4096), same as
    # the original transfer training.
    _from_scratch = _built_from_scratch
    # From-scratch batch defaults to 512, but batch 1024/2048 is now usable once
    # the 4GB single-allocation ceiling is lifted (env: UR_L0_*RELAXED_ALLOCATION*
    # + IGC_ExtraOCLOptions=-cl-intel-greater-than-4GB-buffer-required). Bigger
    # batches amortize XPU kernel-dispatch overhead — measured ~3.5x throughput at
    # 2048 — so WP_BATCH_SIZE overrides it. Scale LR with batch (WP_LR).
    _fs_batch = int(os.environ.get("WP_BATCH_SIZE", "512")) if _from_scratch else BATCH_SIZE
    _fs_lr = float(os.environ.get("WP_LR", str(FROM_SCRATCH_LR))) if _from_scratch else LEARNING_RATE
    trainer = WPTrainer(model, dataset, device, model_dir,
                        class_weight=class_weight,
                        amp=not _from_scratch,
                        batch_size=_fs_batch,
                        learning_rate=_fs_lr,
                        warmup_epochs=WARMUP_EPOCHS if _from_scratch else 0)
    best_path = trainer.train(checkpoint_path=checkpoint_path)

    # 7. Load best model
    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    sd = checkpoint["model_state_dict"]
    nan_fixed = _sanitize_state_dict(sd)
    if nan_fixed:
        logger.warning("Sanitized %d NaN weights in %s state_dict", nan_fixed, best_path.name)
    model.load_state_dict(sd)
    model.to(device)
    trainer.model = model

    # 8. Fit Platt scaling calibration
    val_logits, val_labels = trainer.collect_val_logits()
    calibrator = PlattCalibrator().fit(val_logits, val_labels)
    calibrator_path = model_dir / f"wp_v{version}_calibrator.json"
    calibrator.save(calibrator_path)

    wall_time = int(_time.time() - t_start)
    cutoff = session.query(func.max(Battle.battle_time)).scalar()

    # 9. Register in model registry
    mv = register_model(
        session,
        model_type="wp",
        filename=filename,
        status="candidate",
        epochs=EPOCHS,
        best_epoch=checkpoint["epoch"],
        training_games=len(dataset),
        training_cutoff=cutoff.isoformat() if cutoff else None,
        wall_time_seconds=wall_time,
        device=str(device),
        val_loss=checkpoint["val_loss"],
        val_accuracy=checkpoint["val_acc"],
        metrics_json={
            "platt_a": calibrator.a,
            "platt_b": calibrator.b,
            "calibrator_path": str(calibrator_path),
            "n_wins": int(n_wins),
            "n_losses": int(n_losses),
            "n_total_params": n_total,
            "n_trainable_params": n_trainable,
        },
    )

    # 10. Auto-promote if better than current production
    promoted = False
    if auto_promote and prod:
        if checkpoint["val_acc"] > (prod.val_accuracy or 0):
            delta = checkpoint["val_acc"] - (prod.val_accuracy or 0)
            mv.improvement_delta = delta
            mv.prev_version_id = prod.id
            promote(session, "wp", version)
            promoted = True
            print(f"  ✓ Promoted v{version} (acc {checkpoint['val_acc']:.3f} > "
                  f"v{prod.version} acc {prod.val_accuracy:.3f}, +{delta:.3f})")
        else:
            print(f"  · v{version} acc {checkpoint['val_acc']:.3f} <= "
                  f"v{prod.version} acc {prod.val_accuracy:.3f} — kept as candidate")
    elif auto_promote and not prod:
        # No production model — auto-promote the first one
        promote(session, "wp", version)
        promoted = True
        print(f"  ✓ Promoted v{version} (first model)")
    else:
        print(f"  → Registered as candidate v{version} — use --promote-model wp {version} to promote")

    session.commit()

    # 11. Run inference only if promoted
    if promoted:
        # Derive battle_ids
        battle_ids = session.execute(
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
        battle_ids = list(battle_ids)[:len(dataset)]

        # Clear old WP data
        session.execute(
            sa_text("DELETE FROM win_probability WHERE model_version = :v"),
            {"v": WP_MODEL_VERSION},
        )
        session.execute(
            sa_text("DELETE FROM game_wp_summary WHERE model_version = :v"),
            {"v": WP_MODEL_VERSION},
        )
        session.commit()

        print(f"  → Running WP inference on {len(battle_ids)} games...")
        processed = trainer.run_inference(
            session, dataset, battle_ids, vocab, calibrator=calibrator,
        )
        print(f"  ✓ {processed} games processed with per-tick P(win) + WPA")

    print(f"  ✓ WP v{version}: val_loss={checkpoint['val_loss']:.4f}, "
          f"val_acc={checkpoint['val_acc']:.3f}, {len(dataset)} games, "
          f"{wall_time}s on {device}")
    print(f"  ✓ Saved to {best_path}")
