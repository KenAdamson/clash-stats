"""Win probability training loop (ADR-004).

Handles training the causal TCN win probability model with per-tick
BCE loss, optional transfer learning from ADR-003, Platt scaling
calibration, and per-game WPA inference + storage.
"""

import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from sqlalchemy import text as sa_text
from sqlalchemy.orm import Session

from tracker.ml.card_metadata import CardVocabulary, kebab_to_title
from tracker.ml.sequence_dataset import SequenceDataset, MIN_EVENTS
from tracker.ml.wp_dataset import wp_collate_fn
from tracker.ml.win_probability import WinProbabilityModel
from tracker.ml.wp_storage import WinProbability, GameWPSummary

logger = logging.getLogger(__name__)

WP_MODEL_VERSION = "wp-v1"

# Training hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10
DROPOUT = 0.2
VAL_FRACTION = 0.2


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
        dataset: SequenceDataset,
        device: torch.device,
        model_dir: Path,
        class_weight: Optional[float] = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.model_dir = model_dir

        # Train/val split — last 20% as validation (ordered by battle_time)
        n = len(dataset)
        n_val = int(n * VAL_FRACTION)
        n_train = n - n_val

        train_indices = list(range(n_train))
        val_indices = list(range(n_train, n))

        self.train_loader = DataLoader(
            Subset(dataset, train_indices),
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=wp_collate_fn,
            num_workers=0,
        )
        self.val_loader = DataLoader(
            Subset(dataset, val_indices),
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=wp_collate_fn,
            num_workers=0,
        )
        self.full_loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=wp_collate_fn,
            num_workers=0,
        )

        # Class-weighted BCE for imbalanced win/loss
        pos_weight = torch.tensor([class_weight], device=device) if class_weight else None
        self.criterion = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)

        # Only optimize parameters that require gradients (frozen encoder excluded)
        trainable = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = AdamW(trainable, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=EPOCHS)

        n_trainable = sum(p.numel() for p in trainable)
        n_total = sum(p.numel() for p in model.parameters())
        logger.info(
            "Train/val split: %d / %d games | %s/%s params trainable",
            n_train, n_val, f"{n_trainable:,}", f"{n_total:,}",
        )

    def train(self) -> Path:
        """Run training loop with early stopping.

        Returns:
            Path to saved best model checkpoint.
        """
        self.model_dir.mkdir(parents=True, exist_ok=True)
        best_path = self.model_dir / "wp_v1.pt"

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, EPOCHS + 1):
            t0 = time.time()

            # Training phase
            self.model.train()
            train_loss = 0.0
            train_ticks = 0

            for card_ids, features, lengths, labels, mask in self.train_loader:
                card_ids = card_ids.to(self.device)
                features = features.to(self.device)
                lengths = lengths.to(self.device)
                labels = labels.to(self.device)
                mask = mask.to(self.device)

                self.optimizer.zero_grad()
                logits = self.model(card_ids, features, lengths)  # (batch, seq_len)

                # Masked per-tick BCE
                loss_per_tick = self.criterion(logits, labels)  # (batch, seq_len)
                loss = (loss_per_tick * mask).sum() / mask.sum().clamp(min=1)
                loss.backward()
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
            for card_ids, features, lengths, labels, mask in loader:
                card_ids = card_ids.to(self.device)
                features = features.to(self.device)
                lengths = lengths.to(self.device)
                labels = labels.to(self.device)
                mask = mask.to(self.device)

                logits = self.model(card_ids, features, lengths)
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
    def run_inference(
        self,
        session: Session,
        dataset: SequenceDataset,
        battle_ids: list[str],
        vocab: CardVocabulary,
    ) -> int:
        """Run WP inference on all games and store results.

        Computes P(win) at each tick, WPA, criticality, and per-game
        summary statistics.

        Args:
            session: Database session.
            dataset: The full SequenceDataset.
            battle_ids: Battle IDs in dataset order.
            vocab: Card vocabulary for reverse lookups.

        Returns:
            Number of games processed.
        """
        from tracker.models import ReplayEvent

        self.model.eval()
        processed = 0

        # Load replay events grouped by battle_id for card name lookups
        # Process in batches matching the dataloader
        sample_idx = 0

        for card_ids, features, lengths, labels, mask in self.full_loader:
            card_ids = card_ids.to(self.device)
            features = features.to(self.device)
            lengths = lengths.to(self.device)

            logits = self.model(card_ids, features, lengths)  # (batch, seq_len)
            probs = torch.sigmoid(logits).cpu().numpy()  # (batch, seq_len)
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

                # Store per-tick records
                for j in range(seq_len):
                    game_tick = events[j][1] if j < len(events) else j
                    session.add(WinProbability(
                        battle_id=bid,
                        game_tick=game_tick,
                        win_prob=float(wp_curve[j]),
                        wpa=float(wpa[j]),
                        criticality=float(criticality[j]),
                        event_index=j,
                        model_version=WP_MODEL_VERSION,
                    ))

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

    wp_path = model_dir / "wp_v1.pt"
    if not wp_path.exists():
        print("  ✗ No trained WP model found. Run --train-wp first.")
        return

    device = _detect_device()
    vocab = CardVocabulary(session)

    checkpoint = torch.load(wp_path, map_location=device, weights_only=True)
    saved_vocab_size = checkpoint["vocab_size"]

    model = WinProbabilityModel(vocab_size=saved_vocab_size, dropout=0.0)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info(
        "Loaded WP checkpoint from epoch %d (val_loss=%.4f, val_acc=%.3f)",
        checkpoint["epoch"], checkpoint["val_loss"], checkpoint["val_acc"],
    )

    dataset = SequenceDataset(session, vocab)
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

    # Clear old data
    session.execute(sa_text("DELETE FROM win_probability WHERE model_version = :v"), {"v": WP_MODEL_VERSION})
    session.execute(sa_text("DELETE FROM game_wp_summary WHERE model_version = :v"), {"v": WP_MODEL_VERSION})
    session.commit()

    trainer = WPTrainer.__new__(WPTrainer)
    trainer.model = model
    trainer.device = device
    trainer.full_loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=wp_collate_fn, num_workers=0,
    )

    print(f"  → Running WP inference on {len(battle_ids)} games...")
    processed = trainer.run_inference(session, dataset, battle_ids, vocab)
    print(f"  ✓ {processed} games processed with per-tick P(win) + WPA")
    print(f"  ✓ val_loss={checkpoint['val_loss']:.4f}, val_acc={checkpoint['val_acc']:.3f}")


def train_wp(session: Session, model_dir: Optional[Path] = None) -> None:
    """Full win probability pipeline: train, infer, store.

    Args:
        session: Database session.
        model_dir: Directory for model files.
    """
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

    # 3. Compute class weight for imbalanced data
    labels = [s[2] for s in dataset._samples]
    n_wins = sum(labels)
    n_losses = len(labels) - n_wins
    if n_losses > 0 and n_wins > 0:
        class_weight = n_losses / n_wins
        logger.info("Class weight: %.3f (%.1f%% wins)", class_weight, 100 * n_wins / len(labels))
    else:
        class_weight = None

    # 4. Initialize model — try transfer learning from ADR-003 TCN
    tcn_path = model_dir / "tcn_v1.pt"
    if tcn_path.exists():
        logger.info("Loading pretrained TCN encoder from %s", tcn_path)
        print("  → Transfer learning from ADR-003 TCN encoder")
        model = WinProbabilityModel.from_pretrained_tcn(
            str(tcn_path), vocab.size, device,
            freeze_encoder=True, dropout=DROPOUT,
        )
    else:
        logger.info("No pretrained TCN found — training from scratch")
        print("  → Training from scratch (no TCN checkpoint)")
        model = WinProbabilityModel(vocab_size=vocab.size, dropout=DROPOUT)

    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model: %s total params, %s trainable", f"{n_total:,}", f"{n_trainable:,}")

    # 5. Train
    trainer = WPTrainer(model, dataset, device, model_dir, class_weight=class_weight)
    best_path = trainer.train()

    # 6. Load best model for inference
    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    trainer.model = model
    logger.info(
        "Loaded best model from epoch %d (val_loss=%.4f, val_acc=%.3f)",
        checkpoint["epoch"], checkpoint["val_loss"], checkpoint["val_acc"],
    )

    # 7. Derive battle_ids in dataset order using JOIN (avoids slow correlated subquery)
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

    # 8. Clear old WP data for this model version
    session.execute(
        sa_text("DELETE FROM win_probability WHERE model_version = :v"),
        {"v": WP_MODEL_VERSION},
    )
    session.execute(
        sa_text("DELETE FROM game_wp_summary WHERE model_version = :v"),
        {"v": WP_MODEL_VERSION},
    )
    session.commit()

    # 9. Run inference and store
    print(f"  → Running WP inference on {len(battle_ids)} games...")
    processed = trainer.run_inference(session, dataset, battle_ids, vocab)

    print(f"  ✓ Win probability training complete")
    print(f"  ✓ Model: {n_total:,} params ({n_trainable:,} trainable)")
    print(f"  ✓ Best val_loss={checkpoint['val_loss']:.4f}, val_acc={checkpoint['val_acc']:.3f}")
    print(f"  ✓ {processed} games processed with per-tick P(win) + WPA")
    print(f"  ✓ Saved to {best_path}")
