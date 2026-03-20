"""CVAE training loop with beta-VAE KL annealing (ADR-006).

Trains the CounterfactualVAE with a composite loss:
  L_card(CE) + 0.5*L_tick(MSE) + 0.5*L_pos(MSE) + 0.3*L_side(BCE) + beta*L_KL

KL annealing: beta linearly 0->1 over 20 epochs to prevent posterior collapse.
Transfer learning: TCN + card_embedding from wp_v1.pt, frozen first 10 epochs.
"""

import contextlib
import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset
from sqlalchemy import text as sa_text
from sqlalchemy.orm import Session

from tracker.ml.card_metadata import CardVocabulary
from tracker.ml.cvae import CounterfactualVAE
from tracker.ml.cvae_dataset import CVAEDataset, cvae_collate_fn
from tracker.ml.sequence_dataset import MIN_EVENTS
from tracker.models import Battle

logger = logging.getLogger(__name__)

CVAE_MODEL_VERSION = "cvae-v1"

# Training hyperparameters
BATCH_SIZE = 256
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 150
EARLY_STOPPING_PATIENCE = 15
DROPOUT = 0.2
VAL_FRACTION = 0.2

# KL annealing
KL_ANNEAL_EPOCHS = 20

# Loss weights
WEIGHT_TICK = 0.5
WEIGHT_POS = 0.5
WEIGHT_SIDE = 0.3

# Encoder freeze schedule
FREEZE_EPOCHS = 10


def _detect_device() -> torch.device:
    """Detect the best available device: XPU -> CUDA -> CPU.

    XPU requires intel-opencl-icd for oneDNN's SDPA primitive. Without it,
    Transformer attention fails with "could not create a primitive".
    On XPU, IPEX_FP32_MATH_MODE=FP32 prevents NaN from bfloat16 auto-promotion.
    """
    # XPU disabled for training — oneDNN accumulates numerical errors over
    # ~100+ optimizer steps, producing NaN regardless of shuffle/validation/
    # grad clip settings. CUDA and CPU are stable. XPU inference works fine.
    # TODO: re-evaluate when T4 arrives for CUDA baseline comparison.
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA: %s", torch.cuda.get_device_name(0))
        return device
    logger.info("Using CPU")
    return torch.device("cpu")


def _autocast_ctx(device: torch.device):
    """Return autocast context for CUDA, no-op for XPU/CPU."""
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()




class CVAETrainer:
    """Trains the CounterfactualVAE with KL annealing and early stopping.

    Args:
        model: CounterfactualVAE instance.
        dataset: CVAEDataset.
        device: Torch device.
        model_dir: Directory to save the best model.
    """

    def __init__(
        self,
        model: CounterfactualVAE,
        dataset: CVAEDataset,
        device: torch.device,
        model_dir: Path,
        training_cutoff: str | None = None,
    ):
        self.model = model.to(device)
        self.device = device
        self.model_dir = model_dir
        self.training_cutoff = training_cutoff

        # Train/val split
        n = len(dataset)
        n_val = int(n * VAL_FRACTION)
        n_train = n - n_val

        train_indices = list(range(n_train))
        val_indices = list(range(n_train, n))

        self.train_loader = DataLoader(
            Subset(dataset, train_indices),
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=cvae_collate_fn,
            num_workers=6,
        )
        self.val_loader = DataLoader(
            Subset(dataset, val_indices),
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=cvae_collate_fn,
            num_workers=6,
        )

        self.card_criterion = nn.CrossEntropyLoss(
            ignore_index=0, reduction="none",
        )

        # Optimizer: all trainable params
        trainable = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = AdamW(trainable, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=EPOCHS)

        n_trainable = sum(p.numel() for p in trainable)
        n_total = sum(p.numel() for p in model.parameters())
        logger.info(
            "CVAE train/val: %d / %d games | %s/%s params trainable",
            n_train, n_val, f"{n_trainable:,}", f"{n_total:,}",
        )

    def _compute_loss(
        self,
        outputs: dict[str, torch.Tensor],
        target_card_ids: torch.Tensor,
        target_features: torch.Tensor,
        mask: torch.Tensor,
        beta: float,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute CVAE composite loss.

        Args:
            outputs: Decoder outputs + mu + logvar.
            target_card_ids: (batch, seq_len) ground truth card indices.
            target_features: (batch, seq_len, 17) ground truth features.
            mask: (batch, seq_len) padding mask.
            beta: KL weight.

        Returns:
            (total_loss, loss_dict) for logging.
        """
        batch_size, seq_len = target_card_ids.shape

        # Card loss (CE)
        card_logits = outputs["card_logits"]  # (batch, seq, vocab)
        card_loss_per_tok = self.card_criterion(
            card_logits.reshape(-1, card_logits.size(-1)),
            target_card_ids.reshape(-1),
        ).reshape(batch_size, seq_len)
        card_loss = (card_loss_per_tok * mask).sum() / mask.sum().clamp(min=1)

        # Tick delta loss (MSE) — compare to normalized tick differences
        tick_norms = target_features[:, :, 1]  # game_tick_norm
        tick_deltas = torch.zeros_like(tick_norms)
        tick_deltas[:, 1:] = tick_norms[:, 1:] - tick_norms[:, :-1]
        tick_deltas[:, 0] = tick_norms[:, 0]
        tick_loss = (F.mse_loss(outputs["tick_delta"], tick_deltas.clamp(min=0), reduction="none") * mask).sum() / mask.sum().clamp(min=1)

        # Position loss (MSE) — arena_x_norm and arena_y_norm
        target_x = (target_features[:, :, 6] + 1) / 2  # [-1,1] -> [0,1]
        target_y = (target_features[:, :, 7] + 1) / 2  # [-1,1] -> [0,1]
        target_pos = torch.stack([target_x, target_y], dim=2)  # (batch, seq, 2)
        pos_loss = (F.mse_loss(outputs["arena_xy"], target_pos, reduction="none").mean(dim=2) * mask).sum() / mask.sum().clamp(min=1)

        # Side loss (BCE)
        target_side = target_features[:, :, 0]  # side is feature index 0
        side_loss = (F.binary_cross_entropy_with_logits(outputs["side_logit"], target_side, reduction="none") * mask).sum() / mask.sum().clamp(min=1)

        # KL divergence
        mu = outputs["mu"]
        logvar = outputs["logvar"]
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        total = card_loss + WEIGHT_TICK * tick_loss + WEIGHT_POS * pos_loss + WEIGHT_SIDE * side_loss + beta * kl_loss

        return total, {
            "card": card_loss.item(),
            "tick": tick_loss.item(),
            "pos": pos_loss.item(),
            "side": side_loss.item(),
            "kl": kl_loss.item(),
            "beta": beta,
            "total": total.item(),
        }

    def train(self) -> Path:
        """Run training loop with KL annealing and early stopping.

        Returns:
            Path to saved best model checkpoint.
        """
        self.model_dir.mkdir(parents=True, exist_ok=True)
        best_path = self.model_dir / "cvae_v2.pt"

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, EPOCHS + 1):
            t0 = time.time()

            # KL annealing
            beta = min(1.0, epoch / KL_ANNEAL_EPOCHS)

            # Unfreeze encoder after FREEZE_EPOCHS with differential LR
            if epoch == FREEZE_EPOCHS + 1:
                unfrozen = 0
                for name, param in self.model.encoder.named_parameters():
                    if not param.requires_grad:
                        param.requires_grad = True
                        unfrozen += 1
                if unfrozen > 0:
                    # Differential LR: encoder at 0.01x, rest at current LR
                    encoder_params = list(self.model.encoder.parameters())
                    encoder_ids = {id(p) for p in encoder_params}
                    other_params = [p for p in self.model.parameters()
                                    if p.requires_grad and id(p) not in encoder_ids]
                    self.optimizer = AdamW([
                        {"params": encoder_params, "lr": LEARNING_RATE * 0.01},
                        {"params": other_params, "lr": LEARNING_RATE},
                    ], weight_decay=WEIGHT_DECAY)
                    self.scheduler = CosineAnnealingLR(
                        self.optimizer, T_max=EPOCHS - epoch,
                    )
                    logger.info("Unfroze %d encoder params at epoch %d (lr=%.1e)", unfrozen, epoch, LEARNING_RATE * 0.01)

            # Training
            self.model.train()
            epoch_losses: dict[str, float] = {}
            n_batches = 0

            for batch in self.train_loader:
                card_ids, features, lengths, labels, p_decks, o_decks, mask = [
                    b.to(self.device) for b in batch
                ]

                self.optimizer.zero_grad()
                with _autocast_ctx(self.device):
                    outputs = self.model(card_ids, features, lengths, p_decks, o_decks)
                    loss, loss_dict = self._compute_loss(
                        outputs, card_ids, features, mask, beta,
                    )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                for k, v in loss_dict.items():
                    epoch_losses[k] = epoch_losses.get(k, 0) + v
                n_batches += 1

            self.scheduler.step()

            # Average training losses
            for k in epoch_losses:
                epoch_losses[k] /= max(n_batches, 1)

            # Validation
            val_loss, val_losses = self._evaluate(beta)

            elapsed = time.time() - t0
            logger.info(
                "Epoch %d/%d [%.1fs] beta=%.2f — "
                "train: total=%.4f card=%.4f kl=%.4f | "
                "val: total=%.4f card=%.4f kl=%.4f",
                epoch, EPOCHS, elapsed, beta,
                epoch_losses.get("total", 0), epoch_losses.get("card", 0),
                epoch_losses.get("kl", 0),
                val_loss, val_losses.get("card", 0), val_losses.get("kl", 0),
            )

            # Early stopping on val loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save({
                    "model_state_dict": self.model.state_dict(),
                    "vocab_size": self.model.vocab_size,
                    "latent_dim": self.model.latent_dim,
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "beta": beta,
                    "training_cutoff": self.training_cutoff,
                }, best_path)
                logger.info("  -> New best model saved (val_loss=%.4f)", val_loss)
            else:
                patience_counter += 1
                if patience_counter >= EARLY_STOPPING_PATIENCE:
                    logger.info(
                        "Early stopping at epoch %d (patience=%d)",
                        epoch, EARLY_STOPPING_PATIENCE,
                    )
                    break

        logger.info("CVAE training complete. Best val loss: %.4f", best_val_loss)
        return best_path

    def _evaluate(self, beta: float) -> tuple[float, dict[str, float]]:
        """Evaluate on validation set.

        Returns:
            (total_loss, loss_dict)
        """
        self.model.eval()
        total_losses: dict[str, float] = {}
        n_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                card_ids, features, lengths, labels, p_decks, o_decks, mask = [
                    b.to(self.device) for b in batch
                ]

                with _autocast_ctx(self.device):
                    outputs = self.model(card_ids, features, lengths, p_decks, o_decks)
                    _, loss_dict = self._compute_loss(
                        outputs, card_ids, features, mask, beta,
                    )

                for k, v in loss_dict.items():
                    total_losses[k] = total_losses.get(k, 0) + v
                n_batches += 1

        for k in total_losses:
            total_losses[k] /= max(n_batches, 1)

        return total_losses.get("total", 0), total_losses


def train_cvae(
    session: Session,
    model_dir: Optional[Path] = None,
) -> None:
    """Full CVAE training pipeline.

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
    dataset = CVAEDataset(session, vocab)
    if len(dataset) < 100:
        logger.error("Need at least 100 games with replay + deck data (have %d)", len(dataset))
        print(f"  ✗ Need at least 100 games (have {len(dataset)})")
        return

    # 3. Initialize model — try transfer from WP model
    wp_path = model_dir / "wp_v1.pt"
    if wp_path.exists():
        logger.info("Loading pretrained weights from %s", wp_path)
        print("  -> Transfer learning from WP model (ADR-004)")
        model = CounterfactualVAE.from_pretrained_wp(
            str(wp_path), vocab.size, device,
            freeze_encoder=True, dropout=DROPOUT,
        )
    else:
        logger.info("No WP checkpoint found — training from scratch")
        print("  -> Training from scratch (no WP checkpoint)")
        model = CounterfactualVAE(vocab_size=vocab.size, dropout=DROPOUT)

    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  -> Model: {n_total:,} params ({n_trainable:,} trainable)")

    # 4. Query training data cutoff date
    from sqlalchemy import func
    cutoff = session.query(func.max(Battle.battle_time)).scalar()
    cutoff_str = cutoff.isoformat() if cutoff else None
    logger.info("Training data cutoff: %s", cutoff_str)

    # 5. Train
    trainer = CVAETrainer(model, dataset, device, model_dir, training_cutoff=cutoff_str)
    best_path = trainer.train()

    # 6. Report
    checkpoint = torch.load(best_path, map_location=device, weights_only=True)
    print(f"  ✓ CVAE training complete")
    print(f"  ✓ Best val_loss={checkpoint['val_loss']:.4f} at epoch {checkpoint['epoch']}")
    print(f"  ✓ Training data cutoff: {checkpoint.get('training_cutoff', 'unknown')}")
    print(f"  ✓ Saved to {best_path}")
