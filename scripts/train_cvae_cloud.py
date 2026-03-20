#!/usr/bin/env python3
"""Train CVAE from exported dataset file — no database needed.

Designed for cloud GPU training (Azure T4/V100). Loads the dataset
from a .pt file exported by export_cvae_dataset.py.

Usage:
    python scripts/train_cvae_cloud.py \
        --dataset cvae_dataset.pt \
        --wp-checkpoint wp_v1.pt \
        --output cvae_v3.pt \
        --epochs 150

The output checkpoint can be downloaded and placed in
data/ml_models/ on the local machine.
"""

import argparse
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, Subset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Training hyperparameters
BATCH_SIZE = 256
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 15
DROPOUT = 0.2
VAL_FRACTION = 0.2

# KL targeting
KL_TARGET = 15.0
KL_ANNEAL_EPOCHS = 20

# Loss weights
WEIGHT_TICK = 0.5
WEIGHT_POS = 0.5
WEIGHT_SIDE = 0.3

# Encoder freeze schedule
FREEZE_EPOCHS = 10


class FileDataset(Dataset):
    """Dataset loaded from exported .pt file."""

    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    """Pad sequences and stack deck tensors."""
    card_ids_list, features_list, labels_list, p_decks, o_decks = zip(*batch)

    max_seq = 200  # truncate
    lengths = torch.tensor(
        [min(len(c), max_seq) for c in card_ids_list], dtype=torch.int64,
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

    arange = torch.arange(max_len).unsqueeze(0)
    mask = (arange < lengths.unsqueeze(1)).float()

    return padded_card_ids, padded_features, lengths, labels, player_deck_ids, opponent_deck_ids, mask


def detect_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA: %s", torch.cuda.get_device_name(0))
        return device
    logger.info("Using CPU")
    return torch.device("cpu")


def compute_loss(outputs, target_card_ids, target_features, mask, beta, vocab_size):
    batch_size, seq_len = target_card_ids.shape

    card_criterion = nn.CrossEntropyLoss(ignore_index=0, reduction="none")
    card_logits = outputs["card_logits"]
    card_loss = card_criterion(
        card_logits.reshape(-1, card_logits.size(-1)),
        target_card_ids.reshape(-1),
    ).reshape(batch_size, seq_len)
    card_loss = (card_loss * mask).sum() / mask.sum().clamp(min=1)

    tick_norms = target_features[:, :, 1]
    tick_deltas = torch.zeros_like(tick_norms)
    tick_deltas[:, 1:] = tick_norms[:, 1:] - tick_norms[:, :-1]
    tick_deltas[:, 0] = tick_norms[:, 0]
    tick_loss = (F.mse_loss(outputs["tick_delta"], tick_deltas.clamp(min=0), reduction="none") * mask).sum() / mask.sum().clamp(min=1)

    target_x = (target_features[:, :, 6] + 1) / 2
    target_y = (target_features[:, :, 7] + 1) / 2
    target_pos = torch.stack([target_x, target_y], dim=2)
    pos_loss = (F.mse_loss(outputs["arena_xy"], target_pos, reduction="none").mean(dim=2) * mask).sum() / mask.sum().clamp(min=1)

    target_side = target_features[:, :, 0]
    side_loss = (F.binary_cross_entropy_with_logits(outputs["side_logit"], target_side, reduction="none") * mask).sum() / mask.sum().clamp(min=1)

    mu = outputs["mu"]
    logvar = outputs["logvar"]
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kl_total = kl_per_dim.sum(dim=1).mean()
    kl_loss = torch.abs(kl_total - KL_TARGET)

    total = card_loss + WEIGHT_TICK * tick_loss + WEIGHT_POS * pos_loss + WEIGHT_SIDE * side_loss + beta * kl_loss

    return total, {
        "card": card_loss.item(),
        "tick": tick_loss.item(),
        "pos": pos_loss.item(),
        "side": side_loss.item(),
        "kl": kl_total.item(),
        "total": total.item(),
    }


def main():
    parser = argparse.ArgumentParser(description="Train CVAE from exported dataset")
    parser.add_argument("--dataset", required=True, help="Path to cvae_dataset.pt")
    parser.add_argument("--wp-checkpoint", required=True, help="Path to wp_v1.pt")
    parser.add_argument("--output", default="cvae_v3.pt", help="Output checkpoint path")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    device = detect_device()

    # Load dataset
    logger.info("Loading dataset from %s", args.dataset)
    data = torch.load(args.dataset, map_location="cpu", weights_only=False)
    logger.info("  %d games, vocab_size=%d", data["n_games"], data["vocab_size"])

    dataset = FileDataset(data["samples"])
    vocab_size = data["vocab_size"]

    n = len(dataset)
    n_val = int(n * VAL_FRACTION)
    n_train = n - n_val

    train_loader = DataLoader(
        Subset(dataset, list(range(n_train))),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        Subset(dataset, list(range(n_train, n))),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=(device.type == "cuda"),
    )

    # Build model
    from tracker.ml.cvae import CounterfactualVAE

    if args.resume:
        logger.info("Resuming from %s", args.resume)
        cp = torch.load(args.resume, map_location=device, weights_only=True)
        model = CounterfactualVAE(
            vocab_size=cp["vocab_size"],
            latent_dim=cp.get("latent_dim", 64),
            deck_bottleneck_dim=cp.get("deck_bottleneck_dim", 8),
        )
        model.load_state_dict(cp["model_state_dict"])
        model.to(device)
        logger.info("  Resumed from epoch %d (val_loss=%.6f)", cp["epoch"], cp["val_loss"])
    else:
        logger.info("Transfer learning from %s", args.wp_checkpoint)
        model = CounterfactualVAE.from_pretrained_wp(
            args.wp_checkpoint, vocab_size, device,
            freeze_encoder=True, dropout=DROPOUT,
        )

    model.train()

    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model: %s params (%s trainable)", f"{n_total:,}", f"{n_trainable:,}")

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    best_path = Path(args.output)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        beta = min(1.0, epoch / KL_ANNEAL_EPOCHS)

        # Unfreeze encoder
        if epoch == FREEZE_EPOCHS + 1:
            unfrozen = 0
            for name, param in model.encoder.named_parameters():
                if not param.requires_grad:
                    param.requires_grad = True
                    unfrozen += 1
            if unfrozen > 0:
                encoder_params = list(model.encoder.parameters())
                encoder_ids = {id(p) for p in encoder_params}
                other_params = [p for p in model.parameters()
                                if p.requires_grad and id(p) not in encoder_ids]
                optimizer = AdamW([
                    {"params": encoder_params, "lr": LEARNING_RATE * 0.01},
                    {"params": other_params, "lr": LEARNING_RATE},
                ], weight_decay=WEIGHT_DECAY)
                scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs - epoch)
                logger.info("Unfroze %d encoder params (lr=%.1e)", unfrozen, LEARNING_RATE * 0.01)

        # Train
        model.train()
        epoch_losses = {}
        n_batches = 0
        for batch in train_loader:
            card_ids, features, lengths, labels, p_decks, o_decks, mask = [
                b.to(device) for b in batch
            ]
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                outputs = model(card_ids, features, lengths, p_decks, o_decks)
                loss, loss_dict = compute_loss(outputs, card_ids, features, mask, beta, vocab_size)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for k, v in loss_dict.items():
                epoch_losses[k] = epoch_losses.get(k, 0) + v
            n_batches += 1

        scheduler.step()
        for k in epoch_losses:
            epoch_losses[k] /= max(n_batches, 1)

        # Validate
        model.eval()
        val_losses = {}
        val_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                card_ids, features, lengths, labels, p_decks, o_decks, mask = [
                    b.to(device) for b in batch
                ]
                with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "cuda")):
                    outputs = model(card_ids, features, lengths, p_decks, o_decks)
                    _, loss_dict = compute_loss(outputs, card_ids, features, mask, beta, vocab_size)
                for k, v in loss_dict.items():
                    val_losses[k] = val_losses.get(k, 0) + v
                val_batches += 1

        for k in val_losses:
            val_losses[k] /= max(val_batches, 1)
        val_loss = val_losses.get("total", 0)

        elapsed = time.time() - t0
        logger.info(
            "Epoch %d/%d [%.1fs] beta=%.2f — "
            "train: total=%.4f card=%.4f kl=%.2f (target=%d) | "
            "val: total=%.4f card=%.4f kl=%.2f",
            epoch, args.epochs, elapsed, beta,
            epoch_losses.get("total", 0), epoch_losses.get("card", 0),
            epoch_losses.get("kl", 0), KL_TARGET,
            val_loss, val_losses.get("card", 0), val_losses.get("kl", 0),
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "vocab_size": vocab_size,
                "latent_dim": model.latent_dim,
                "deck_bottleneck_dim": model.deck_bottleneck_dim,
                "epoch": epoch,
                "val_loss": val_loss,
                "beta": beta,
                "kl_target": KL_TARGET,
                "training_cutoff": data.get("exported_at"),
            }, best_path)
            logger.info("  -> New best (val_loss=%.4f)", val_loss)
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                logger.info("Early stopping at epoch %d", epoch)
                break

    logger.info("Training complete. Best val_loss=%.6f", best_val_loss)
    logger.info("Checkpoint: %s", best_path)


if __name__ == "__main__":
    main()
