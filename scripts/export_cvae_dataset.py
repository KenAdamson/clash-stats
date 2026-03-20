#!/usr/bin/env python3
"""Export CVAEDataset to a .pt file for cloud training.

Saves all tensors needed for CVAE training so the cloud VM doesn't
need PostgreSQL access. Upload the output file to Azure blob storage,
download on the GPU VM, and train from file.

Usage:
    docker exec cr-tracker python scripts/export_cvae_dataset.py

Output:
    data/ml_models/cvae_dataset.pt
"""

import os
import sys
import time

import torch

from tracker.database import init_db, get_session
from tracker.ml.card_metadata import CardVocabulary
from tracker.ml.cvae_dataset import CVAEDataset


def main():
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        print("DATABASE_URL not set")
        sys.exit(1)

    engine = init_db(db_url)
    session = get_session(engine)

    print("Building vocabulary...")
    vocab = CardVocabulary(session)
    print(f"  Vocab size: {vocab.size}")

    print("Loading dataset...")
    t0 = time.time()
    dataset = CVAEDataset(session, vocab)
    print(f"  {len(dataset)} games loaded in {time.time() - t0:.1f}s")

    # Extract all samples into serializable format
    print("Extracting tensors...")
    samples = []
    for i in range(len(dataset)):
        card_ids, features, label, p_deck, o_deck = dataset[i]
        samples.append((card_ids, features, label, p_deck, o_deck))

    output_path = "data/ml_models/cvae_dataset.pt"
    print(f"Saving to {output_path}...")
    torch.save({
        "samples": samples,
        "vocab_size": vocab.size,
        "n_games": len(dataset),
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, output_path)

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"  Saved: {size_mb:.1f} MB, {len(dataset)} games, vocab={vocab.size}")

    session.close()
    engine.dispose()


if __name__ == "__main__":
    main()
