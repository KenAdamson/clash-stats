"""Measure the real per-game cost of WP inference, to size a full rescore.

A full corpus rescore is a multi-hour commitment, so the estimate that decides
whether to start it should be measured rather than guessed. Times the two phases
separately because they scale differently and only one of them is on the GPU:
building the SequenceDataset (reads replay_events, CPU + IO bound) and the
forward pass (XPU).

Run with cwd=/app:
  PYTHONPATH=/app/src python3 tools/diag/wp_rescore_estimate.py [n_games]
"""

import logging
import os
import sys
import time
from pathlib import Path

import torch
from sqlalchemy import text

sys.path.insert(0, "/app/src")
from tracker.database import get_engine, get_session            # noqa: E402
from tracker.ml.card_metadata import CardVocabulary             # noqa: E402
from tracker.ml.sequence_dataset import SequenceDataset         # noqa: E402
from tracker.ml.wp_dataset import wp_collate_fn                 # noqa: E402
from tracker.ml.wp_training import _resolve_wp_path, load_wp_model  # noqa: E402

logging.basicConfig(level=logging.WARNING)


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    session = get_session(get_engine(os.environ["DATABASE_URL"]))

    bids = [r[0] for r in session.execute(
        text("SELECT battle_id FROM game_wp_summary "
             "WHERE model_version = :v LIMIT :n"), {"v": "wp-v1", "n": n})]
    if not bids:
        raise SystemExit("no wp-v1 games found to sample")
    print("sampled %d already-scored games" % len(bids))

    vocab = CardVocabulary(session)
    t0 = time.time()
    ds = SequenceDataset(session, vocab, battle_ids=bids, extra_features=True)
    t_build = time.time() - t0
    if len(ds) == 0:
        raise SystemExit("dataset built empty — sample not usable")
    print("  dataset build : %6.1fs  %5d games  %.4f s/game" % (
        t_build, len(ds), t_build / len(ds)))

    path = _resolve_wp_path(session, Path("/app/data/ml_models"))
    model, ck = load_wp_model(path)
    dev = torch.device("xpu" if (hasattr(torch, "xpu") and torch.xpu.is_available()) else "cpu")
    model = model.to(dev).eval()

    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=wp_collate_fn)
    t0 = time.time()
    ticks = 0
    with torch.no_grad():
        for batch in loader:
            card_ids, feats, lengths, labels, mask, deck_ids, deck_vars = batch
            card_ids, feats = card_ids.to(dev), feats.to(dev)
            kw = {}
            if ck.get("deck_features"):
                kw["deck_ids"] = deck_ids.to(dev)
                kw["deck_variants"] = deck_vars.to(dev)
            out = model(card_ids, feats, lengths.to(dev), **kw)
            ticks += int(out.numel())
    t_fwd = time.time() - t0
    print("  forward (%-3s) : %6.1fs  %5d games  %.4f s/game  (%d ticks)" % (
        str(dev), t_fwd, len(ds), t_fwd / len(ds), ticks))

    per_game = (t_build + t_fwd) / len(ds)
    total = 585541
    print("\n  model %s on %s" % (path.name, dev))
    print("  combined      : %.4f s/game  (excludes the row INSERTs)" % per_game)
    print("  => %d games ~ %.1f hours of compute, plus ~%.1fM tick rows to write" % (
        total, total * per_game / 3600.0, total * (ticks / len(ds)) / 1e6))


if __name__ == "__main__":
    main()
