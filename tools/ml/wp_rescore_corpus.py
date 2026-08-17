"""Rescore every legacy-labelled game with the current production WP model.

The 585k games carrying model_version='wp-v1' were scored by whatever model was
production at the time -- v1 through v9 -- under a hardcoded label that made them
indistinguishable. This re-runs them all under the model the registry resolves
today, stamping the real checkpoint stem, and drops the legacy rows as it goes
(Ken's call: storage over an A/B we can reconstruct by re-running).

The two tables behave DIFFERENTLY, which drives the whole design:

  - game_wp_summary's primary key is battle_id ALONE, so run_inference's
    session.merge() RELABELS the existing row in place, wp-v1 -> wp_v13. One
    summary row per game, always; there is never a window where a game has no
    summary (which matters, because the live 5-minute cron queues work by
    anti-joining game_wp_summary WITHOUT filtering on model_version -- a gap
    would make it race us for the GPU).
  - win_probability is keyed (battle_id, game_tick, model_version), so its
    per-tick rows genuinely coexist and the old ones must be deleted explicitly.

That asymmetry creates one failure mode: a crash between the summary relabel
and the tick delete strands wp-v1 tick rows whose game is no longer in the
queue (the queue is driven by summary version). _sweep_orphans() clears them at
startup, so a killed run self-heals on restart rather than leaking silently.

Resumability needs no state file: the work queue is "summaries still labelled
wp-v1", which shrinks as batches commit. Safe to kill and restart at any point.

Politeness, because this shares one small-BAR A770 and one Postgres with live
ingest:
  - takes xpu_train.lock so it cannot collide with a training run
  - sleeps between batches (WP_RESCORE_SLEEP) to leave IO for the scrapers
  - bounded batches so no single transaction holds a long write lock

Run with cwd=/app:
  PYTHONPATH=/app/src python3 tools/ml/wp_rescore_corpus.py [batch_size] [max_batches]
"""

import logging
import os
import sys
import time
from pathlib import Path

import torch
from sqlalchemy import text
from torch.utils.data import DataLoader

sys.path.insert(0, "/app/src")
from tracker.database import get_engine, get_session                  # noqa: E402
from tracker.ml.calibration import PlattCalibrator                    # noqa: E402
from tracker.ml.card_metadata import CardVocabulary                   # noqa: E402
from tracker.ml.sequence_dataset import SequenceDataset               # noqa: E402
from tracker.ml.wp_dataset import wp_collate_fn                       # noqa: E402
from tracker.ml.wp_training import (                                  # noqa: E402
    WPTrainer, _resolve_wp_path, load_wp_model, BATCH_SIZE,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("wp_rescore")

LEGACY_VERSION = os.environ.get("WP_RESCORE_FROM", "wp-v1")
SLEEP_BETWEEN = float(os.environ.get("WP_RESCORE_SLEEP", "2.0"))


def main() -> None:
    batch_size = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    max_batches = int(sys.argv[2]) if len(sys.argv) > 2 else 0     # 0 = until done

    session = get_session(get_engine(os.environ["DATABASE_URL"]))
    model_dir = Path("/app/data/ml_models")
    wp_path = _resolve_wp_path(session, model_dir)
    if wp_path is None:
        raise SystemExit("no production WP model resolved — refusing to run")
    target = wp_path.stem
    if target == LEGACY_VERSION:
        raise SystemExit("production model is already labelled %s" % LEGACY_VERSION)

    model, ck = load_wp_model(wp_path)
    device = torch.device("xpu" if (hasattr(torch, "xpu") and torch.xpu.is_available())
                          else "cpu")
    model = model.to(device).eval()
    feat_dim = ck.get("feature_dim", 17)

    calibrator = None
    cal_path = model_dir / ("%s_calibrator.json" % target)
    if cal_path.exists():
        calibrator = PlattCalibrator.load(cal_path)
        logger.info("Loaded calibrator %s", cal_path.name)
    else:
        logger.warning("NO calibrator for %s — scores will be uncalibrated, "
                       "which would make them incomparable to live rows", target)
        raise SystemExit("refusing to write uncalibrated scores")

    _sweep_orphans(session)

    vocab = CardVocabulary(session)
    remaining = session.execute(
        text("SELECT count(*) FROM game_wp_summary WHERE model_version = :v"),
        {"v": LEGACY_VERSION}).scalar()
    logger.info("Rescoring %s -> %s on %s | %d games queued | batch=%d",
                LEGACY_VERSION, target, device, remaining, batch_size)

    done = total_deleted = 0
    batch_no = 0
    t_start = time.time()
    while True:
        if max_batches and batch_no >= max_batches:
            logger.info("Reached max_batches=%d — stopping", max_batches)
            break

        bids = [r[0] for r in session.execute(
            text("SELECT battle_id FROM game_wp_summary "
                 "WHERE model_version = :v ORDER BY battle_id LIMIT :n"),
            {"v": LEGACY_VERSION, "n": batch_size})]
        if not bids:
            logger.info("Queue empty — rescore complete")
            break
        batch_no += 1

        ds = SequenceDataset(session, vocab, battle_ids=bids,
                             extra_features=(feat_dim > 17))
        if len(ds) == 0:
            # These games no longer meet the event threshold (events pruned, or
            # reclassified). They can never be rescored, so drop the stale rows
            # rather than spinning on them forever -- without this the LIMIT
            # would return the same unusable batch every iteration.
            d = _drop_legacy(session, bids)
            total_deleted += d
            logger.warning("Batch %d: no usable games; dropped %d stale rows", batch_no, d)
            continue

        # MUST be the dataset's own ordering, not the input list: games that
        # fail extraction are dropped, so bids[:len(ds)] would shift every
        # subsequent score onto the wrong game -- silently, since both are
        # plausible battle_ids.
        ordered = ds.battle_ids_in_order

        trainer = WPTrainer.__new__(WPTrainer)
        trainer.model = model
        trainer.device = device
        trainer.model_version = target
        trainer.full_loader = DataLoader(
            ds, batch_size=(512 if feat_dim > 17 else BATCH_SIZE), shuffle=False,
            collate_fn=wp_collate_fn, num_workers=0)

        n = trainer.run_inference(session, ds, ordered, vocab, calibrator=calibrator)
        # Only now that the new rows are committed is it safe to drop the old.
        total_deleted += _drop_legacy(session, bids)
        done += n

        rate = done / max(time.time() - t_start, 1e-9)
        left = max(remaining - done, 0)
        logger.info("Batch %d: scored %d (total %d, %.1f games/s, ~%.1fh left)",
                    batch_no, n, done, rate, left / rate / 3600 if rate else 0)
        if SLEEP_BETWEEN:
            time.sleep(SLEEP_BETWEEN)

    logger.info("DONE: %d games rescored as %s, %d legacy TICK rows deleted, %.1f min",
                done, target, total_deleted, (time.time() - t_start) / 60)


def _sweep_orphans(session) -> None:
    """Delete legacy tick rows whose game has already been relabelled.

    Only possible if a previous run died between the two writes. Scoped by a
    join on the summary so it can never touch a game still awaiting rescore.
    """
    n = session.execute(text("""
        DELETE FROM win_probability w
        USING game_wp_summary g
        WHERE w.battle_id = g.battle_id
          AND w.model_version = :legacy
          AND g.model_version <> :legacy
    """), {"legacy": LEGACY_VERSION}).rowcount
    session.commit()
    if n:
        logger.warning("Swept %d orphaned %s tick rows from an interrupted run",
                       n, LEGACY_VERSION)


def _drop_legacy(session, bids: list) -> int:
    """Remove the legacy rows for these battles, both tables."""
    # The summary DELETE normally matches nothing -- merge() already relabelled
    # it -- and exists only to clear games that dropped out of the dataset.
    session.execute(
        text("DELETE FROM game_wp_summary WHERE model_version = :v "
             "AND battle_id = ANY(:b)"), {"v": LEGACY_VERSION, "b": bids})
    n = session.execute(
        text("DELETE FROM win_probability WHERE model_version = :v "
             "AND battle_id = ANY(:b)"), {"v": LEGACY_VERSION, "b": bids}).rowcount
    session.commit()
    return n or 0


if __name__ == "__main__":
    main()
