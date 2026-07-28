"""Weekly drift tripwire for the production WP model.

Evaluates the current production checkpoint on the NEWEST games with replay
data and appends the result to a history file. The point is to convert "hold
off retraining until the corpus doubles" from a bet into a monitored
position: a flat accuracy line means keep waiting with confidence; a
sustained drop means the meta has moved under the model (e.g. after a
balance patch) and a retrain is justified early, volume target or not.

Methodology mirrors WPTrainer._evaluate exactly — last-tick accuracy
(final prediction vs actual result) and masked per-tick BCE — so `acc` here
is directly comparable to the training run's reported val_acc. `loss` is
UNWEIGHTED BCE (training bakes in a class weight), so compare loss only
within this history, never to training logs.

Drift rule: WARN when accuracy falls more than DRIFT_THRESHOLD below the
median of this model version's own prior runs (fresh promotions start a
fresh baseline — a new model is expected to score differently).

Run with cwd=/app (tools convention):
    PYTHONPATH=/app/src python3 tools/eval/wp_drift_check.py
Env: WP_DRIFT_GAMES (default 5000), WP_DRIFT_THRESHOLD (default 0.015),
     WP_DRIFT_HISTORY (default data/wp_drift_history.jsonl)
"""

import json
import logging
import os
import time
from pathlib import Path

import torch
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from torch.utils.data import DataLoader

from tracker.ml.card_metadata import CardVocabulary
from tracker.ml.sequence_dataset import SequenceDataset
from tracker.ml.win_probability import WinProbabilityModel
from tracker.ml.wp_dataset import wp_collate_fn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("tracker.ml.wp_drift")

N_GAMES = int(os.environ.get("WP_DRIFT_GAMES", "5000"))
THRESHOLD = float(os.environ.get("WP_DRIFT_THRESHOLD", "0.015"))
HISTORY = Path(os.environ.get("WP_DRIFT_HISTORY", "data/wp_drift_history.jsonl"))


def production_checkpoint(session: Session) -> tuple[str, int]:
    """Resolve the production WP checkpoint (filename, version) from the registry."""
    row = session.execute(text(
        "SELECT filename, version FROM model_versions "
        "WHERE model_type='wp' AND status='production' "
        "ORDER BY version DESC LIMIT 1"
    )).first()
    if row is None:
        raise RuntimeError("no production WP model in the registry")
    return row[0], row[1]


def newest_battle_ids(session: Session, n: int) -> list[str]:
    """Newest n finished PvP/PoL battles that have replay events."""
    rows = session.execute(text("""
        SELECT b.battle_id FROM battles b
        JOIN (SELECT battle_id, count(*) AS ev FROM replay_events GROUP BY battle_id) e
          ON e.battle_id = b.battle_id
        WHERE b.result IN ('win','loss')
          AND b.battle_type IN ('PvP','pathOfLegend')
          AND e.ev >= 10
        ORDER BY b.battle_time DESC
        LIMIT :n
    """), {"n": n}).scalars().all()
    return list(rows)


def main() -> int:
    engine = create_engine(os.environ["DATABASE_URL"])
    device = torch.device("xpu" if hasattr(torch, "xpu") and torch.xpu.is_available() else "cpu")

    with Session(engine) as session:
        filename, version = production_checkpoint(session)
        ckpt_path = Path("data/ml_models") / filename
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        feat_dim = ckpt.get("feature_dim", 17)
        base_feat = 17
        model = WinProbabilityModel(
            vocab_size=ckpt["vocab_size"],
            feature_dim=feat_dim,
            extra_feature_dim=ckpt.get("extra_feature_dim", 0),
            tcn_channels=ckpt.get("tcn_channels"),
            card_embed_dim=ckpt.get("card_embed_dim", 16),
        )
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device).eval()

        vocab = CardVocabulary(session)
        ids = newest_battle_ids(session, N_GAMES)
        if len(ids) < 500:
            logger.warning("only %d fresh games — skipping drift check", len(ids))
            return 0
        ds = SequenceDataset(session, vocab, battle_ids=ids,
                             extra_features=(feat_dim > base_feat))

    loader = DataLoader(ds, batch_size=256, collate_fn=wp_collate_fn)
    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
    total_loss = total_ticks = correct = total_games = 0
    newest = oldest = None
    with torch.no_grad():
        for card_ids, features, lengths, labels, mask in loader:
            card_ids, features = card_ids.to(device), features.to(device)
            lengths, labels, mask = lengths.to(device), labels.to(device), mask.to(device)
            logits = model(card_ids, features, lengths)
            loss_per_tick = criterion(logits, labels)
            total_loss += (loss_per_tick * mask).sum().item()
            total_ticks += mask.sum().item()
            bsz = logits.size(0)
            last = (lengths - 1).clamp(min=0).long()
            last_logits = logits[torch.arange(bsz, device=device), last]
            preds = (last_logits > 0).float()
            correct += (preds == labels[:, 0]).sum().item()
            total_games += bsz

    acc = correct / max(total_games, 1)
    loss = total_loss / max(total_ticks, 1)

    # Baseline = median accuracy of THIS model version's prior runs.
    prior = []
    if HISTORY.exists():
        with open(HISTORY) as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                    if rec.get("version") == version:
                        prior.append(rec["acc"])
                except (ValueError, KeyError):
                    continue
    baseline = sorted(prior)[len(prior) // 2] if prior else None

    rec = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "version": version, "filename": filename,
        "n_games": total_games, "acc": round(acc, 4), "loss": round(loss, 4),
        "baseline": round(baseline, 4) if baseline is not None else None,
        "device": device.type,
    }
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY, "a") as fh:
        fh.write(json.dumps(rec) + "\n")

    if baseline is not None and acc < baseline - THRESHOLD:
        logger.warning(
            "WP DRIFT: v%d fresh-game acc %.4f is %.1fpp below its baseline %.4f "
            "(n=%d) — the meta may have moved; consider retraining early",
            version, acc, 100 * (baseline - acc), baseline, total_games,
        )
    else:
        logger.info(
            "WP drift check: v%d acc %.4f loss %.4f on %d fresh games "
            "(baseline %s) — no drift",
            version, acc, loss, total_games,
            f"{baseline:.4f}" if baseline is not None else "establishing",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
