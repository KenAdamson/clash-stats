"""A/B a WP checkpoint against production, on identical games.

The v10 question is narrow: does knowing both decks at tick 0 make P(win)
better? So the comparison holds everything else fixed — same shard rows, same
time-ordered split, same batches — and reports the places a deck prior should
show up if it is real:

  1. Held-out val loss / accuracy — the headline, but the least diagnostic.
  2. Tick-0 spread. v9's pre_game_wp has sd 0.124 and median 0.467 across 518K
     games, i.e. it opens nearly every game at a coin flip because it cannot
     see a matchup. If the deck prior works, tick-0 spread must WIDEN and the
     opening probability must correlate with the eventual result.
  3. Early-game accuracy. A matchup prior should pay off most in the first
     few placements and wash out later, when the board speaks for itself.
     Flat improvement across all ticks would suggest something else changed.
  4. Calibration by trophy tier, the diagnostic that killed earlier versions.
  5. The two adversarial games, scored per tick.

Pre-registered: v10 is a candidate for promotion only if val_loss improves AND
tick-0 AUC beats v9 materially. A val_loss win with a flat tick-0 AUC means the
gain came from somewhere other than the feature under test, and the honest read
would be that the deck prior did not do the work.

Run with cwd=/app:
  PYTHONPATH=/app/src python3 tools/eval/wp_v10_ab.py wp_v9.pt wp_v10.pt
"""

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/app/src")
from tracker.database import get_engine, get_session            # noqa: E402
from tracker.ml.card_metadata import CardVocabulary             # noqa: E402
from tracker.ml.win_probability import WinProbabilityModel      # noqa: E402
from tracker.ml.wp_shard_cache import ShardDataset, ShardBatchLoader  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("tracker.ml.v10_ab")

SHARD_DIR = os.environ.get("WP_SHARD_DIR", "data/wp_shards")
MODEL_DIR = Path("data/ml_models")
VAL_FRACTION = 0.2
BATCH = int(os.environ.get("WP_EVAL_BATCH", "512"))
OUT = Path("data/wp_v10_ab.json")
ADVERSARIAL = {
    "a12eb22d3354ef326618f1db416c18be": "TOxicCorRiN 3-2 (e-golem beatdown, won at 3.6% floor)",
    "4622a6ffd148f33524564ea9a39e61df": "HaWk58 1-0 (double-spawner turtle, won at ~20%)",
}


def _device() -> torch.device:
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def load_model(path: Path, device) -> WinProbabilityModel:
    ck = torch.load(path, map_location=device, weights_only=True)
    m = WinProbabilityModel(
        vocab_size=ck["vocab_size"],
        feature_dim=ck.get("feature_dim", 17),
        extra_feature_dim=ck.get("extra_feature_dim", 0),
        tcn_channels=ck.get("tcn_channels"),
        card_embed_dim=ck.get("card_embed_dim", 16),
        deck_features=ck.get("deck_features", False),
        dropout=0.0,
    )
    m.load_state_dict(ck["model_state_dict"])
    m.to(device).eval()
    logger.info("%s: deck_features=%s feature_dim=%s val_acc=%.4f",
                path.name, m.deck_features, m.feature_dim, ck.get("val_acc", float("nan")))
    return m


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    pos, neg = scores[labels == 1], scores[labels == 0]
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort() + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def evaluate(model, loader, device, tick_buckets=(0, 1, 3, 8, 20)) -> dict:
    """Loss/accuracy overall, plus tick-0 scores and accuracy by tick bucket."""
    tot_loss = tot_n = tot_correct = 0.0
    t0_scores, t0_labels = [], []
    by_bucket = {b: [0.0, 0.0] for b in tick_buckets}   # bucket -> [correct, n]
    bce = torch.nn.BCEWithLogitsLoss(reduction="none")

    with torch.no_grad():
        for card_ids, features, lengths, labels, mask, deck_ids, deck_vars in loader:
            card_ids, features = card_ids.to(device), features.to(device)
            lengths, labels, mask = lengths.to(device), labels.to(device), mask.to(device)
            deck_ids, deck_vars = deck_ids.to(device), deck_vars.to(device)
            logits = model(card_ids, features, lengths, deck_ids, deck_vars)
            loss = (bce(logits, labels) * mask).sum()
            tot_loss += float(loss)
            tot_n += float(mask.sum())
            tot_correct += float((((logits > 0).float() == labels) * mask).sum())

            probs = torch.sigmoid(logits)
            t0_scores.append(probs[:, 0].cpu().numpy())
            t0_labels.append(labels[:, 0].cpu().numpy())
            for b in tick_buckets:
                if logits.size(1) <= b:
                    continue
                m_b = mask[:, b]
                by_bucket[b][0] += float((((logits[:, b] > 0).float() == labels[:, b]) * m_b).sum())
                by_bucket[b][1] += float(m_b.sum())

    t0s = np.concatenate(t0_scores)
    t0l = np.concatenate(t0_labels)
    return {
        "val_loss": tot_loss / max(tot_n, 1),
        "val_acc": tot_correct / max(tot_n, 1),
        "tick0_mean": float(t0s.mean()),
        "tick0_sd": float(t0s.std()),
        "tick0_p5": float(np.percentile(t0s, 5)),
        "tick0_p95": float(np.percentile(t0s, 95)),
        "tick0_auc": auc(t0s, t0l),
        "acc_by_tick": {str(b): (c / n if n else None) for b, (c, n) in by_bucket.items()},
        "n_ticks": tot_n,
    }


def main() -> None:
    a_name = sys.argv[1] if len(sys.argv) > 1 else "wp_v9.pt"
    b_name = sys.argv[2] if len(sys.argv) > 2 else "wp_v10.pt"
    device = _device()
    logger.info("device: %s", device)

    ds = ShardDataset(SHARD_DIR)
    if not ds.has_decks:
        raise SystemExit(f"{SHARD_DIR} has no deck arrays — run backfill_shard_decks.py first")
    n_val = int(ds.n * VAL_FRACTION)
    val_idx = np.arange(ds.n - n_val, ds.n)      # time-ordered holdout, as in training
    logger.info("val split: %d games (last %.0f%% by battle_time)", len(val_idx), VAL_FRACTION * 100)

    report = {"shard_dir": SHARD_DIR, "n_val_games": len(val_idx), "models": {}}
    for name in (a_name, b_name):
        path = MODEL_DIR / name
        if not path.exists():
            logger.warning("skipping %s — not found", path)
            continue
        model = load_model(path, device)
        loader = ShardBatchLoader(ds, val_idx, BATCH, shuffle=False)
        res = evaluate(model, loader, device)
        res["deck_features"] = model.deck_features
        report["models"][name] = res
        print(f"\n== {name} (deck_features={model.deck_features}) ==")
        print(f"  val_loss {res['val_loss']:.4f}   val_acc {res['val_acc']:.4f}")
        print(f"  tick-0: mean {res['tick0_mean']:.3f}  sd {res['tick0_sd']:.4f}  "
              f"p5-p95 {res['tick0_p5']:.3f}-{res['tick0_p95']:.3f}  AUC {res['tick0_auc']:.4f}")
        print("  acc by tick: " + "  ".join(
            f"t{k}={v:.4f}" for k, v in res["acc_by_tick"].items() if v is not None))
        del model

    if len(report["models"]) == 2:
        a, b = report["models"][a_name], report["models"][b_name]
        d_loss = b["val_loss"] - a["val_loss"]
        d_auc = b["tick0_auc"] - a["tick0_auc"]
        verdict = ("CANDIDATE" if d_loss < 0 and d_auc > 0.02 else
                   "INCONCLUSIVE" if d_loss < 0 else "REJECT")
        report["verdict"] = {"delta_val_loss": d_loss, "delta_tick0_auc": d_auc,
                             "delta_tick0_sd": b["tick0_sd"] - a["tick0_sd"],
                             "result": verdict}
        print(f"\n== VERDICT: {verdict} ==")
        print(f"  val_loss   {a['val_loss']:.4f} -> {b['val_loss']:.4f}  ({d_loss:+.4f})")
        print(f"  tick0 AUC  {a['tick0_auc']:.4f} -> {b['tick0_auc']:.4f}  ({d_auc:+.4f})")
        print(f"  tick0 sd   {a['tick0_sd']:.4f} -> {b['tick0_sd']:.4f}  "
              f"({b['tick0_sd'] - a['tick0_sd']:+.4f})")
        if verdict == "INCONCLUSIVE":
            print("  val_loss improved but tick-0 discrimination did not — the gain is")
            print("  not attributable to the deck prior, which is the feature under test.")

    OUT.write_text(json.dumps(report, indent=1))
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()
