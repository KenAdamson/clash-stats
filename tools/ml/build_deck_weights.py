"""Per-game training weights that make rare deck SHAPES matter to the loss.

The v10 evaluation found the model is catastrophically wrong exactly where the
deck prior should be worth most: real six-spell decks win 21% of their games and
the model opens them at 47%. The cause is not only architectural. There are 510
spell-heavy games in 1.68M -- 0.03% of training -- so the entire cohort could be
predicted backwards without moving val_loss. No architecture learns from a
gradient that thin.

Rarity is measured on deck SHAPE, not card popularity: an all-spells deck is
built from Fireball, Zap and The Log, three of the most popular cards in the
game, so per-card frequency would rank it as utterly ordinary. The composition
vector (spells, buildings, troops, avg elixir) is what makes it an outlier.

Weights are z-scored Euclidean distance from the corpus centroid, capped, so a
normal deck stays at 1.0 and the strangest decks reach WEIGHT_CAP. Applied by
multiplying the loss mask, which is already a weighted mean -- so nothing in the
loss or the batch signature changes.

Run with cwd=/app:
  PYTHONPATH=/app/src python3 tools/ml/build_deck_weights.py [shard_dir]
"""
import json, os, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/app/src")
from tracker.database import get_engine, get_session      # noqa: E402
from tracker.ml.card_metadata import CardVocabulary       # noqa: E402

WEIGHT_CAP = float(os.environ.get("WP_RARITY_CAP", "8.0"))
BETA = float(os.environ.get("WP_RARITY_BETA", "1.0"))

def main() -> None:
    shard = Path(sys.argv[1] if len(sys.argv) > 1 else "data/wp_shards")
    ids = np.load(shard / "deck_ids.npy", mmap_mode="r")
    n = ids.shape[0]
    vocab = CardVocabulary(get_session(get_engine(os.environ["DATABASE_URL"])))
    idx2name = {i: nm for nm, i in vocab._card_to_idx.items()}

    V = vocab.size
    spell = np.zeros(V, np.float32); bld = np.zeros(V, np.float32)
    elix = np.zeros(V, np.float32); known = np.zeros(V, np.float32)
    for i, nm in idx2name.items():
        if i < 2:
            continue
        known[i] = 1.0
        t = vocab.card_type(nm)          # authoritative: troop / spell / building
        spell[i] = t == "spell"
        bld[i] = t == "building"
        elix[i] = vocab.elixir(nm) or 0

    own = np.asarray(ids[:, 0])                     # (n, 8)
    n_cards = known[own].sum(1)
    comp = np.stack([
        spell[own].sum(1),
        bld[own].sum(1),
        (n_cards - spell[own].sum(1) - bld[own].sum(1)),   # troops
        np.where(n_cards > 0, elix[own].sum(1) / np.maximum(n_cards, 1), 0.0),
    ], axis=1)

    have = n_cards == 8                             # stubs keep weight 1.0
    mu = comp[have].mean(0); sd = comp[have].std(0)
    sd[sd == 0] = 1.0
    z = (comp - mu) / sd
    dist = np.linalg.norm(z, axis=1)
    # Centre on the TYPICAL distance, not on zero. The norm of 4 independent
    # z-scores averages ~1.9 even for a perfectly ordinary deck (chi with 4 df),
    # so `1 + dist` hands ~2.8 to everything and reweights nothing -- the first
    # version of this put 88% of games above weight 2.0. Scaling by the spread
    # between the median and the 95th percentile puts an ordinary deck at 1.0
    # and a genuine outlier near 1 + BETA, before the cap.
    d50 = np.median(dist[have])
    d95 = np.percentile(dist[have], 95)
    scale = max(d95 - d50, 1e-6)
    w = np.ones(n, np.float32)
    w[have] = np.clip(1.0 + BETA * (dist[have] - d50) / scale, 1.0, WEIGHT_CAP)
    print("distance: median %.3f  p95 %.3f  max %.3f" % (d50, d95, dist[have].max()))

    np.save(shard / "deck_weights.npy", w.astype(np.float16))
    print("deck shape stats over %d full decks:" % have.sum())
    for j, nm in enumerate(("spells", "buildings", "troops", "avg_elixir")):
        print("   %-11s mean %.2f  sd %.2f" % (nm, mu[j], sd[j]))
    print("weights: mean %.3f  median %.3f  p99 %.3f  max %.3f  (>2.0: %d games, >4.0: %d)"
          % (w.mean(), np.median(w), np.percentile(w, 99), w.max(),
             int((w > 2).sum()), int((w > 4).sum())))
    meta = json.loads((shard / "meta.json").read_text())
    meta["deck_weights"] = {"cap": WEIGHT_CAP, "beta": BETA, "basis": "deck-shape z-distance"}
    (shard / "meta.json").write_text(json.dumps(meta, indent=2))
    print("written: %s/deck_weights.npy" % shard)


if __name__ == "__main__":
    main()
