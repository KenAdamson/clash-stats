"""Build the dashboard manifold from the hand-crafted features, not the TCN.

The TCN manifold was abandoned for a structural reason, not a tuning one:
tcn_v2 is trained with BCEWithLogitsLoss on win/loss and nothing else, so its
128-dim layer is the penultimate layer of a binary classifier and encodes the
decision boundary. Clustering it recovered outcome -- the three largest clusters
were 80%, 1% and 2% win rate, with only 2.1% of games in near-base-rate
clusters. No parameter choice fixes that; the space simply does not carry
tempo except insofar as tempo predicts winning.

features.py's 50-dim vector is descriptive instead: durations, play spacing,
aggression, lane balance, elixir, card counts. Those are causes rather than a
compression of the outcome, so structure found in them means something a player
can act on.

Writes into game_embeddings.embedding_vec_3d and cluster_id, which is what the
dashboard already reads -- so the existing Plotly view works untouched. The
128-dim TCN embeddings in embedding_tcn_128d are LEFT ALONE: what is being
dropped is the TCN's UMAP projection, not the encoder output, which was
expensive and may still be useful for similarity work.

Run with cwd=/app:
  PYTHONPATH=/app/src python3 tools/ml/feature_manifold.py [--corpus-sample N] [--dry]
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from sqlalchemy import select, text

sys.path.insert(0, "/app/src")
from tracker.database import get_engine, get_session                # noqa: E402
from tracker.models import Battle, DeckCard, ReplayEvent, ReplaySummary  # noqa: E402
from tracker.ml.features import _extract_features_from_loaded, FEATURE_VERSION  # noqa: E402
from tracker.ml.storage import GameFeature, to_blob                 # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("feature_manifold")

WORK = Path(os.environ.get("FEATURE_MANIFOLD_WORK", "/app/data/feature_manifold"))
CHUNK = 1000


def select_battles(session, corpus_sample: int) -> list[str]:
    """All personal games, plus the most recent N corpus games.

    Personal games are never sampled -- they are the ones actually looked at on
    the dashboard, and dropping any of them would put holes in the view. The
    corpus sample exists to give UMAP enough neighbours to find structure; the
    dashboard only renders ~20k of them anyway.
    """
    personal = [r[0] for r in session.execute(text("""
        SELECT b.battle_id FROM battles b
        WHERE b.corpus = 'personal' AND b.battle_type IN ('PvP','pathOfLegend')
          AND b.result IN ('win','loss')
          AND EXISTS (SELECT 1 FROM replay_events re WHERE re.battle_id = b.battle_id)
        ORDER BY b.battle_time
    """))]
    corpus = [r[0] for r in session.execute(text("""
        SELECT b.battle_id FROM battles b
        WHERE b.corpus <> 'personal' AND b.battle_type IN ('PvP','pathOfLegend')
          AND b.result IN ('win','loss')
          AND EXISTS (SELECT 1 FROM replay_events re WHERE re.battle_id = b.battle_id)
        ORDER BY b.battle_time DESC LIMIT :n
    """), {"n": corpus_sample})]
    logger.info("selected %d personal + %d corpus games", len(personal), len(corpus))
    return personal + corpus


def extract(session, bids: list[str]) -> tuple[list[str], np.ndarray]:
    """Bulk-load per chunk and extract the 50-dim vector for each game."""
    cache = WORK / "features_v3.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        logger.info("reusing %d cached feature vectors", len(d["ids"]))
        return list(d["ids"]), d["vecs"]

    ids: list[str] = []
    vecs: list[np.ndarray] = []
    skipped = 0
    t0 = time.time()
    for s in range(0, len(bids), CHUNK):
        chunk = bids[s:s + CHUNK]
        battles = {b.battle_id: b for b in session.execute(
            select(Battle).where(Battle.battle_id.in_(chunk))).scalars()}
        events: dict[str, list] = {b: [] for b in chunk}
        for e in session.execute(select(ReplayEvent)
                                 .where(ReplayEvent.battle_id.in_(chunk))
                                 .order_by(ReplayEvent.game_tick)).scalars():
            events[e.battle_id].append(e)
        summaries: dict[str, list] = {b: [] for b in chunk}
        for r in session.execute(select(ReplaySummary)
                                 .where(ReplaySummary.battle_id.in_(chunk))).scalars():
            summaries[r.battle_id].append(r)
        decks: dict[str, list] = {b: [] for b in chunk}
        for dc in session.execute(select(DeckCard)
                                  .where(DeckCard.battle_id.in_(chunk))).scalars():
            decks[dc.battle_id].append(dc)

        for bid in chunk:
            b = battles.get(bid)
            if b is None:
                skipped += 1
                continue
            v = _extract_features_from_loaded(b, events[bid], summaries[bid], decks[bid])
            if v is None or not np.isfinite(v).all():
                skipped += 1
                continue
            ids.append(bid)
            vecs.append(v)
        session.expire_all()
        if (s // CHUNK) % 25 == 0:
            logger.info("  %d/%d extracted (%d skipped, %.0f/s)", len(ids), len(bids),
                        skipped, len(ids) / max(time.time() - t0, 1e-9))

    arr = np.asarray(vecs, dtype=np.float32)
    WORK.mkdir(parents=True, exist_ok=True)
    np.savez(cache, ids=np.array(ids, dtype=object), vecs=arr)
    logger.info("extracted %d vectors of dim %d in %.1f min (%d skipped)",
                len(ids), arr.shape[1] if len(arr) else 0,
                (time.time() - t0) / 60, skipped)
    return ids, arr


def project_and_cluster(vecs: np.ndarray, min_cluster_size: int):
    """Standardise, then a display projection and a separate clustering one.

    Standardising first is not optional here: unlike a neural embedding these
    features are in wildly different units -- seconds beside ratios beside card
    counts -- and Euclidean distance would otherwise be dominated by whichever
    column happens to have the largest range.

    Two projections for the reason learned on the TCN manifold: the display
    params (min_dist=0.3) deliberately spread points so the scatter does not
    overplot, which erases exactly the density gaps HDBSCAN needs. Clustering a
    visualisation produced a single blob holding 99% of games.
    """
    import hdbscan
    from sklearn.preprocessing import StandardScaler
    from umap import UMAP

    scaled = StandardScaler().fit_transform(vecs.astype(np.float64))
    logger.info("projecting %d x %d for display", *scaled.shape)
    t0 = time.time()
    disp = UMAP(n_components=3, n_neighbors=30, min_dist=0.3, spread=1.5,
                metric="euclidean", random_state=42).fit_transform(scaled)
    logger.info("display projection in %.1f min", (time.time() - t0) / 60)

    t0 = time.time()
    clust_space = UMAP(n_components=10, n_neighbors=30, min_dist=0.0,
                       metric="euclidean", random_state=42).fit_transform(scaled)
    labels = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=5,
                             cluster_selection_method="eom",
                             core_dist_n_jobs=4).fit_predict(clust_space)
    k = len(set(labels.tolist())) - (1 if -1 in labels else 0)
    noise = 100.0 * (labels == -1).sum() / len(labels)
    big = 100.0 * np.bincount(labels[labels >= 0]).max() / len(labels) if k else 0
    logger.info("clustering in %.1f min: %d clusters, %.1f%% noise, largest %.1f%%",
                (time.time() - t0) / 60, k, noise, big)
    return disp.astype(np.float32), labels.astype(np.int32)


def store(session, ids, disp, labels) -> int:
    """Write display coords + cluster into the table the dashboard reads.

    Only embedding_vec_3d and cluster_id are touched; embedding_tcn_128d keeps
    the encoder output.
    """
    sql = text("""
        INSERT INTO game_embeddings (battle_id, embedding_vec_3d, cluster_id, model_version)
        VALUES (:bid, :v3, :cid, :ver)
        ON CONFLICT (battle_id) DO UPDATE SET
            embedding_vec_3d = EXCLUDED.embedding_vec_3d,
            cluster_id = EXCLUDED.cluster_id,
            model_version = EXCLUDED.model_version
    """)
    ver = "feat-%s" % FEATURE_VERSION
    done = 0
    for s in range(0, len(ids), 5000):
        e = min(s + 5000, len(ids))
        session.execute(sql, [{
            "bid": ids[i], "v3": disp[i].tolist(),
            "cid": int(labels[i]) if labels[i] >= 0 else None, "ver": ver,
        } for i in range(s, e)])
        session.commit()
        done += e - s
    logger.info("stored %d rows as %s", done, ver)
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-sample", type=int, default=250000)
    ap.add_argument("--min-cluster-size", type=int, default=500)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()

    session = get_session(get_engine(os.environ["DATABASE_URL"]))
    WORK.mkdir(parents=True, exist_ok=True)
    bids = select_battles(session, args.corpus_sample)
    ids, vecs = extract(session, bids)
    if len(ids) < 1000:
        raise SystemExit("only %d vectors — too few to project" % len(ids))
    disp, labels = project_and_cluster(vecs, args.min_cluster_size)
    np.savez(WORK / "manifold.npz", ids=np.array(ids, dtype=object),
             disp=disp, labels=labels)
    if args.dry:
        logger.info("--dry: nothing written to the database")
        return
    store(session, ids, disp, labels)
    logger.info("DONE")


if __name__ == "__main__":
    main()
