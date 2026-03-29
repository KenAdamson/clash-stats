"""HDBSCAN clustering on UMAP embeddings with cluster profiling."""

import logging

import numpy as np
from hdbscan import HDBSCAN
from sqlalchemy import select
from sqlalchemy.orm import Session

from tracker.models import Battle
from tracker.ml.storage import GameEmbedding

logger = logging.getLogger(__name__)

# HDBSCAN parameters (ADR-003)
HDBSCAN_PARAMS = dict(
    min_cluster_size=10,
    min_samples=5,
    cluster_selection_method="eom",
)


def label_clusters(embeddings_15d: np.ndarray) -> np.ndarray:
    """Run HDBSCAN on 15-dim embeddings and return cluster labels.

    Args:
        embeddings_15d: Shape (n_games, 15).

    Returns:
        Array of cluster IDs. -1 means noise/outlier.
    """
    clusterer = HDBSCAN(**HDBSCAN_PARAMS)
    labels = clusterer.fit_predict(embeddings_15d)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int(np.sum(labels == -1))
    logger.info("HDBSCAN: %d clusters, %d noise points (%.1f%%)",
                n_clusters, n_noise, 100 * n_noise / len(labels))

    return labels


def profile_clusters(
    session: Session,
    feature_names: list[str] | None = None,
) -> list[dict]:
    """Generate per-cluster statistics.

    Returns a list of dicts with cluster_id, size, win_rate,
    and distinguishing feature values.
    """
    # Load all embeddings with cluster assignments
    rows = session.execute(
        select(
            GameEmbedding.battle_id,
            GameEmbedding.cluster_id,
            GameEmbedding.embedding_vec_3d,
        )
    ).all()

    if not rows:
        return []

    # Load battle results
    battle_ids = [r[0] for r in rows]
    results = session.execute(
        select(Battle.battle_id, Battle.result, Battle.corpus)
        .where(Battle.battle_id.in_(battle_ids))
    ).all()
    result_map = {r[0]: r[1] for r in results}
    corpus_map = {r[0]: r[2] for r in results}

    # Group by cluster
    clusters: dict[int, list] = {}
    for battle_id, cluster_id, vec_3d in rows:
        cid = cluster_id if cluster_id is not None else -1
        if cid not in clusters:
            clusters[cid] = []
        if vec_3d is None or len(vec_3d) != 3:
            continue
        xyz = vec_3d
        clusters[cid].append({
            "battle_id": battle_id,
            "result": result_map.get(battle_id, "unknown"),
            "corpus": corpus_map.get(battle_id, "unknown"),
            "x": float(xyz[0]),
            "y": float(xyz[1]),
            "z": float(xyz[2]),
        })

    profiles = []
    for cid in sorted(clusters.keys()):
        games = clusters[cid]
        wins = sum(1 for g in games if g["result"] == "win")
        personal = sum(1 for g in games if g["corpus"] == "personal")
        xs = [g["x"] for g in games]
        ys = [g["y"] for g in games]
        zs = [g["z"] for g in games]

        profiles.append({
            "cluster_id": cid,
            "size": len(games),
            "win_rate": wins / max(len(games), 1),
            "personal_count": personal,
            "centroid_x": float(np.mean(xs)),
            "centroid_y": float(np.mean(ys)),
            "centroid_z": float(np.mean(zs)),
            "label": "noise" if cid == -1 else f"cluster-{cid}",
        })

    return profiles
