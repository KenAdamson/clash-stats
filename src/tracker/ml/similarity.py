"""Similarity search on standardized feature vectors.

Reports two metrics per result:
- Percentile rank: "Top N%" — what fraction of all games are further away.
  Immediately human-readable. Top 1% = very close, 50% = mediocre.
- Gaussian kernel similarity: exp(-d²/2σ²) where σ = median distance.
  Natural [0, 1] scale adapted to the data distribution.
"""

import logging

import numpy as np
from sklearn.preprocessing import StandardScaler
from sqlalchemy import select
from sqlalchemy.orm import Session

from tracker.models import Battle, DeckCard
from tracker.archetypes import ARCHETYPES
from tracker.ml.storage import GameFeature, GameEmbedding, from_blob

logger = logging.getLogger(__name__)


def _enrich_results(session: Session, results: list[dict]) -> None:
    """Add battle metadata, opponent deck, and archetype to result dicts."""
    if not results:
        return

    result_ids = [r["battle_id"] for r in results]

    # Battle metadata
    battles = session.execute(
        select(
            Battle.battle_id, Battle.result, Battle.player_crowns,
            Battle.opponent_crowns, Battle.opponent_name,
            Battle.player_starting_trophies, Battle.battle_time,
            Battle.corpus,
        ).where(Battle.battle_id.in_(result_ids))
    ).all()

    meta = {b[0]: b for b in battles}
    for r in results:
        b = meta.get(r["battle_id"])
        if b:
            r["result"] = b[1]
            r["player_crowns"] = b[2]
            r["opponent_crowns"] = b[3]
            r["opponent_name"] = b[4]
            r["trophies"] = b[5]
            r["battle_time"] = b[6]
            r["corpus"] = b[7]

    # Add 3D embedding coordinates for visualization lines
    emb_rows = session.execute(
        select(GameEmbedding.battle_id, GameEmbedding.embedding_vec_3d)
        .where(GameEmbedding.battle_id.in_(result_ids))
    ).all()
    emb_map = {r[0]: r[1] for r in emb_rows if r[1] is not None}
    for r in results:
        coords = emb_map.get(r["battle_id"])
        if coords is not None:
            r["x"] = float(coords[0])
            r["y"] = float(coords[1])
            r["z"] = float(coords[2])
        else:
            r["x"] = r["y"] = r["z"] = None

    # Opponent deck cards and archetype
    deck_cards = session.execute(
        select(DeckCard.battle_id, DeckCard.card_name, DeckCard.card_variant)
        .where(DeckCard.battle_id.in_(result_ids))
        .where(DeckCard.is_player_deck == 0)
        .order_by(DeckCard.battle_id, DeckCard.card_name)
    ).all()

    decks_by_battle: dict[str, list[dict]] = {}
    for bid, name, variant in deck_cards:
        decks_by_battle.setdefault(bid, []).append({"name": name, "variant": variant})

    for r in results:
        cards = decks_by_battle.get(r["battle_id"], [])
        r["opponent_deck"] = [c["name"] for c in cards]

        card_names = {c["name"] for c in cards}
        archetype = "Unknown"
        for arch, win_conditions in ARCHETYPES.items():
            if any(wc in card_names for wc in win_conditions):
                archetype = arch
                break
        r["archetype"] = archetype


def find_similar(
    session: Session,
    battle_id: str,
    k: int = 10,
) -> dict:
    """Find the k most similar games to a given battle.

    Uses Euclidean distance on StandardScaler'd feature vectors.
    Reports percentile rank and Gaussian kernel similarity.

    Args:
        session: DB session.
        battle_id: The reference battle to find neighbors for.
        k: Number of similar games per category.

    Returns:
        Dict with 'corpus' and 'personal' lists of similar games.
    """
    # Load all feature vectors
    rows = session.execute(
        select(GameFeature.battle_id, GameFeature.feature_vector)
    ).all()

    if not rows:
        return {"corpus": [], "personal": []}

    # Find the reference vector index
    ref_idx = None
    ids = []
    vectors = []
    for i, (bid, fv) in enumerate(rows):
        vec = from_blob(fv, -1)
        if bid == battle_id:
            ref_idx = i
        ids.append(bid)
        vectors.append(vec)

    if ref_idx is None:
        logger.warning("No feature vector found for battle %s", battle_id)
        return {"corpus": [], "personal": []}

    matrix = np.stack(vectors)

    # Standardize features so each dimension contributes equally
    scaler = StandardScaler()
    scaled = scaler.fit_transform(matrix)
    ref_scaled = scaled[ref_idx]

    # Euclidean distances from reference
    distances = np.linalg.norm(scaled - ref_scaled, axis=1)

    # Percentile rank: fraction of games that are further away (higher = more similar)
    # Distance of 0 (self) → 100th percentile; max distance → 0th percentile
    n = len(distances)
    ranks = np.zeros(n)
    sorted_indices = np.argsort(distances)
    for rank, idx in enumerate(sorted_indices):
        ranks[idx] = 1.0 - (rank / (n - 1))  # 1.0 = closest, 0.0 = furthest

    # Gaussian kernel: exp(-d²/2σ²), σ = median distance (adaptive bandwidth)
    sigma = float(np.median(distances[distances > 0]))
    gaussian = np.exp(-distances**2 / (2 * sigma**2))

    # Load cluster IDs
    cluster_rows = session.execute(
        select(GameEmbedding.battle_id, GameEmbedding.cluster_id)
    ).all()
    cluster_map = {r[0]: r[1] for r in cluster_rows}

    # Load corpus labels
    corpus_rows = session.execute(
        select(Battle.battle_id, Battle.corpus)
        .where(Battle.battle_id.in_(ids))
    ).all()
    corpus_map = {r[0]: r[1] for r in corpus_rows}

    # Sort by distance (ascending = most similar first), split by corpus
    indices = np.argsort(distances)
    corpus_results = []
    personal_results = []

    for idx in indices:
        if ids[idx] == battle_id:
            continue

        entry = {
            "battle_id": ids[idx],
            "percentile": float(ranks[idx]),
            "similarity": float(gaussian[idx]),
            "cluster_id": cluster_map.get(ids[idx]),
        }

        corpus = corpus_map.get(ids[idx], "unknown")
        if corpus == "personal":
            if len(personal_results) < k:
                personal_results.append(entry)
        else:
            if len(corpus_results) < k:
                corpus_results.append(entry)

        if len(corpus_results) >= k and len(personal_results) >= k:
            break

    _enrich_results(session, corpus_results)
    _enrich_results(session, personal_results)

    # Reference point 3D coordinates for visualization lines
    ref_vec = session.execute(
        select(GameEmbedding.embedding_vec_3d)
        .where(GameEmbedding.battle_id == battle_id)
    ).scalar_one_or_none()
    ref_coords = None
    if ref_vec is not None and len(ref_vec) == 3:
        ref_coords = {"x": float(ref_vec[0]), "y": float(ref_vec[1]), "z": float(ref_vec[2])}

    return {
        "corpus": corpus_results,
        "personal": personal_results,
        "ref_coords": ref_coords,
    }
