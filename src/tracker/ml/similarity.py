"""Similarity search via VectorChord kNN on native vector columns.

Uses PostgreSQL-side Euclidean distance (<->) with VectorChord's
vchordrq index for approximate nearest neighbor search. No Python-side
distance computation — queries return pre-ranked results in ~10ms
instead of loading 500K+ vectors into memory.

Reports:
- Distance: raw L2 distance from VectorChord
- Similarity: exp(-d²/2σ²) Gaussian kernel (σ = median of top-k distances)
"""

import logging

import numpy as np
from sqlalchemy import select, text as sa_text, func
from sqlalchemy.orm import Session

from tracker.models import Battle, DeckCard
from tracker.archetypes import ARCHETYPES
from tracker.ml.storage import GameFeature, GameEmbedding

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

    Uses VectorChord's kNN index (<-> operator) for server-side
    Euclidean distance on native vector columns. Returns top-k
    results per category (personal/corpus) in ~10ms instead of
    loading 500K+ vectors into Python.

    Args:
        session: DB session.
        battle_id: The reference battle to find neighbors for.
        k: Number of similar games per category.

    Returns:
        Dict with 'corpus' and 'personal' lists of similar games.
    """
    # Get the reference vector
    ref_row = session.execute(
        select(GameFeature.feature_vec)
        .where(GameFeature.battle_id == battle_id)
    ).one_or_none()

    if ref_row is None or ref_row[0] is None:
        logger.warning("No feature vector found for battle %s", battle_id)
        return {"corpus": [], "personal": []}

    ref_vec = ref_row[0]
    # Format as pgvector string: '[f1,f2,...,fn]'
    ref_str = '[' + ','.join(str(float(v)) for v in ref_vec) + ']'

    # Phase 1: Pure kNN — uses VectorChord index, ~11ms for 544K vectors.
    # Fetch more than k to have enough after corpus/personal split.
    fetch_limit = k * 10
    knn_rows = session.execute(
        sa_text("""
            SELECT battle_id,
                   feature_vec <-> CAST(:ref AS vector) AS distance
            FROM game_features
            WHERE battle_id != :bid
              AND feature_vec IS NOT NULL
            ORDER BY feature_vec <-> CAST(:ref AS vector)
            LIMIT :lim
        """),
        {"ref": ref_str, "bid": battle_id, "lim": fetch_limit},
    ).all()

    if not knn_rows:
        return {"corpus": [], "personal": []}

    # Phase 2: Enrich only the top-k*10 results with corpus/cluster metadata
    neighbor_ids = [r[0] for r in knn_rows]
    dist_map = {r[0]: r[1] for r in knn_rows}

    corpus_rows = session.execute(
        select(Battle.battle_id, Battle.corpus)
        .where(Battle.battle_id.in_(neighbor_ids))
    ).all()
    corpus_map = {r[0]: r[1] for r in corpus_rows}

    cluster_rows = session.execute(
        select(GameEmbedding.battle_id, GameEmbedding.cluster_id)
        .where(GameEmbedding.battle_id.in_(neighbor_ids))
    ).all()
    cluster_map = {r[0]: r[1] for r in cluster_rows}

    # Phase 3: Split into corpus/personal, take top-k of each
    corpus_results_raw = []
    personal_results_raw = []
    for bid in neighbor_ids:
        corpus = corpus_map.get(bid, "unknown")
        entry = (bid, dist_map[bid], cluster_map.get(bid))
        if corpus == "personal":
            if len(personal_results_raw) < k:
                personal_results_raw.append(entry)
        else:
            if len(corpus_results_raw) < k:
                corpus_results_raw.append(entry)
        if len(corpus_results_raw) >= k and len(personal_results_raw) >= k:
            break

    # Compute Gaussian kernel similarity from distances
    all_distances = [e[1] for e in corpus_results_raw] + [e[1] for e in personal_results_raw]
    if all_distances:
        sigma = float(np.median(all_distances)) or 1.0
    else:
        sigma = 1.0

    def _build_results(raw):
        results = []
        for bid, distance, cluster_id in raw:
            similarity = float(np.exp(-distance**2 / (2 * sigma**2)))
            results.append({
                "battle_id": bid,
                "distance": float(distance),
                "similarity": similarity,
                "percentile": None,
                "cluster_id": cluster_id,
            })
        return results

    corpus_results = _build_results(corpus_results_raw)
    personal_results = _build_results(personal_results_raw)

    _enrich_results(session, corpus_results)
    _enrich_results(session, personal_results)

    # Reference point 3D coordinates for visualization lines
    ref_emb = session.execute(
        select(GameEmbedding.embedding_vec_3d)
        .where(GameEmbedding.battle_id == battle_id)
    ).scalar_one_or_none()
    ref_coords = None
    if ref_emb is not None and len(ref_emb) == 3:
        ref_coords = {"x": float(ref_emb[0]), "y": float(ref_emb[1]), "z": float(ref_emb[2])}

    return {
        "corpus": corpus_results,
        "personal": personal_results,
        "ref_coords": ref_coords,
    }
