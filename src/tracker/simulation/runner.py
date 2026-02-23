"""Simulation runner — computes and caches Monte Carlo results.

Results are written to JSON on the data volume so the Flask dashboard
can serve them without re-running the simulations on every request.
"""

import json
import logging
import os
from pathlib import Path
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from tracker.simulation.interaction_matrix import (
    build_card_interaction_matrix,
    build_card_cooccurrence,
    detect_sub_archetypes,
)
from tracker.simulation.matchup_model import (
    compute_matchup_posteriors,
    compute_threat_ranking,
)

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(os.environ.get(
    "SIM_RESULTS_DIR", "/app/data/simulation"
))

# Win conditions worth sub-archetype analysis (high-frequency in corpus)
MAJOR_WIN_CONDITIONS = [
    "Hog Rider", "Mega Knight", "Golem", "P.E.K.K.A",
    "Royal Giant", "Lava Hound", "Balloon", "Miner",
    "Graveyard", "Goblin Barrel", "X-Bow", "Monk",
    "Archer Queen", "Skeleton King",
]


def run_full_simulation(
    session: Session,
    player_tag: str | None = None,
) -> dict:
    """Run all simulations and cache results.

    Args:
        session: SQLAlchemy session.
        player_tag: Personal player tag for personal matchup analysis.

    Returns:
        Dict with all simulation results.
    """
    results = {
        "computed_at": datetime.now(timezone.utc).isoformat(),
        "player_tag": player_tag,
    }

    # 1. Matchup posteriors (corpus-wide)
    logger.info("Computing corpus-wide matchup posteriors...")
    corpus_posteriors = compute_matchup_posteriors(
        session, corpus=None, min_battles=5, use_sub_archetypes=True,
    )
    results["corpus_matchups"] = corpus_posteriors
    results["corpus_threats"] = compute_threat_ranking(corpus_posteriors, min_battles=10)

    # 2. Personal matchup posteriors (if player_tag provided)
    if player_tag:
        logger.info("Computing personal matchup posteriors for %s...", player_tag)
        personal_posteriors = compute_matchup_posteriors(
            session, player_tag=player_tag, min_battles=3,
            use_sub_archetypes=True,
        )
        results["personal_matchups"] = personal_posteriors
        results["personal_threats"] = compute_threat_ranking(
            personal_posteriors, min_battles=5
        )

    # 3. Card interaction matrix (corpus-wide)
    logger.info("Building card interaction matrix...")
    card_matrix = build_card_interaction_matrix(
        session, min_appearances=10,
    )
    results["card_interactions"] = card_matrix

    # 4. Personal card interactions
    if player_tag:
        logger.info("Building personal card interactions...")
        personal_matrix = build_card_interaction_matrix(
            session, player_tag=player_tag, min_appearances=3,
        )
        results["personal_card_interactions"] = personal_matrix

    # 5. Sub-archetype breakdowns for major win conditions
    logger.info("Detecting sub-archetypes for major win conditions...")
    sub_archetypes = {}
    for wc in MAJOR_WIN_CONDITIONS:
        subs = detect_sub_archetypes(
            session, wc, min_cluster_size=10, similarity_threshold=0.55,
        )
        if subs:
            sub_archetypes[wc] = subs
    results["sub_archetypes"] = sub_archetypes

    # 6. Card co-occurrence
    logger.info("Building card co-occurrence data...")
    cooccurrence = build_card_cooccurrence(session, min_battles=20)
    # Don't serialize the full pair matrix — just top correlations
    top_pairs = sorted(
        cooccurrence["pair_counts"].items(), key=lambda x: -x[1]
    )[:200]
    results["top_card_pairs"] = [
        {"cards": list(pair), "count": count}
        for pair, count in top_pairs
    ]
    results["total_decks_analyzed"] = cooccurrence["total_decks"]

    # Cache to disk
    _save_results(results)

    logger.info(
        "Simulation complete: %d archetype matchups, %d card interactions, "
        "%d sub-archetype groups.",
        len(corpus_posteriors),
        len(card_matrix),
        sum(len(v) for v in sub_archetypes.values()),
    )

    return results


def get_cached_results() -> dict | None:
    """Load cached simulation results from disk.

    Returns:
        Cached results dict, or None if no cache exists.
    """
    path = RESULTS_DIR / "latest.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read cached simulation results: %s", e)
        return None


def _save_results(results: dict) -> None:
    """Save simulation results to disk."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "latest.json"

    # Custom serializer for numpy types
    def _default(obj):
        if hasattr(obj, "item"):
            return obj.item()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    path.write_text(json.dumps(results, indent=2, default=_default))
    logger.info("Simulation results cached to %s", path)
