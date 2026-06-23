"""Simulation runner — computes and caches Monte Carlo results.

Results are written to JSON on the data volume so the Flask dashboard
can serve them without re-running the simulations on every request.

Uses BattlesRepository for single-pass paginated data loading —
constant ~5MB memory regardless of corpus size (was ~41GB before).
"""

import json
import logging
import os
from pathlib import Path
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from tracker.simulation.battles_repo import compute_simulation_data
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

# Set before forking the sub-archetype pool so workers inherit the (large)
# SimulationData via copy-on-write rather than pickling it per task.
_SUBARCH_SIM_DATA = None


def _subarch_worker(args):
    """Detect sub-archetypes for one win condition in a forked worker."""
    win_condition, min_cluster_size, similarity_threshold = args
    subs = detect_sub_archetypes(
        win_condition,
        sim_data=_SUBARCH_SIM_DATA,
        min_cluster_size=min_cluster_size,
        similarity_threshold=similarity_threshold,
    )
    return win_condition, subs


def _detect_sub_archetypes_parallel(
    corpus_data, win_conditions, *, min_cluster_size, similarity_threshold,
):
    """Run detect_sub_archetypes across win conditions in parallel processes.

    Each win condition is independent, so we fan them across a fork-based
    process pool. Workers inherit corpus_data via copy-on-write (no pickling),
    but each worker's clustering allocates several GiB that is NOT shared, so
    peak memory scales with worker count. Defaults to 2 workers (memory-safe;
    6 OOM-thrashed a 62 GiB host); tune with SIM_MAX_WORKERS. Wall time is
    bounded by the single longest archetype anyway, so few workers cost little.
    Falls back to sequential on any pool failure.
    """
    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor, as_completed

    global _SUBARCH_SIM_DATA
    # Each worker runs HDBSCAN/UMAP clustering, which allocates large arrays that
    # are NOT copy-on-write-shared (and Python refcount writes defeat COW on the
    # inherited corpus_data anyway). So peak memory scales ~linearly with worker
    # count: 6 workers was ~36 GiB and OOM-thrashed the host. Default to a
    # memory-safe 2; allow SIM_MAX_WORKERS to tune up on a box with headroom.
    _cpu_cap = min(len(win_conditions), 6, max(2, (os.cpu_count() or 4) - 2))
    _env = os.environ.get("SIM_MAX_WORKERS")
    if _env:
        try:
            max_workers = max(1, min(int(_env), len(win_conditions)))
        except ValueError:
            logger.warning("Invalid SIM_MAX_WORKERS=%r; falling back to default.", _env)
            max_workers = min(_cpu_cap, 2)
    else:
        max_workers = min(_cpu_cap, 2)
    sub_archetypes: dict = {}
    try:
        ctx = mp.get_context("fork")
        _SUBARCH_SIM_DATA = corpus_data
        with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as ex:
            futures = {
                ex.submit(
                    _subarch_worker, (wc, min_cluster_size, similarity_threshold)
                ): wc
                for wc in win_conditions
            }
            for fut in as_completed(futures):
                wc, subs = fut.result()
                if subs:
                    sub_archetypes[wc] = subs
    except Exception as e:
        logger.warning(
            "Parallel sub-archetype detection failed (%s); running sequentially.", e
        )
        sub_archetypes = {}
        for wc in win_conditions:
            subs = detect_sub_archetypes(
                wc, sim_data=corpus_data,
                min_cluster_size=min_cluster_size,
                similarity_threshold=similarity_threshold,
            )
            if subs:
                sub_archetypes[wc] = subs
    finally:
        _SUBARCH_SIM_DATA = None
    return sub_archetypes


def run_full_simulation(
    session: Session,
    player_tag: str | None = None,
) -> dict:
    """Run all simulations and cache results.

    Single-pass data loading via BattlesRepository — all downstream
    functions share pre-aggregated SimulationData instead of each
    independently scanning the full battles table.

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

    # Single pass: load all battles, aggregate card stats + archetypes
    logger.info("Loading corpus data (single pass, paginated)...")
    corpus_data = compute_simulation_data(session)

    # 1. Matchup posteriors (corpus-wide) — uses pre-aggregated data
    logger.info("Computing corpus-wide matchup posteriors...")
    corpus_posteriors = compute_matchup_posteriors(
        sim_data=corpus_data, min_battles=5, use_sub_archetypes=True,
    )
    results["corpus_matchups"] = corpus_posteriors
    results["corpus_threats"] = compute_threat_ranking(corpus_posteriors, min_battles=10)

    # 2. Personal matchup posteriors (separate pass, much smaller dataset)
    if player_tag:
        logger.info("Loading personal data for %s...", player_tag)
        personal_data = compute_simulation_data(session, player_tag=player_tag)
        personal_posteriors = compute_matchup_posteriors(
            sim_data=personal_data, min_battles=3, use_sub_archetypes=True,
        )
        results["personal_matchups"] = personal_posteriors
        results["personal_threats"] = compute_threat_ranking(
            personal_posteriors, min_battles=5
        )

    # 3. Card interaction matrix (corpus-wide) — uses pre-aggregated data
    logger.info("Building card interaction matrix...")
    card_matrix = build_card_interaction_matrix(
        sim_data=corpus_data, min_appearances=10,
    )
    results["card_interactions"] = card_matrix

    # 4. Personal card interactions
    if player_tag:
        logger.info("Building personal card interactions...")
        personal_matrix = build_card_interaction_matrix(
            sim_data=personal_data, min_appearances=3,
        )
        results["personal_card_interactions"] = personal_matrix

    # 5. Sub-archetype breakdowns — uses pre-collected deck lists from corpus_data
    logger.info("Detecting sub-archetypes for major win conditions...")
    sub_archetypes = _detect_sub_archetypes_parallel(
        corpus_data, MAJOR_WIN_CONDITIONS,
        min_cluster_size=10, similarity_threshold=0.55,
    )
    results["sub_archetypes"] = sub_archetypes

    # 6. Card co-occurrence — uses pre-aggregated pair counts
    logger.info("Building card co-occurrence data...")
    cooccurrence = build_card_cooccurrence(sim_data=corpus_data, min_battles=20)
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
    except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
        # UnicodeDecodeError caught a fiasco-era partial-write cache file
        # with null bytes in the middle of the JSON — surfaced to /api/simulation
        # as 500s until 2026-05-29.
        logger.warning("Failed to read cached simulation results: %s", e)
        return None


def _save_results(results: dict) -> None:
    """Save simulation results to disk."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "latest.json"

    def _default(obj):
        if hasattr(obj, "item"):
            return obj.item()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    path.write_text(json.dumps(results, indent=2, default=_default))
    logger.info("Simulation results cached to %s", path)
