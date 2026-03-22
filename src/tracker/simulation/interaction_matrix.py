"""Card interaction matrix and co-occurrence analysis (ADR-002 §2).

Builds P(win | opponent_has_card_X) across the corpus and detects
sub-archetypes via card co-occurrence clustering.

Functions accept pre-aggregated SimulationData from battles_repo
to avoid loading the full battles table per function call.
Legacy session-based signatures are preserved for CLI callers
that don't use the full simulation runner.
"""

import logging
from collections import Counter, defaultdict
from typing import Optional

from scipy.stats import beta as beta_dist
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)


def build_card_interaction_matrix(
    session_or_data=None,
    *,
    sim_data=None,
    corpus: Optional[str] = None,
    player_tag: Optional[str] = None,
    min_appearances: int = 10,
) -> dict:
    """Build P(win | opponent_has_card_X) for every opponent card.

    Args:
        session_or_data: SQLAlchemy session (legacy) or SimulationData.
        sim_data: Pre-aggregated SimulationData (keyword, preferred).
        min_appearances: Minimum battles with card to include.

    Returns:
        Dict mapping card_name -> stats dict.
    """
    from tracker.simulation.battles_repo import SimulationData, compute_simulation_data

    if sim_data is not None:
        pass
    elif isinstance(session_or_data, SimulationData):
        sim_data = session_or_data
    elif session_or_data is not None:
        sim_data = compute_simulation_data(session_or_data, corpus=corpus, player_tag=player_tag)
    else:
        raise ValueError("Either session or sim_data required")

    matrix = {}
    all_cards = set(sim_data.card_wins.keys()) | set(sim_data.card_losses.keys())

    for card_name in all_cards:
        wins = sim_data.card_wins.get(card_name, 0)
        losses = sim_data.card_losses.get(card_name, 0)
        total = wins + losses
        if total < min_appearances:
            continue

        a = wins + 1
        b = losses + 1
        ci_low, ci_high = beta_dist.ppf([0.025, 0.975], a, b)

        matrix[card_name] = {
            "wins": wins,
            "losses": losses,
            "total": total,
            "win_rate": wins / total,
            "expected": a / (a + b),
            "ci_low": ci_low,
            "ci_high": ci_high,
        }

    return dict(sorted(matrix.items(), key=lambda x: x[1]["win_rate"]))


def build_card_cooccurrence(
    sim_data=None,
    session: Session | None = None,
    corpus: Optional[str] = None,
    min_battles: int = 20,
) -> dict:
    """Build card co-occurrence matrix for opponent decks.

    Args:
        sim_data: Pre-aggregated SimulationData (preferred).
        session: SQLAlchemy session (legacy fallback).
        min_battles: Minimum co-occurrences to include a pair.

    Returns:
        Dict with pair_counts, card_counts, total_decks.
    """
    if sim_data is None:
        from tracker.simulation.battles_repo import compute_simulation_data
        sim_data = compute_simulation_data(session, corpus=corpus)

    filtered_pairs = {
        pair: count
        for pair, count in sim_data.pair_counts.items()
        if count >= min_battles
    }

    return {
        "pair_counts": filtered_pairs,
        "card_counts": dict(sim_data.card_counts),
        "total_decks": sim_data.total_battles,
    }


def detect_sub_archetypes(
    win_condition: str,
    sim_data=None,
    session: Session | None = None,
    corpus: Optional[str] = None,
    min_cluster_size: int = 15,
    similarity_threshold: float = 0.55,
) -> list[dict]:
    """Detect sub-archetypes within a win-condition archetype.

    Uses greedy Jaccard clustering on support cards.

    Args:
        win_condition: The win-condition card name (e.g., 'Hog Rider').
        sim_data: Pre-aggregated SimulationData (preferred).
        session: SQLAlchemy session (legacy fallback).
        min_cluster_size: Minimum decks to form a sub-archetype.
        similarity_threshold: Jaccard threshold for clustering.

    Returns:
        List of sub-archetype dicts.
    """
    if sim_data is None:
        from tracker.simulation.battles_repo import compute_simulation_data
        sim_data = compute_simulation_data(session, corpus=corpus)

    # Collect decks containing the win condition from all archetypes
    decks_with_wc = []
    for archetype, deck_list in sim_data.archetype_decks.items():
        for d in deck_list:
            if win_condition in d["card_names"]:
                support = frozenset(c for c in d["card_names"] if c != win_condition)
                decks_with_wc.append({
                    "support": support,
                    "full_deck": tuple(d["card_names"]),
                    "result": d["result"],
                    "elixir": d["elixir"],
                })

    if len(decks_with_wc) < min_cluster_size:
        return []

    logger.info(
        "Clustering %d %s decks into sub-archetypes.",
        len(decks_with_wc), win_condition,
    )

    # Group by exact deck composition
    deck_groups: dict[tuple, list[dict]] = defaultdict(list)
    for d in decks_with_wc:
        deck_groups[d["full_deck"]].append(d)

    sorted_groups = sorted(deck_groups.items(), key=lambda x: -len(x[1]))

    # Greedy Jaccard clustering
    clusters: list[dict] = []
    for deck_tuple, group in sorted_groups:
        support = frozenset(c for c in deck_tuple if c != win_condition)
        best_cluster = None
        best_sim = 0.0

        for cluster in clusters:
            sim = _jaccard(support, cluster["support_union"])
            if sim > best_sim:
                best_sim = sim
                best_cluster = cluster

        if best_cluster and best_sim >= similarity_threshold:
            best_cluster["decks"].extend(group)
            best_cluster["support_union"] |= support
            best_cluster["deck_variants"][deck_tuple] += len(group)
        else:
            clusters.append({
                "decks": list(group),
                "support_union": set(support),
                "deck_variants": Counter({deck_tuple: len(group)}),
            })

    results = []
    for cluster in clusters:
        if len(cluster["decks"]) < min_cluster_size:
            continue

        wins = sum(1 for d in cluster["decks"] if d["result"] == "win")
        total = len(cluster["decks"])
        avg_elixir = sum(d["elixir"] for d in cluster["decks"]) / total

        card_freq: Counter = Counter()
        for d in cluster["decks"]:
            for c in d["support"]:
                card_freq[c] += 1

        signature = [
            card for card, count in card_freq.most_common()
            if count / total >= 0.60
        ]

        most_common_deck = cluster["deck_variants"].most_common(1)[0][0]

        results.append({
            "signature_cards": signature,
            "sample_deck": list(most_common_deck),
            "count": total,
            "win_rate": wins / total if total > 0 else 0.0,
            "avg_elixir": round(avg_elixir / 8, 1),
            "variants": len(cluster["deck_variants"]),
        })

    results.sort(key=lambda x: -x["count"])
    return results


def _jaccard(a: frozenset, b: set) -> float:
    """Jaccard similarity between two sets."""
    if not a and not b:
        return 1.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union > 0 else 0.0
