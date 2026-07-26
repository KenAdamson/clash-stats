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

    import time
    _t0 = time.perf_counter()

    # Collect unique decks containing the win condition. archetype_decks is
    # pre-aggregated by exact composition (one entry per unique deck with its
    # win/loss counts), so no per-battle dedup pass is needed here.
    decks_with_wc = []
    matched_battles = 0
    for deck_map in sim_data.archetype_decks.values():
        for deck_tuple, (wins, losses, elixir) in deck_map.items():
            if win_condition in deck_tuple:
                count = wins + losses
                support = frozenset(c for c in deck_tuple if c != win_condition)
                decks_with_wc.append({
                    "support": support,
                    "full_deck": deck_tuple,
                    "wins": wins,
                    "count": count,
                    "elixir": elixir,
                })
                matched_battles += count
    _t_collect = time.perf_counter() - _t0

    if matched_battles < min_cluster_size:
        return []

    logger.info(
        "Clustering %d unique %s decks (%d battles) into sub-archetypes.",
        len(decks_with_wc), win_condition, matched_battles,
    )

    # Already unique per composition; order by battle frequency (descending) to
    # preserve the previous group-size ordering.
    _t1 = time.perf_counter()
    sorted_groups = sorted(decks_with_wc, key=lambda d: -d["count"])
    _t_dedup = time.perf_counter() - _t1
    _n_unique = len(sorted_groups)

    # Greedy Jaccard clustering with an inverted-index candidate filter.
    #
    # The naive form compares each deck against every existing cluster — O(unique
    # x clusters), which is ~99.7% of total sim runtime (profiled 2026-05-31:
    # 9872s for Hog Rider alone). But Jaccard >= similarity_threshold has a hard
    # lower bound on shared cards: a deck's support is always 7 cards (8-card deck
    # minus the win condition), so for J = |A∩B| / |A∪B| >= 0.55 the intersection
    # must be at least ceil(0.55*(7+7) / 1.55) = 5 cards (larger cluster unions
    # require even more). So any cluster sharing < MIN_SHARED cards with the deck
    # cannot possibly merge and need not be scored.
    #
    # card_to_clusters maps each card -> indices of clusters whose support_union
    # contains it. For each deck we tally candidate clusters by shared-card count
    # via the index, score only those with >= MIN_SHARED shared cards, and pick
    # the best exactly as the naive loop would. This is output-identical, just
    # without the dead comparisons. (Verified against the un-pruned cache.)
    _t2 = time.perf_counter()
    # Smallest possible support is 7 cards (8-card deck minus the single win
    # condition). For J = |A∩B|/|A∪B| >= t with |A| = 7 and any union |B| >= 7,
    # the intersection must be at least ceil(14t / (1+t)) cards (t=0.55 -> 5).
    # Larger unions require more, so this is a safe lower bound for all clusters.
    import math
    min_shared = max(1, math.ceil(14 * similarity_threshold / (1 + similarity_threshold)))
    clusters: list[dict] = []
    card_to_clusters: dict[str, set] = defaultdict(set)
    for d in sorted_groups:
        support = d["support"]
        deck_tuple = d["full_deck"]

        # Tally shared-card counts against candidate clusters via the index.
        cand_counts: dict[int, int] = defaultdict(int)
        for c in support:
            for ci in card_to_clusters.get(c, ()):
                cand_counts[ci] += 1

        best_idx = -1
        best_sim = 0.0
        # Iterate in ascending cluster index to match the naive loop's
        # first-max tie-breaking (strict >).
        for ci in sorted(cand_counts):
            if cand_counts[ci] < min_shared:
                continue
            sim = _jaccard(support, clusters[ci]["support_union"])
            if sim > best_sim:
                best_sim = sim
                best_idx = ci

        if best_idx >= 0 and best_sim >= similarity_threshold:
            cluster = clusters[best_idx]
            cluster["members"].append(d)
            new_cards = support - cluster["support_union"]
            cluster["support_union"] |= support
            cluster["deck_variants"][deck_tuple] += d["count"]
            for c in new_cards:
                card_to_clusters[c].add(best_idx)
        else:
            new_idx = len(clusters)
            clusters.append({
                "members": [d],
                "support_union": set(support),
                "deck_variants": Counter({deck_tuple: d["count"]}),
            })
            for c in support:
                card_to_clusters[c].add(new_idx)
    _t_cluster = time.perf_counter() - _t2
    _n_clusters = len(clusters)

    _t3 = time.perf_counter()
    results = []
    for cluster in clusters:
        total = sum(m["count"] for m in cluster["members"])
        if total < min_cluster_size:
            continue

        wins = sum(m["wins"] for m in cluster["members"])
        # elixir is constant per composition, so weight each unique deck's elixir
        # by how many battles it represents to reproduce the per-battle average.
        avg_elixir = sum(m["elixir"] * m["count"] for m in cluster["members"]) / total

        card_freq: Counter = Counter()
        for m in cluster["members"]:
            for c in m["support"]:
                card_freq[c] += m["count"]

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
    _t_agg = time.perf_counter() - _t3
    logger.info(
        "%s profile: collect=%.1fs dedup=%.1fs cluster=%.1fs agg=%.1fs "
        "| %d matched, %d unique compositions, %d clusters",
        win_condition, _t_collect, _t_dedup, _t_cluster, _t_agg,
        matched_battles, _n_unique, _n_clusters,
    )
    return results


def _jaccard(a: frozenset, b: set) -> float:
    """Jaccard similarity between two sets."""
    if not a and not b:
        return 1.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union > 0 else 0.0
