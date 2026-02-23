"""Card interaction matrix and co-occurrence analysis (ADR-002 §2).

Builds P(win | opponent_has_card_X) across the corpus and detects
sub-archetypes via card co-occurrence clustering.
"""

import json
import logging
from collections import Counter, defaultdict
from typing import Optional

import numpy as np
from scipy.stats import beta as beta_dist
from sqlalchemy import select
from sqlalchemy.orm import Session

from tracker.models import Battle

logger = logging.getLogger(__name__)

# Cards that appear in >50% of decks are noise for co-occurrence
# (e.g., Log, Arrows) — dynamically computed per corpus
HIGH_FREQUENCY_THRESHOLD = 0.50


def build_card_interaction_matrix(
    session: Session,
    corpus: Optional[str] = None,
    player_tag: Optional[str] = None,
    min_appearances: int = 10,
) -> dict:
    """Build P(win | opponent_has_card_X) for every opponent card.

    Args:
        session: SQLAlchemy session.
        corpus: Filter by corpus type (e.g., 'personal', 'top_ladder').
            None = all corpora.
        player_tag: Filter to battles involving this player tag.
        min_appearances: Minimum battles with card to include.

    Returns:
        Dict mapping card_name -> {
            'wins': int, 'losses': int, 'total': int,
            'win_rate': float, 'ci_low': float, 'ci_high': float,
            'appearances': int
        }
    """
    stmt = select(
        Battle.opponent_deck, Battle.result
    ).where(
        Battle.battle_type.in_(["PvP", "pathOfLegend"]),
        Battle.result.in_(["win", "loss"]),
    )
    if corpus:
        stmt = stmt.where(Battle.corpus == corpus)
    if player_tag:
        tag_clean = player_tag.lstrip("#")
        stmt = stmt.where(Battle.player_tag.like(f"%{tag_clean}%"))

    rows = session.execute(stmt).all()
    logger.info("Building interaction matrix from %d battles.", len(rows))

    card_stats: dict[str, dict] = defaultdict(lambda: {"wins": 0, "losses": 0})

    for opponent_deck_json, result in rows:
        if not opponent_deck_json:
            continue
        try:
            deck = json.loads(opponent_deck_json)
        except (json.JSONDecodeError, TypeError):
            continue

        card_names = {card.get("name") for card in deck if card.get("name")}
        for card_name in card_names:
            if result == "win":
                card_stats[card_name]["wins"] += 1
            else:
                card_stats[card_name]["losses"] += 1

    matrix = {}
    for card_name, stats in card_stats.items():
        total = stats["wins"] + stats["losses"]
        if total < min_appearances:
            continue

        # Beta posterior: Beta(wins + 1, losses + 1)
        a = stats["wins"] + 1
        b = stats["losses"] + 1
        ci_low, ci_high = beta_dist.ppf([0.025, 0.975], a, b)

        matrix[card_name] = {
            "wins": stats["wins"],
            "losses": stats["losses"],
            "total": total,
            "win_rate": stats["wins"] / total,
            "expected": a / (a + b),
            "ci_low": ci_low,
            "ci_high": ci_high,
        }

    return dict(sorted(matrix.items(), key=lambda x: x[1]["win_rate"]))


def build_card_cooccurrence(
    session: Session,
    corpus: Optional[str] = None,
    min_battles: int = 20,
) -> dict:
    """Build card co-occurrence matrix for opponent decks.

    For every pair of opponent cards (A, B), counts how often they
    appear together. This is the raw signal for sub-archetype detection.

    Args:
        session: SQLAlchemy session.
        corpus: Filter by corpus type.
        min_battles: Minimum co-occurrences to include a pair.

    Returns:
        Dict with:
            'pair_counts': {(card_a, card_b): count}
            'card_counts': {card_name: count}
            'total_decks': int
    """
    stmt = select(Battle.opponent_deck).where(
        Battle.battle_type.in_(["PvP", "pathOfLegend"]),
        Battle.result.in_(["win", "loss"]),
    )
    if corpus:
        stmt = stmt.where(Battle.corpus == corpus)

    rows = session.execute(stmt).all()

    card_counts: Counter = Counter()
    pair_counts: Counter = Counter()
    total_decks = 0

    for (opponent_deck_json,) in rows:
        if not opponent_deck_json:
            continue
        try:
            deck = json.loads(opponent_deck_json)
        except (json.JSONDecodeError, TypeError):
            continue

        card_names = sorted({card.get("name") for card in deck if card.get("name")})
        if len(card_names) < 2:
            continue

        total_decks += 1
        for name in card_names:
            card_counts[name] += 1

        # All pairs
        for i, a in enumerate(card_names):
            for b in card_names[i + 1:]:
                pair_counts[(a, b)] += 1

    # Filter to significant pairs
    filtered_pairs = {
        pair: count
        for pair, count in pair_counts.items()
        if count >= min_battles
    }

    return {
        "pair_counts": filtered_pairs,
        "card_counts": dict(card_counts),
        "total_decks": total_decks,
    }


def detect_sub_archetypes(
    session: Session,
    win_condition: str,
    corpus: Optional[str] = None,
    min_cluster_size: int = 15,
    similarity_threshold: float = 0.55,
) -> list[dict]:
    """Detect sub-archetypes within a win-condition archetype using
    card co-occurrence clustering.

    Uses agglomerative clustering on Jaccard similarity of support cards
    (all cards except the win condition itself).

    Args:
        session: SQLAlchemy session.
        win_condition: The win-condition card name (e.g., 'Hog Rider').
        corpus: Filter by corpus type.
        min_cluster_size: Minimum decks to form a sub-archetype.
        similarity_threshold: Jaccard similarity threshold for clustering.

    Returns:
        List of sub-archetype dicts, each with:
            'signature_cards': list of defining cards
            'sample_deck': most common full deck
            'count': number of decks in cluster
            'win_rate': win rate against this sub-archetype
            'avg_elixir': average deck cost
    """
    # Fetch all opponent decks containing the win condition
    stmt = select(Battle.opponent_deck, Battle.result).where(
        Battle.battle_type.in_(["PvP", "pathOfLegend"]),
        Battle.result.in_(["win", "loss"]),
    )
    if corpus:
        stmt = stmt.where(Battle.corpus == corpus)

    rows = session.execute(stmt).all()

    # Parse decks containing the win condition
    decks_with_wc = []  # (frozenset of support cards, full deck names, result)
    for opponent_deck_json, result in rows:
        if not opponent_deck_json:
            continue
        try:
            deck = json.loads(opponent_deck_json)
        except (json.JSONDecodeError, TypeError):
            continue

        card_names = [card.get("name") for card in deck if card.get("name")]
        if win_condition not in card_names:
            continue

        support = frozenset(c for c in card_names if c != win_condition)
        elixir = sum(card.get("elixirCost", 0) for card in deck)
        decks_with_wc.append({
            "support": support,
            "full_deck": tuple(sorted(card_names)),
            "result": result,
            "elixir": elixir,
        })

    if len(decks_with_wc) < min_cluster_size:
        logger.info(
            "Only %d decks with %s — too few for sub-archetype detection.",
            len(decks_with_wc), win_condition,
        )
        return []

    logger.info(
        "Clustering %d %s decks into sub-archetypes.",
        len(decks_with_wc), win_condition,
    )

    # Group by exact deck composition first (deck hash proxy)
    deck_groups: dict[tuple, list[dict]] = defaultdict(list)
    for d in decks_with_wc:
        deck_groups[d["full_deck"]].append(d)

    # Sort by frequency — most common deck variants
    sorted_groups = sorted(deck_groups.items(), key=lambda x: -len(x[1]))

    # Greedy clustering: assign each deck group to the most similar
    # existing cluster, or start a new one
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

    # Build output
    results = []
    for cluster in clusters:
        if len(cluster["decks"]) < min_cluster_size:
            continue

        wins = sum(1 for d in cluster["decks"] if d["result"] == "win")
        total = len(cluster["decks"])
        avg_elixir = sum(d["elixir"] for d in cluster["decks"]) / total

        # Signature cards: cards in >60% of decks in this cluster
        card_freq: Counter = Counter()
        for d in cluster["decks"]:
            for c in d["support"]:
                card_freq[c] += 1

        signature = [
            card for card, count in card_freq.most_common()
            if count / total >= 0.60
        ]

        # Most common full deck
        most_common_deck = cluster["deck_variants"].most_common(1)[0][0]

        results.append({
            "signature_cards": signature,
            "sample_deck": list(most_common_deck),
            "count": total,
            "win_rate": wins / total if total > 0 else 0.0,
            "avg_elixir": round(avg_elixir / 8, 1),  # per card
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
