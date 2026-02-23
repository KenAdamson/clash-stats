"""Bayesian matchup estimation (ADR-002 §3).

Beta-binomial model for win probability per opponent sub-archetype,
with hierarchical priors for sparse matchups.
"""

import json
import logging
from collections import defaultdict
from typing import Optional

import numpy as np
from scipy.stats import beta as beta_dist
from sqlalchemy import select
from sqlalchemy.orm import Session

from tracker.archetypes import classify_archetype
from tracker.models import Battle
from tracker.simulation.interaction_matrix import detect_sub_archetypes

logger = logging.getLogger(__name__)

# Uninformative prior: Beta(1, 1) = uniform
DEFAULT_PRIOR = (1, 1)

# Weakly informative prior for unseen matchups: assume 50% ± wide uncertainty
HIERARCHICAL_PRIOR = (2, 2)


def compute_matchup_posteriors(
    session: Session,
    corpus: Optional[str] = None,
    player_tag: Optional[str] = None,
    min_battles: int = 5,
    use_sub_archetypes: bool = True,
) -> dict:
    """Compute Beta posterior distributions for each opponent archetype.

    Args:
        session: SQLAlchemy session.
        corpus: Filter by corpus type.
        player_tag: Filter to battles involving this player.
        min_battles: Minimum battles to include a matchup.
        use_sub_archetypes: If True, detect and use sub-archetypes for
            major win conditions.

    Returns:
        Dict mapping archetype_name -> {
            'wins': int, 'losses': int, 'total': int,
            'posterior_mean': float,
            'ci_low': float, 'ci_high': float,
            'prior': (alpha, beta),
            'sub_archetypes': list (if detected),
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
    logger.info("Computing matchup posteriors from %d battles.", len(rows))

    # First pass: classify by top-level archetype
    archetype_stats: dict[str, dict] = defaultdict(
        lambda: {"wins": 0, "losses": 0, "decks": []}
    )

    for opponent_deck_json, result in rows:
        if not opponent_deck_json:
            continue
        try:
            deck = json.loads(opponent_deck_json)
        except (json.JSONDecodeError, TypeError):
            continue

        archetype = classify_archetype(deck)
        stats = archetype_stats[archetype]
        if result == "win":
            stats["wins"] += 1
        else:
            stats["losses"] += 1
        stats["decks"].append(deck)

    # Compute posteriors
    matchups = {}
    for archetype, stats in sorted(
        archetype_stats.items(), key=lambda x: -(x[1]["wins"] + x[1]["losses"])
    ):
        total = stats["wins"] + stats["losses"]
        if total < min_battles:
            continue

        prior = DEFAULT_PRIOR
        a = stats["wins"] + prior[0]
        b = stats["losses"] + prior[1]
        ci_low, ci_high = beta_dist.ppf([0.025, 0.975], a, b)

        matchup = {
            "wins": stats["wins"],
            "losses": stats["losses"],
            "total": total,
            "posterior_mean": a / (a + b),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "ci_width": float(ci_high - ci_low),
            "prior": prior,
        }

        # Sub-archetype detection for major archetypes
        if use_sub_archetypes and total >= 30:
            win_condition = _get_win_condition(archetype)
            if win_condition:
                sub_archetypes = detect_sub_archetypes(
                    session,
                    win_condition,
                    corpus=corpus,
                    min_cluster_size=max(10, total // 10),
                )
                if sub_archetypes:
                    matchup["sub_archetypes"] = _enrich_sub_archetypes(
                        sub_archetypes, prior
                    )

        matchups[archetype] = matchup

    return matchups


def _enrich_sub_archetypes(
    sub_archetypes: list[dict], prior: tuple
) -> list[dict]:
    """Add Beta posteriors to each sub-archetype."""
    enriched = []
    for sa in sub_archetypes:
        wins = int(sa["win_rate"] * sa["count"])
        losses = sa["count"] - wins
        a = wins + prior[0]
        b = losses + prior[1]
        ci_low, ci_high = beta_dist.ppf([0.025, 0.975], a, b)

        enriched.append({
            **sa,
            "wins": wins,
            "losses": losses,
            "posterior_mean": a / (a + b),
            "ci_low": float(ci_low),
            "ci_high": float(ci_high),
            "ci_width": float(ci_high - ci_low),
        })
    return enriched


def compute_threat_ranking(
    matchup_posteriors: dict,
    min_battles: int = 10,
) -> list[dict]:
    """Rank matchups by threat level (lowest expected win rate).

    Args:
        matchup_posteriors: Output from compute_matchup_posteriors().
        min_battles: Minimum battles for credible ranking.

    Returns:
        Sorted list of (archetype, posterior_mean, ci_low, ci_high, total).
    """
    threats = []
    for archetype, data in matchup_posteriors.items():
        if data["total"] < min_battles:
            continue
        threats.append({
            "archetype": archetype,
            "posterior_mean": data["posterior_mean"],
            "ci_low": data["ci_low"],
            "ci_high": data["ci_high"],
            "total": data["total"],
            "wins": data["wins"],
            "losses": data["losses"],
        })

    # Sort by posterior mean ascending (worst matchups first)
    threats.sort(key=lambda x: x["posterior_mean"])
    return threats


def sample_matchup_distribution(
    wins: int,
    losses: int,
    prior: tuple = DEFAULT_PRIOR,
    n_samples: int = 10000,
) -> np.ndarray:
    """Draw samples from the matchup posterior.

    Useful for Monte Carlo propagation of matchup uncertainty
    into downstream simulations.

    Args:
        wins: Observed wins.
        losses: Observed losses.
        prior: Beta prior (alpha, beta).
        n_samples: Number of samples to draw.

    Returns:
        Array of win probability samples.
    """
    a = wins + prior[0]
    b = losses + prior[1]
    return np.random.default_rng().beta(a, b, size=n_samples)


# Map archetype names back to their win-condition card
_ARCHETYPE_TO_WC: dict[str, str] = {
    "Golem Beatdown": "Golem",
    "Lava Hound": "Lava Hound",
    "Giant Beatdown": "Giant",
    "Royal Giant": "Royal Giant",
    "Hog Cycle": "Hog Rider",
    "X-Bow Siege": "X-Bow",
    "Mortar Siege": "Mortar",
    "Bridge Spam": "Ram Rider",
    "Graveyard Control": "Graveyard",
    "Miner Control": "Miner",
    "Three Musketeers": "Three Musketeers",
    "Sparky": "Sparky",
    "Balloon": "Balloon",
    "Elite Barbarians": "Elite Barbarians",
    "P.E.K.K.A Control": "P.E.K.K.A",
    "Mega Knight": "Mega Knight",
    "Goblin Barrel Bait": "Goblin Barrel",
    "Skeleton King": "Skeleton King",
    "Monk": "Monk",
    "Archer Queen": "Archer Queen",
    "Goblin Giant": "Goblin Giant",
    "Electro Giant": "Electro Giant",
    "Egiant": "Elixir Golem",
}


def _get_win_condition(archetype: str) -> Optional[str]:
    """Get the win-condition card for an archetype name."""
    return _ARCHETYPE_TO_WC.get(archetype)
