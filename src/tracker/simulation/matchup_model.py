"""Bayesian matchup estimation (ADR-002 §3).

Beta-binomial model for win probability per opponent sub-archetype,
with hierarchical priors for sparse matchups.
"""

import logging
from typing import Optional

import numpy as np
from scipy.stats import beta as beta_dist
from sqlalchemy.orm import Session

from tracker.simulation.interaction_matrix import detect_sub_archetypes

logger = logging.getLogger(__name__)

# Uninformative prior: Beta(1, 1) = uniform
DEFAULT_PRIOR = (1, 1)

# Weakly informative prior for unseen matchups: assume 50% ± wide uncertainty
HIERARCHICAL_PRIOR = (2, 2)


def compute_matchup_posteriors(
    session_or_data=None,
    *,
    sim_data=None,
    corpus: Optional[str] = None,
    player_tag: Optional[str] = None,
    min_battles: int = 5,
    use_sub_archetypes: bool = True,
) -> dict:
    """Compute Beta posterior distributions for each opponent archetype.

    Args:
        session_or_data: SQLAlchemy session (legacy) or SimulationData.
        sim_data: Pre-aggregated SimulationData (keyword, preferred).
        corpus: Filter by corpus type.
        player_tag: Filter to battles involving this player.
        min_battles: Minimum battles to include a matchup.
        use_sub_archetypes: If True, detect and use sub-archetypes.

    Returns:
        Dict mapping archetype_name -> posterior stats.
    """
    from tracker.simulation.battles_repo import SimulationData, compute_simulation_data

    if sim_data is not None:
        pass  # Use provided sim_data
    elif isinstance(session_or_data, SimulationData):
        sim_data = session_or_data
    elif session_or_data is not None:
        # Legacy: session passed positionally
        sim_data = compute_simulation_data(session_or_data, corpus=corpus, player_tag=player_tag)
    else:
        raise ValueError("Either session or sim_data required")

    logger.info("Computing matchup posteriors from %d battles.", sim_data.total_battles)

    # Build posteriors from pre-aggregated archetype stats
    all_archetypes = set(sim_data.archetype_wins.keys()) | set(sim_data.archetype_losses.keys())

    matchups = {}
    for archetype in sorted(
        all_archetypes,
        key=lambda a: -(sim_data.archetype_wins.get(a, 0) + sim_data.archetype_losses.get(a, 0)),
    ):
        wins = sim_data.archetype_wins.get(archetype, 0)
        losses = sim_data.archetype_losses.get(archetype, 0)
        total = wins + losses
        if total < min_battles:
            continue

        prior = DEFAULT_PRIOR
        a = wins + prior[0]
        b = losses + prior[1]
        ci_low, ci_high = beta_dist.ppf([0.025, 0.975], a, b)

        matchup = {
            "wins": wins,
            "losses": losses,
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
                    win_condition,
                    sim_data=sim_data,
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
