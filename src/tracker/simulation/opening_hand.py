"""Opening hand simulator (ADR-002 section 1).

Enumerates all C(8,4) = 70 possible opening hands and correlates
hand composition with observed win rates from replay data.
"""

import json
import logging
from collections import defaultdict
from itertools import combinations
from typing import Optional

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from tracker.archetypes import classify_archetype
from tracker.ml.card_metadata import CardVocabulary, kebab_to_title
from tracker.models import Battle, ReplayEvent

logger = logging.getLogger(__name__)


def analyze_opening_hands(
    session: Session,
    player_tag: str,
    archetype_filter: Optional[str] = None,
    min_games: int = 3,
) -> dict:
    """Analyze opening hand quality vs win rate from replay data.

    Uses the first card played by each side to infer which cards were
    in the opening hand. Cross-references with game outcomes.

    Args:
        session: SQLAlchemy session.
        player_tag: Player to analyze.
        archetype_filter: Only analyze games vs this archetype.
        min_games: Minimum games to include an opening card.

    Returns:
        Dict with opening hand analysis results.
    """
    elixir_lookup = {}
    vocab = CardVocabulary(session)
    for name in vocab.card_names():
        cost = vocab.elixir(name)
        if cost is not None:
            elixir_lookup[name] = cost
            elixir_lookup[name.lower().replace(" ", "-").replace(".", "")] = cost

    # Get player's deck cards (most recent deck)
    stmt = select(Battle.player_deck).where(
        Battle.player_tag.like(f"%{player_tag.lstrip('#')}%"),
        Battle.battle_type.in_(["PvP", "pathOfLegend"]),
    ).order_by(Battle.battle_time.desc()).limit(1)

    row = session.execute(stmt).first()
    if not row or not row[0]:
        return {"error": "No deck data found"}

    try:
        deck = json.loads(row[0])
        deck_cards = [card["name"] for card in deck if card.get("name")]
    except (json.JSONDecodeError, TypeError):
        return {"error": "Could not parse deck"}

    if len(deck_cards) != 8:
        return {"error": f"Expected 8 cards, got {len(deck_cards)}"}

    # Enumerate all possible hands
    all_hands = list(combinations(deck_cards, 4))

    # Get battles with replay data
    stmt = select(
        Battle.battle_id, Battle.opponent_deck, Battle.result,
    ).where(
        Battle.battle_type.in_(["PvP", "pathOfLegend"]),
        Battle.result.in_(["win", "loss"]),
        Battle.player_tag.like(f"%{player_tag.lstrip('#')}%"),
    )

    battles = session.execute(stmt).all()

    # Track first card played -> win/loss
    opener_stats: dict[str, dict] = defaultdict(
        lambda: {"wins": 0, "losses": 0, "costs": []}
    )
    games_analyzed = 0

    for battle_id, opp_deck_json, result in battles:
        if archetype_filter and opp_deck_json:
            try:
                arch = classify_archetype(json.loads(opp_deck_json))
            except (json.JSONDecodeError, TypeError):
                continue
            if arch != archetype_filter:
                continue

        # Get first team play
        first_play = session.execute(
            select(ReplayEvent.card_name, ReplayEvent.game_tick)
            .where(
                ReplayEvent.battle_id == battle_id,
                ReplayEvent.side == "team",
            )
            .order_by(ReplayEvent.game_tick)
            .limit(1)
        ).first()

        if not first_play:
            continue

        card_name = kebab_to_title(first_play[0])
        first_tick = first_play[1]

        cost = elixir_lookup.get(card_name, 0)
        time_to_first = round(first_tick / 20.0, 1)  # ticks to seconds

        stats = opener_stats[card_name]
        if result == "win":
            stats["wins"] += 1
        else:
            stats["losses"] += 1
        stats["costs"].append(cost)
        games_analyzed += 1

    # Compute hand quality metrics
    hand_analysis = []
    for hand in all_hands:
        costs = [elixir_lookup.get(c, 0) for c in hand]
        total = sum(costs)
        cheapest = min(costs) if costs else 0
        cheap_count = sum(1 for c in costs if c <= 3)

        # Time to first play (cheapest cost / generation rate)
        # At 1e per 2.8s, starting at 5e: can play cheapest immediately
        # if cheapest <= 5, otherwise wait (cheapest - 5) * 2.8s
        wait = max(0, (cheapest - 5) * 2.8)

        hand_analysis.append({
            "cards": list(hand),
            "total_cost": total,
            "cheapest": cheapest,
            "cheap_cards": cheap_count,
            "time_to_first": round(wait, 1),
        })

    # Sort by total cost
    hand_analysis.sort(key=lambda h: h["total_cost"])

    # Opener card performance
    opener_results = {}
    for card, stats in opener_stats.items():
        total = stats["wins"] + stats["losses"]
        if total < min_games:
            continue
        opener_results[card] = {
            "wins": stats["wins"],
            "losses": stats["losses"],
            "total": total,
            "win_rate": round(stats["wins"] / total, 3),
            "avg_cost": round(float(np.mean(stats["costs"])), 1),
        }

    # Sort by win rate
    opener_results = dict(
        sorted(opener_results.items(), key=lambda x: -x[1]["win_rate"])
    )

    return {
        "deck_cards": deck_cards,
        "total_hands": len(all_hands),
        "games_analyzed": games_analyzed,
        "cost_distribution": {
            "min_hand": hand_analysis[0]["total_cost"],
            "max_hand": hand_analysis[-1]["total_cost"],
            "mean_hand": round(float(np.mean([h["total_cost"] for h in hand_analysis])), 1),
            "cheapest_possible": hand_analysis[0]["cards"],
            "most_expensive": hand_analysis[-1]["cards"],
        },
        "opener_performance": opener_results,
        "all_hands": hand_analysis,
        "archetype_filter": archetype_filter,
    }
