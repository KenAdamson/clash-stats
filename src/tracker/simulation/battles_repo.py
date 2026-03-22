"""Paginated battles repository for simulation layer.

Loads opponent deck data from PostgreSQL in pages, aggregates card stats,
co-occurrence, and per-archetype deck lists in a single pass. Memory
usage is constant (~5MB) regardless of corpus size.

Replaces the pattern of each simulation function independently loading
the full battles table (1.3M rows × 2KB JSON = ~2.6GB per scan).
"""

import logging
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Optional

from scipy.stats import beta as beta_dist
from sqlalchemy import text
from sqlalchemy.orm import Session

from tracker.archetypes import classify_archetype

logger = logging.getLogger(__name__)

PAGE_SIZE = 10_000


@dataclass
class SimulationData:
    """Pre-aggregated simulation data from a single pass over battles."""

    total_battles: int = 0

    # Per-card win/loss stats (for interaction matrix)
    card_wins: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    card_losses: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    # Card co-occurrence pair counts
    pair_counts: Counter = field(default_factory=Counter)
    card_counts: Counter = field(default_factory=Counter)

    # Per-archetype win/loss + deck collection (for matchup posteriors + sub-archetypes)
    archetype_wins: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    archetype_losses: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    archetype_decks: dict[str, list] = field(default_factory=lambda: defaultdict(list))


def compute_simulation_data(
    session: Session,
    corpus: Optional[str] = None,
    player_tag: Optional[str] = None,
) -> SimulationData:
    """Single-pass paginated aggregation of all simulation data.

    Iterates over battles in pages using keyset pagination on battle_id.
    Extracts card names via jsonb_array_elements in PostgreSQL (no JSON
    parsing in Python). Computes card stats, co-occurrence, and archetype
    classification in one pass.

    Args:
        session: SQLAlchemy session.
        corpus: Filter by corpus type (e.g., 'personal', 'top_ladder').
        player_tag: Filter to battles involving this player.

    Returns:
        SimulationData with all pre-aggregated results.
    """
    data = SimulationData()
    last_id = ""
    page_num = 0

    # Build WHERE clause
    where_parts = [
        "b.battle_type IN ('PvP', 'pathOfLegend')",
        "b.result IN ('win', 'loss')",
        "b.opponent_deck IS NOT NULL",
    ]
    params: dict = {"page_size": PAGE_SIZE}

    if corpus:
        where_parts.append("b.corpus = :corpus")
        params["corpus"] = corpus
    if player_tag:
        tag_clean = player_tag.lstrip("#")
        where_parts.append("b.player_tag LIKE :ptag")
        params["ptag"] = f"%{tag_clean}%"

    where_clause = " AND ".join(where_parts)

    while True:
        params["last_id"] = last_id

        # Paginated query: one row per card per battle
        # PostgreSQL extracts card names from JSON — no Python json.loads
        rows = session.execute(
            text(f"""
                SELECT b.battle_id, b.result,
                       c->>'name' AS card_name,
                       (c->>'elixirCost')::int AS elixir
                FROM battles b,
                     jsonb_array_elements(b.opponent_deck::jsonb) AS c
                WHERE {where_clause}
                  AND b.battle_id > :last_id
                ORDER BY b.battle_id
                LIMIT :page_size
            """),
            params,
        ).all()

        if not rows:
            break

        # Group rows by battle_id (each battle produces ~8 rows, one per card)
        current_bid = None
        current_cards = []
        current_result = None
        current_elixir = 0

        for bid, result, card_name, elixir in rows:
            if bid != current_bid:
                # Process previous battle
                if current_bid and current_cards:
                    _process_battle(
                        data, current_cards, current_result, current_elixir,
                    )
                current_bid = bid
                current_cards = []
                current_result = result
                current_elixir = 0

            if card_name:
                current_cards.append(card_name)
                current_elixir += elixir or 0

        # Process last battle in page
        if current_bid and current_cards:
            _process_battle(data, current_cards, current_result, current_elixir)

        last_id = rows[-1][0]  # last battle_id in page
        page_num += 1

        if page_num % 10 == 0:
            logger.info(
                "Simulation scan: %d pages, %d battles processed",
                page_num, data.total_battles,
            )

        # If we got fewer rows than page_size, we've hit the end
        # (accounting for ~8 rows per battle)
        if len(rows) < PAGE_SIZE:
            break

    logger.info(
        "Simulation scan complete: %d battles, %d unique cards, %d archetypes",
        data.total_battles,
        len(data.card_counts),
        len(data.archetype_wins) + len(data.archetype_losses),
    )
    return data


def _process_battle(
    data: SimulationData,
    card_names: list[str],
    result: str,
    elixir_total: int,
) -> None:
    """Aggregate one battle's data into SimulationData."""
    data.total_battles += 1
    card_set = set(card_names)
    sorted_cards = sorted(card_set)

    # Card interaction stats
    for card in card_set:
        if result == "win":
            data.card_wins[card] += 1
        else:
            data.card_losses[card] += 1
        data.card_counts[card] += 1

    # Card co-occurrence pairs
    for i, a in enumerate(sorted_cards):
        for b in sorted_cards[i + 1:]:
            data.pair_counts[(a, b)] += 1

    # Archetype classification
    # classify_archetype expects list of dicts with "name" key
    deck_dicts = [{"name": c} for c in card_names]
    archetype = classify_archetype(deck_dicts)
    if result == "win":
        data.archetype_wins[archetype] += 1
    else:
        data.archetype_losses[archetype] += 1

    # Store lightweight deck info for sub-archetype clustering
    data.archetype_decks[archetype].append({
        "card_names": sorted_cards,
        "result": result,
        "elixir": elixir_total,
    })
