"""Elixir economy simulator (ADR-002 section 2).

Reconstructs elixir state from replay events and builds empirical
distributions of elixir exchanges per matchup. Answers: "Given my deck
vs archetype X, what's the expected elixir differential over time?"

The key insight: we don't simulate card interactions from rules. We
observe them from real games. When PEKKA drops and the opponent responds
with Inferno Dragon, we record the sequence and its outcome. Over
thousands of games, this builds an empirical interaction model.
"""

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.stats import beta as beta_dist
from sqlalchemy import select
from sqlalchemy.orm import Session

from tracker.archetypes import classify_archetype
from tracker.ml.card_metadata import CardVocabulary, kebab_to_title, CARD_TYPES
from tracker.models import Battle, ReplayEvent, ReplaySummary

logger = logging.getLogger(__name__)

# Elixir generation: 1e per 2.8s in regular, 1e per 1.4s in double.
# Tick rate is ~20 ticks/second (empirically verified from replay data).
TICKS_PER_SECOND = 20
REGULAR_ELIXIR_RATE = 1.0 / (2.8 * TICKS_PER_SECOND)  # per tick
DOUBLE_ELIXIR_RATE = 2.0 * REGULAR_ELIXIR_RATE

# Game phases (in ticks at 20 tps)
REGULAR_TIME_END = 120 * TICKS_PER_SECOND  # 2:00 mark
DOUBLE_ELIXIR_START = REGULAR_TIME_END     # double elixir at 2:00
OT_START = 180 * TICKS_PER_SECOND         # 3:00 overtime
OT_END = 300 * TICKS_PER_SECOND           # 5:00 triple elixir OT

STARTING_ELIXIR = 5
MAX_ELIXIR = 10

# Response window: if opponent plays within N ticks of your play,
# it's considered a response to your card
RESPONSE_WINDOW_TICKS = 6 * TICKS_PER_SECOND  # 6 seconds


@dataclass
class ElixirState:
    """Tracks elixir state for one side over a game."""
    elixir: float = STARTING_ELIXIR
    total_spent: int = 0
    total_leaked: float = 0.0
    plays: list = field(default_factory=list)

    def spend(self, cost: int, tick: int) -> None:
        self.elixir = max(0, self.elixir - cost)
        self.total_spent += cost
        self.plays.append((tick, cost))

    def generate(self, ticks: int, rate: float) -> None:
        for _ in range(ticks):
            if self.elixir >= MAX_ELIXIR:
                self.total_leaked += rate
            self.elixir = min(MAX_ELIXIR, self.elixir + rate)


@dataclass
class ExchangeRecord:
    """A single elixir exchange: your play -> opponent response(s)."""
    your_card: str
    your_cost: int
    your_tick: int
    responses: list = field(default_factory=list)  # [(card, cost, tick)]
    game_phase: str = "regular"

    @property
    def response_cost(self) -> int:
        return sum(cost for _, cost, _ in self.responses)

    @property
    def net_elixir(self) -> int:
        """Positive = you gained elixir advantage."""
        return self.response_cost - self.your_cost

    @property
    def response_cards(self) -> list[str]:
        return [card for card, _, _ in self.responses]


def _get_elixir_rate(tick: int) -> float:
    """Get elixir generation rate for a given tick."""
    if tick < DOUBLE_ELIXIR_START:
        return REGULAR_ELIXIR_RATE
    return DOUBLE_ELIXIR_RATE


def _get_game_phase(tick: int) -> str:
    if tick < DOUBLE_ELIXIR_START:
        return "regular"
    elif tick < OT_START:
        return "double"
    else:
        return "overtime"


def _build_elixir_lookup(session: Session) -> dict[str, int]:
    """Build card name -> elixir cost lookup from the database.

    Handles both Title Case (deck_cards) and kebab-case (replay events).
    """
    vocab = CardVocabulary(session)
    lookup = {}
    for name in vocab.card_names():
        cost = vocab.elixir(name)
        if cost is not None:
            lookup[name] = cost
            # Also store kebab-case version for replay event matching
            kebab = name.lower().replace(" ", "-").replace(".", "")
            lookup[kebab] = cost
    return lookup


def extract_exchanges(
    events: list[tuple],
    elixir_lookup: dict[str, int],
) -> list[ExchangeRecord]:
    """Extract elixir exchanges from a sequence of replay events.

    An exchange is: team plays a card, opponent responds within
    RESPONSE_WINDOW_TICKS. Multiple opponent cards in the window
    count as a single exchange.

    Args:
        events: List of (game_tick, side, card_name) tuples, sorted by tick.
        elixir_lookup: Card name -> elixir cost mapping.

    Returns:
        List of ExchangeRecord objects.
    """
    exchanges = []

    team_plays = [(tick, card) for tick, side, card in events if side == "team"]
    opp_plays = [(tick, card) for tick, side, card in events if side == "opponent"]

    for t_tick, t_card in team_plays:
        cost = elixir_lookup.get(t_card) or elixir_lookup.get(kebab_to_title(t_card))
        if cost is None:
            continue

        exchange = ExchangeRecord(
            your_card=t_card,
            your_cost=cost,
            your_tick=t_tick,
            game_phase=_get_game_phase(t_tick),
        )

        # Find opponent responses in the window after this play
        for o_tick, o_card in opp_plays:
            if o_tick < t_tick:
                continue
            if o_tick > t_tick + RESPONSE_WINDOW_TICKS:
                break
            o_cost = elixir_lookup.get(o_card) or elixir_lookup.get(
                kebab_to_title(o_card)
            )
            if o_cost is not None:
                exchange.responses.append((o_card, o_cost, o_tick))

        exchanges.append(exchange)

    return exchanges


def reconstruct_elixir_curve(
    events: list[tuple],
    elixir_lookup: dict[str, int],
) -> dict[str, list[tuple[int, float]]]:
    """Reconstruct elixir state over time for both sides.

    Args:
        events: (game_tick, side, card_name) tuples sorted by tick.
        elixir_lookup: Card name -> elixir cost.

    Returns:
        Dict with 'team' and 'opponent' keys, each a list of
        (tick, elixir_level) tuples sampled every second.
    """
    if not events:
        return {"team": [], "opponent": []}

    max_tick = max(tick for tick, _, _ in events) + TICKS_PER_SECOND

    team = ElixirState()
    opp = ElixirState()

    # Index events by tick
    team_events = defaultdict(list)
    opp_events = defaultdict(list)
    for tick, side, card in events:
        cost = elixir_lookup.get(card) or elixir_lookup.get(kebab_to_title(card))
        if cost is None:
            continue
        if side == "team":
            team_events[tick].append(cost)
        else:
            opp_events[tick].append(cost)

    team_curve = []
    opp_curve = []

    for tick in range(0, max_tick + 1):
        rate = _get_elixir_rate(tick)
        team.generate(1, rate)
        opp.generate(1, rate)

        for cost in team_events.get(tick, []):
            team.spend(cost, tick)
        for cost in opp_events.get(tick, []):
            opp.spend(cost, tick)

        # Sample every second (every TICKS_PER_SECOND ticks)
        if tick % TICKS_PER_SECOND == 0:
            team_curve.append((tick, round(team.elixir, 2)))
            opp_curve.append((tick, round(opp.elixir, 2)))

    return {"team": team_curve, "opponent": opp_curve}


def build_exchange_distributions(
    session: Session,
    player_tag: Optional[str] = None,
    archetype_filter: Optional[str] = None,
    corpus: Optional[str] = None,
    min_exchanges: int = 5,
) -> dict:
    """Build empirical distributions of elixir exchanges per card per archetype.

    For each (your_card, opponent_archetype) pair, computes:
    - Mean net elixir (positive = you gained advantage)
    - Std dev of net elixir
    - Most common opponent responses
    - Win rate when this card is played

    Args:
        session: SQLAlchemy session.
        player_tag: Filter to this player's games.
        archetype_filter: Filter to games against this archetype.
        corpus: Filter by corpus type.
        min_exchanges: Minimum exchanges to include a card-archetype pair.

    Returns:
        Dict mapping card_name -> {
            'overall': {stats},
            'by_archetype': {archetype: {stats}},
            'by_phase': {phase: {stats}},
        }
    """
    elixir_lookup = _build_elixir_lookup(session)

    # Query battles with replay events
    stmt = select(
        Battle.battle_id,
        Battle.opponent_deck,
        Battle.result,
    ).where(
        Battle.battle_type.in_(["PvP", "pathOfLegend"]),
        Battle.result.in_(["win", "loss"]),
    )
    if corpus:
        stmt = stmt.where(Battle.corpus == corpus)
    if player_tag:
        tag_clean = player_tag.lstrip("#")
        stmt = stmt.where(Battle.player_tag.like(f"%{tag_clean}%"))

    battles = session.execute(stmt).all()

    # Collect all exchanges across all games
    card_exchanges: dict[str, dict] = defaultdict(
        lambda: {
            "net_elixir": [],
            "responses": defaultdict(int),
            "wins": 0,
            "losses": 0,
            "by_archetype": defaultdict(lambda: {
                "net_elixir": [], "wins": 0, "losses": 0,
            }),
            "by_phase": defaultdict(lambda: {"net_elixir": []}),
        }
    )

    games_processed = 0
    total_exchanges = 0

    for battle_id, opponent_deck_json, result in battles:
        if not opponent_deck_json:
            continue

        # Classify opponent archetype
        try:
            opp_deck = json.loads(opponent_deck_json)
        except (json.JSONDecodeError, TypeError):
            continue
        archetype = classify_archetype(opp_deck)

        if archetype_filter and archetype != archetype_filter:
            continue

        # Get replay events for this battle
        event_rows = session.execute(
            select(
                ReplayEvent.game_tick,
                ReplayEvent.side,
                ReplayEvent.card_name,
            )
            .where(ReplayEvent.battle_id == battle_id)
            .order_by(ReplayEvent.game_tick)
        ).all()

        if not event_rows:
            continue

        events = [(tick, side, card) for tick, side, card in event_rows]
        exchanges = extract_exchanges(events, elixir_lookup)

        if not exchanges:
            continue

        games_processed += 1
        is_win = result == "win"

        for ex in exchanges:
            card = kebab_to_title(ex.your_card)
            stats = card_exchanges[card]
            stats["net_elixir"].append(ex.net_elixir)
            if is_win:
                stats["wins"] += 1
            else:
                stats["losses"] += 1

            for resp_card, _, _ in ex.responses:
                stats["responses"][kebab_to_title(resp_card)] += 1

            # By archetype
            arch_stats = stats["by_archetype"][archetype]
            arch_stats["net_elixir"].append(ex.net_elixir)
            if is_win:
                arch_stats["wins"] += 1
            else:
                arch_stats["losses"] += 1

            # By phase
            stats["by_phase"][ex.game_phase]["net_elixir"].append(ex.net_elixir)

            total_exchanges += 1

    logger.info(
        "Built exchange distributions: %d games, %d exchanges, %d cards.",
        games_processed, total_exchanges, len(card_exchanges),
    )

    # Compute summary statistics
    results = {}
    for card, stats in card_exchanges.items():
        nets = stats["net_elixir"]
        if len(nets) < min_exchanges:
            continue

        total_plays = stats["wins"] + stats["losses"]

        results[card] = {
            "overall": {
                "mean_net_elixir": round(float(np.mean(nets)), 2),
                "std_net_elixir": round(float(np.std(nets)), 2),
                "median_net_elixir": round(float(np.median(nets)), 1),
                "total_plays": total_plays,
                "win_rate": round(stats["wins"] / total_plays, 3) if total_plays > 0 else 0,
                "top_responses": sorted(
                    stats["responses"].items(), key=lambda x: -x[1]
                )[:5],
            },
            "by_archetype": {},
            "by_phase": {},
        }

        # Per-archetype breakdown
        for arch, arch_stats in stats["by_archetype"].items():
            arch_nets = arch_stats["net_elixir"]
            if len(arch_nets) < min_exchanges:
                continue
            arch_total = arch_stats["wins"] + arch_stats["losses"]
            results[card]["by_archetype"][arch] = {
                "mean_net_elixir": round(float(np.mean(arch_nets)), 2),
                "std_net_elixir": round(float(np.std(arch_nets)), 2),
                "count": len(arch_nets),
                "win_rate": round(arch_stats["wins"] / arch_total, 3) if arch_total > 0 else 0,
            }

        # Per-phase breakdown
        for phase, phase_stats in stats["by_phase"].items():
            phase_nets = phase_stats["net_elixir"]
            if len(phase_nets) < min_exchanges:
                continue
            results[card]["by_phase"][phase] = {
                "mean_net_elixir": round(float(np.mean(phase_nets)), 2),
                "std_net_elixir": round(float(np.std(phase_nets)), 2),
                "count": len(phase_nets),
            }

    return {
        "games_processed": games_processed,
        "total_exchanges": total_exchanges,
        "card_distributions": dict(
            sorted(results.items(), key=lambda x: x[1]["overall"]["mean_net_elixir"])
        ),
    }


def compute_matchup_elixir_profile(
    session: Session,
    player_tag: str,
    archetype: str,
    min_games: int = 5,
) -> Optional[dict]:
    """Compute detailed elixir economy profile for a specific matchup.

    Reconstructs elixir curves for all games against the given archetype
    and computes aggregate statistics.

    Args:
        session: SQLAlchemy session.
        player_tag: Player to analyze.
        archetype: Opponent archetype name.
        min_games: Minimum games needed.

    Returns:
        Dict with elixir economy profile, or None if insufficient data.
    """
    elixir_lookup = _build_elixir_lookup(session)

    stmt = select(
        Battle.battle_id, Battle.opponent_deck, Battle.result,
    ).where(
        Battle.battle_type.in_(["PvP", "pathOfLegend"]),
        Battle.result.in_(["win", "loss"]),
        Battle.player_tag.like(f"%{player_tag.lstrip('#')}%"),
    )

    battles = session.execute(stmt).all()

    wins_curves = []
    losses_curves = []
    wins_exchanges = []
    losses_exchanges = []
    wins_leak = []
    losses_leak = []

    for battle_id, opp_deck_json, result in battles:
        if not opp_deck_json:
            continue
        try:
            opp_deck = json.loads(opp_deck_json)
        except (json.JSONDecodeError, TypeError):
            continue
        if classify_archetype(opp_deck) != archetype:
            continue

        event_rows = session.execute(
            select(
                ReplayEvent.game_tick,
                ReplayEvent.side,
                ReplayEvent.card_name,
            )
            .where(ReplayEvent.battle_id == battle_id)
            .order_by(ReplayEvent.game_tick)
        ).all()

        if not event_rows:
            continue

        events = [(tick, side, card) for tick, side, card in event_rows]
        curve = reconstruct_elixir_curve(events, elixir_lookup)
        exchanges = extract_exchanges(events, elixir_lookup)

        # Get elixir leak from replay summary
        leak_row = session.execute(
            select(ReplaySummary.elixir_leaked)
            .where(
                ReplaySummary.battle_id == battle_id,
                ReplaySummary.side == "team",
            )
        ).first()
        leak = leak_row[0] if leak_row else None

        if result == "win":
            wins_curves.append(curve)
            wins_exchanges.append(exchanges)
            if leak is not None:
                wins_leak.append(leak)
        else:
            losses_curves.append(curve)
            losses_exchanges.append(exchanges)
            if leak is not None:
                losses_leak.append(leak)

    total = len(wins_curves) + len(losses_curves)
    if total < min_games:
        return None

    # Compute aggregate elixir differential curves
    def _avg_differential(curves: list[dict]) -> list[tuple[int, float]]:
        """Average (team - opponent) elixir at each second."""
        if not curves:
            return []
        max_len = max(len(c["team"]) for c in curves)
        diffs_by_sec = defaultdict(list)
        for c in curves:
            for i, ((tick, team_e), (_, opp_e)) in enumerate(
                zip(c["team"], c["opponent"])
            ):
                diffs_by_sec[tick].append(team_e - opp_e)
        return [
            (tick, round(float(np.mean(vals)), 2))
            for tick, vals in sorted(diffs_by_sec.items())
            if len(vals) >= max(2, len(curves) // 3)
        ]

    # Exchange analysis: what cards work best/worst in this matchup?
    def _card_performance(exchange_lists: list[list[ExchangeRecord]]) -> dict:
        card_nets = defaultdict(list)
        for exchanges in exchange_lists:
            for ex in exchanges:
                card = kebab_to_title(ex.your_card)
                card_nets[card].append(ex.net_elixir)
        return {
            card: {
                "mean_net": round(float(np.mean(nets)), 2),
                "count": len(nets),
            }
            for card, nets in card_nets.items()
            if len(nets) >= 3
        }

    return {
        "archetype": archetype,
        "total_games": total,
        "wins": len(wins_curves),
        "losses": len(losses_curves),
        "win_rate": round(len(wins_curves) / total, 3),
        "avg_leak_wins": round(float(np.mean(wins_leak)), 2) if wins_leak else None,
        "avg_leak_losses": round(float(np.mean(losses_leak)), 2) if losses_leak else None,
        "elixir_diff_wins": _avg_differential(wins_curves),
        "elixir_diff_losses": _avg_differential(losses_curves),
        "card_performance_wins": _card_performance(wins_exchanges),
        "card_performance_losses": _card_performance(losses_exchanges),
    }
