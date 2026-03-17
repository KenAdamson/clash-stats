"""Analytics queries using SQLAlchemy ORM."""

import hashlib
import json
from typing import Optional

from sqlalchemy import case, cast, func, Integer, select
from sqlalchemy.orm import Session

from tracker.archetypes import classify_archetype
from tracker.metrics import BATTLES_SCRAPED, BATTLES_DEDUPED
from tracker.models import Battle, DeckCard, PlayerSnapshot

# Battle types considered "ladder" (competitive, with trophy stakes)
LADDER_TYPES = ("PvP",)


def _card_variant(card: dict) -> str:
    """Classify a card as base, evo, or hero from API data."""
    evo_level = card.get("evolutionLevel", 0)
    max_evo = card.get("maxEvolutionLevel", 0)
    if evo_level and evo_level > 0:
        return "hero" if max_evo and max_evo > 1 else "evo"
    return "base"


def _apply_ladder_filter(stmt, ladder_only: bool):
    """Add a WHERE clause to restrict to ladder battles if requested.

    Always filters to personal corpus (excludes training data).
    """
    stmt = stmt.where(Battle.corpus == "personal")
    if ladder_only:
        return stmt.where(Battle.battle_type.in_(LADDER_TYPES))
    return stmt


def generate_battle_id(battle: dict) -> str:
    """Generate unique ID for deduplication.

    Args:
        battle: Raw battle dict from the API.

    Returns:
        32-character hex string (SHA-256 truncated).
    """
    key_data = json.dumps(
        {
            "battleTime": battle.get("battleTime"),
            "team": [
                (p.get("tag"), p.get("crowns")) for p in battle.get("team", [])
            ],
            "opponent": [
                (p.get("tag"), p.get("crowns")) for p in battle.get("opponent", [])
            ],
        },
        sort_keys=True,
    )
    return hashlib.sha256(key_data.encode()).hexdigest()[:32]


def generate_deck_hash(deck: list) -> str:
    """Generate hash for deck (ignoring levels, but including evo status).

    Args:
        deck: List of card dicts from the API.

    Returns:
        16-character hex string (MD5 truncated).
    """
    card_keys = sorted([
        f"{card.get('name', '')}:evo{card.get('evolutionLevel', 0)}"
        for card in deck
    ])
    return hashlib.md5("|".join(card_keys).encode()).hexdigest()[:16]


def battle_exists(session: Session, battle_id: str) -> bool:
    """Check if a battle is already stored.

    Args:
        session: SQLAlchemy session.
        battle_id: Battle ID hash.

    Returns:
        True if the battle already exists in the database.
    """
    stmt = select(Battle.id).where(Battle.battle_id == battle_id).limit(1)
    return session.execute(stmt).first() is not None


def store_player_snapshot(session: Session, player: dict) -> None:
    """Store a player profile snapshot.

    Args:
        session: SQLAlchemy session.
        player: Player profile dict from the API.
    """
    snapshot = PlayerSnapshot(
        player_tag=player.get("tag"),
        name=player.get("name"),
        exp_level=player.get("expLevel"),
        trophies=player.get("trophies"),
        best_trophies=player.get("bestTrophies"),
        wins=player.get("wins"),
        losses=player.get("losses"),
        battle_count=player.get("battleCount"),
        three_crown_wins=player.get("threeCrownWins"),
        challenge_cards_won=player.get("challengeCardsWon"),
        challenge_max_wins=player.get("challengeMaxWins"),
        tournament_battle_count=player.get("tournamentBattleCount"),
        tournament_cards_won=player.get("tournamentCardsWon"),
        war_day_wins=player.get("warDayWins"),
        total_donations=player.get("totalDonations"),
        clan_tag=player.get("clan", {}).get("tag"),
        clan_name=player.get("clan", {}).get("name"),
        arena_name=player.get("arena", {}).get("name"),
        raw_json=json.dumps(player),
    )
    session.add(snapshot)
    session.commit()


def store_battle(
    session: Session, battle: dict, player_tag: str, corpus: str = "personal",
) -> tuple[str, bool]:
    """Store a battle. Returns (battle_id, is_new).

    Args:
        session: SQLAlchemy session.
        battle: Raw battle dict from the API.
        player_tag: Player's tag.
        corpus: Data source provenance ('personal', 'top_ladder', 'matchup_targeted').

    Returns:
        Tuple of (battle_id, is_new).
    """
    bid = generate_battle_id(battle)

    if battle_exists(session, bid):
        BATTLES_DEDUPED.labels(corpus=corpus).inc()
        return bid, False

    team = battle.get("team", [{}])[0]
    opponent = battle.get("opponent", [{}])[0]

    player_deck = team.get("cards", [])
    opponent_deck = opponent.get("cards", [])

    player_crowns = team.get("crowns", 0)
    opponent_crowns = opponent.get("crowns", 0)

    if player_crowns > opponent_crowns:
        result = "win"
    elif player_crowns < opponent_crowns:
        result = "loss"
    else:
        result = "draw"

    b = Battle(
        battle_id=bid,
        battle_time=battle.get("battleTime"),
        battle_type=battle.get("type"),
        arena_name=battle.get("arena", {}).get("name"),
        game_mode_name=battle.get("gameMode", {}).get("name"),
        is_ladder_tournament=1 if battle.get("isLadderTournament") else 0,
        player_tag=team.get("tag"),
        player_name=team.get("name"),
        player_starting_trophies=team.get("startingTrophies"),
        player_trophy_change=team.get("trophyChange"),
        player_crowns=player_crowns,
        player_king_tower_hp=team.get("kingTowerHitPoints"),
        player_princess_tower_hp=json.dumps(team.get("princessTowersHitPoints", [])),
        player_deck=json.dumps(player_deck),
        player_deck_hash=generate_deck_hash(player_deck),
        opponent_tag=opponent.get("tag"),
        opponent_name=opponent.get("name"),
        opponent_starting_trophies=opponent.get("startingTrophies"),
        opponent_trophy_change=opponent.get("trophyChange"),
        opponent_crowns=opponent_crowns,
        opponent_king_tower_hp=opponent.get("kingTowerHitPoints"),
        opponent_princess_tower_hp=json.dumps(
            opponent.get("princessTowersHitPoints", [])
        ),
        opponent_deck=json.dumps(opponent_deck),
        opponent_deck_hash=generate_deck_hash(opponent_deck),
        result=result,
        crown_differential=player_crowns - opponent_crowns,
        raw_json=json.dumps(battle),
        player_elixir_leaked=team.get("elixirLeaked"),
        opponent_elixir_leaked=opponent.get("elixirLeaked"),
        battle_duration=battle.get("battleDuration"),
        corpus=corpus,
    )
    session.add(b)

    # Store individual cards for matchup analysis
    for card in player_deck:
        session.add(DeckCard(
            battle_id=bid,
            card_name=card.get("name"),
            card_level=card.get("level"),
            card_max_level=card.get("maxLevel"),
            card_elixir=card.get("elixirCost"),
            is_player_deck=1,
            evolution_level=card.get("evolutionLevel", 0),
            star_level=card.get("starLevel", 0),
            card_variant=_card_variant(card),
        ))

    for card in opponent_deck:
        session.add(DeckCard(
            battle_id=bid,
            card_name=card.get("name"),
            card_level=card.get("level"),
            card_max_level=card.get("maxLevel"),
            card_elixir=card.get("elixirCost"),
            is_player_deck=0,
            evolution_level=card.get("evolutionLevel", 0),
            star_level=card.get("starLevel", 0),
            card_variant=_card_variant(card),
        ))

    session.commit()
    BATTLES_SCRAPED.labels(corpus=corpus).inc()
    return bid, True


def get_total_battles(session: Session) -> int:
    """Get total number of tracked personal battles."""
    return session.scalar(
        select(func.count()).select_from(Battle).where(Battle.corpus == "personal")
    ) or 0


def get_overall_stats(session: Session, ladder_only: bool = False) -> dict:
    """Get aggregated stats across all battles."""
    stmt = select(
        func.count().label("total"),
        func.sum(case((Battle.result == "win", 1), else_=0)).label("wins"),
        func.sum(case((Battle.result == "loss", 1), else_=0)).label("losses"),
        func.sum(case((Battle.result == "draw", 1), else_=0)).label("draws"),
        func.sum(Battle.player_crowns).label("total_crowns"),
        func.sum(Battle.opponent_crowns).label("crowns_against"),
        func.sum(case((Battle.player_crowns == 3, 1), else_=0)).label("three_crowns"),
        func.avg(Battle.player_crowns).label("avg_crowns"),
        func.avg(Battle.opponent_crowns).label("avg_crowns_against"),
        func.min(Battle.battle_time).label("first_battle"),
        func.max(Battle.battle_time).label("last_battle"),
    )
    stmt = _apply_ladder_filter(stmt, ladder_only)
    row = session.execute(stmt).first()
    if not row:
        return {}
    return row._asdict()


def get_all_time_api_stats(session: Session) -> dict:
    """Get latest snapshot from API (all-time stats)."""
    stmt = select(PlayerSnapshot).order_by(PlayerSnapshot.id.desc()).limit(1)
    snapshot = session.scalars(stmt).first()
    if not snapshot:
        return {}
    return {c.key: getattr(snapshot, c.key) for c in PlayerSnapshot.__table__.columns}


def get_stats_by_battle_type(session: Session) -> list[dict]:
    """Get win rate grouped by battle type."""
    wins_case = func.sum(case((Battle.result == "win", 1), else_=0))
    stmt = (
        select(
            Battle.battle_type,
            func.count().label("total"),
            wins_case.label("wins"),
            func.sum(case((Battle.result == "loss", 1), else_=0)).label("losses"),
            func.round(100.0 * wins_case / func.count(), 1).label("win_rate"),
        )
        .group_by(Battle.battle_type)
        .order_by(func.count().desc())
    )
    return [row._asdict() for row in session.execute(stmt).all()]


def get_deck_stats(session: Session, min_battles: int = 3) -> list[dict]:
    """Get per-deck statistics."""
    wins_case = func.sum(case((Battle.result == "win", 1), else_=0))
    stmt = (
        select(
            Battle.player_deck_hash,
            Battle.player_deck,
            func.count().label("total"),
            wins_case.label("wins"),
            func.sum(case((Battle.result == "loss", 1), else_=0)).label("losses"),
            func.round(100.0 * wins_case / func.count(), 1).label("win_rate"),
            func.sum(case((Battle.player_crowns == 3, 1), else_=0)).label("three_crowns"),
            func.round(func.avg(Battle.player_crowns), 2).label("avg_crowns"),
        )
        .group_by(Battle.player_deck_hash)
        .having(func.count() >= min_battles)
        .order_by(func.count().desc())
    )
    results = []
    for row in session.execute(stmt).all():
        d = row._asdict()
        deck_json = json.loads(d["player_deck"])
        d["deck_cards"] = sorted([c.get("name", "Unknown") for c in deck_json])
        results.append(d)
    return results


def get_crown_distribution(session: Session, ladder_only: bool = False) -> dict:
    """Get crown distribution by result type."""
    stmt = (
        select(
            Battle.result,
            Battle.player_crowns,
            func.count().label("count"),
        )
        .where(Battle.result.in_(["win", "loss"]))
        .group_by(Battle.result, Battle.player_crowns)
        .order_by(Battle.result, Battle.player_crowns)
    )
    stmt = _apply_ladder_filter(stmt, ladder_only)
    distribution: dict = {"win": {}, "loss": {}}
    for row in session.execute(stmt).all():
        distribution[row.result][row.player_crowns] = row.count
    return distribution


def get_card_matchup_stats(session: Session, min_battles: int = 3, ladder_only: bool = False) -> list[dict]:
    """Get win rate vs opponent cards."""
    wins_case = func.sum(case((Battle.result == "win", 1), else_=0))
    distinct_battles = func.count(func.distinct(Battle.battle_id))
    stmt = (
        select(
            DeckCard.card_name,
            distinct_battles.label("times_faced"),
            wins_case.label("wins"),
            func.sum(case((Battle.result == "loss", 1), else_=0)).label("losses"),
            func.round(100.0 * wins_case / distinct_battles, 1).label("win_rate"),
        )
        .join(Battle, DeckCard.battle_id == Battle.battle_id)
        .where(DeckCard.is_player_deck == 0)
        .group_by(DeckCard.card_name)
        .having(distinct_battles >= min_battles)
        .order_by(distinct_battles.desc())
    )
    stmt = _apply_ladder_filter(stmt, ladder_only)
    return [row._asdict() for row in session.execute(stmt).all()]


def get_recent_battles(session: Session, limit: int = 10, ladder_only: bool = False) -> list[dict]:
    """Get last N battles with details."""
    stmt = (
        select(
            Battle.battle_id,
            Battle.battle_time,
            Battle.battle_type,
            Battle.arena_name,
            Battle.player_crowns,
            Battle.opponent_crowns,
            Battle.result,
            Battle.player_trophy_change,
            Battle.player_starting_trophies,
            Battle.opponent_name,
            Battle.player_deck,
            Battle.opponent_deck,
        )
        .order_by(Battle.battle_time.desc())
    )
    stmt = _apply_ladder_filter(stmt, ladder_only)
    stmt = stmt.limit(limit)
    results = []
    for row in session.execute(stmt).all():
        d = row._asdict()
        deck_json = json.loads(d["player_deck"])
        d["deck_cards"] = [c.get("name", "Unknown") for c in deck_json]
        opp_json = json.loads(d["opponent_deck"]) if d.get("opponent_deck") else []
        d["opponent_cards"] = [
            {
                "name": c.get("name", "Unknown"),
                "elixir": c.get("elixirCost"),
                "evo": bool(c.get("evolutionLevel")),
            }
            for c in opp_json
        ]
        results.append(d)
    return results


def get_time_of_day_stats(session: Session, ladder_only: bool = False) -> list[dict]:
    """Get win rate by hour of day."""
    hour_expr = cast(func.substr(Battle.battle_time, 10, 2), Integer)
    wins_case = func.sum(case((Battle.result == "win", 1), else_=0))
    stmt = (
        select(
            hour_expr.label("hour"),
            func.count().label("total"),
            wins_case.label("wins"),
            func.round(100.0 * wins_case / func.count(), 1).label("win_rate"),
        )
        .where(Battle.battle_time.isnot(None))
        .group_by(hour_expr)
        .order_by(hour_expr)
    )
    stmt = _apply_ladder_filter(stmt, ladder_only)
    return [row._asdict() for row in session.execute(stmt).all()]


def get_corpus_traffic_by_hour(session: Session) -> list[dict]:
    """Get corpus battle volume by hour of day (UTC).

    Returns normalized 0-100 traffic index for overlay on WR chart.
    """
    hour_expr = cast(func.substr(Battle.battle_time, 10, 2), Integer)
    stmt = (
        select(
            hour_expr.label("hour"),
            func.count().label("total"),
        )
        .where(Battle.battle_time.isnot(None))
        .where(Battle.corpus != "personal")
        .group_by(hour_expr)
        .order_by(hour_expr)
    )
    rows = [row._asdict() for row in session.execute(stmt).all()]
    if not rows:
        return rows
    counts = [r["total"] for r in rows]
    lo, hi = min(counts), max(counts)
    spread = hi - lo if hi > lo else 1
    for r in rows:
        r["traffic_index"] = round((r["total"] - lo) / spread * 100, 1)
    return rows


def get_streaks(session: Session, ladder_only: bool = False) -> dict:
    """Detect win/loss streaks from battle history.

    Returns:
        Dict with 'current_streak', 'longest_win_streak',
        'longest_loss_streak', and 'streaks' list.
    """
    stmt = (
        select(
            Battle.battle_time,
            Battle.result,
            Battle.player_starting_trophies,
            Battle.player_trophy_change,
        )
        .where(Battle.result.in_(["win", "loss"]))
        .order_by(Battle.battle_time.asc())
    )
    stmt = _apply_ladder_filter(stmt, ladder_only)
    rows = [row._asdict() for row in session.execute(stmt).all()]

    if not rows:
        return {
            "current_streak": None,
            "longest_win_streak": None,
            "longest_loss_streak": None,
            "streaks": [],
        }

    def _finish_streak(stype: str, length: int, start: dict, end: dict) -> dict:
        start_trophies = start.get("player_starting_trophies") or 0
        end_trophies = (end.get("player_starting_trophies") or 0) + (
            end.get("player_trophy_change") or 0
        )
        return {
            "type": stype,
            "length": length,
            "start_trophies": start_trophies,
            "end_trophies": end_trophies,
            "start_date": (start.get("battle_time") or "")[:8],
            "end_date": (end.get("battle_time") or "")[:8],
        }

    streaks: list[dict] = []
    current_type = rows[0]["result"]
    current_start = rows[0]
    current_end = rows[0]
    current_length = 1

    for row in rows[1:]:
        if row["result"] == current_type:
            current_length += 1
            current_end = row
        else:
            streaks.append(
                _finish_streak(current_type, current_length, current_start, current_end)
            )
            current_type = row["result"]
            current_start = row
            current_end = row
            current_length = 1

    streaks.append(
        _finish_streak(current_type, current_length, current_start, current_end)
    )

    win_streaks = [s for s in streaks if s["type"] == "win"]
    loss_streaks = [s for s in streaks if s["type"] == "loss"]

    return {
        "current_streak": streaks[-1] if streaks else None,
        "longest_win_streak": (
            max(win_streaks, key=lambda s: s["length"]) if win_streaks else None
        ),
        "longest_loss_streak": (
            max(loss_streaks, key=lambda s: s["length"]) if loss_streaks else None
        ),
        "streaks": streaks,
    }


def get_rolling_stats(session: Session, window: int = 35, ladder_only: bool = False) -> dict:
    """Get win rate over the last N games.

    Args:
        session: SQLAlchemy session.
        window: Number of recent games to analyze.
        ladder_only: If True, only include ladder (PvP) battles.

    Returns:
        Dict with total, wins, losses, draws, win_rate, three_crowns,
        avg_crowns, trophy_change.
    """
    stmt = (
        select(
            Battle.result,
            Battle.player_crowns,
            Battle.opponent_crowns,
            Battle.player_trophy_change,
            Battle.player_starting_trophies,
            Battle.battle_time,
            Battle.battle_type,
        )
        .order_by(Battle.battle_time.desc())
    )
    stmt = _apply_ladder_filter(stmt, ladder_only)
    stmt = stmt.limit(window)
    rows = [row._asdict() for row in session.execute(stmt).all()]

    if not rows:
        return {
            "total": 0, "wins": 0, "losses": 0, "draws": 0,
            "win_rate": 0.0, "three_crowns": 0, "avg_crowns": 0.0,
            "trophy_change": 0,
        }

    total = len(rows)
    wins = sum(1 for r in rows if r["result"] == "win")
    losses = sum(1 for r in rows if r["result"] == "loss")
    draws = sum(1 for r in rows if r["result"] == "draw")
    three_crowns = sum(1 for r in rows if r["player_crowns"] == 3)
    total_crowns = sum(r["player_crowns"] or 0 for r in rows)
    trophy_change = sum(r["player_trophy_change"] or 0 for r in rows)

    return {
        "total": total,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": round(wins / total * 100, 1) if total > 0 else 0.0,
        "three_crowns": three_crowns,
        "avg_crowns": round(total_crowns / total, 2) if total > 0 else 0.0,
        "trophy_change": trophy_change,
    }


def get_trophy_history(session: Session, ladder_only: bool = False) -> list[dict]:
    """Get trophy progression over time from battle data.

    Returns:
        List of dicts with battle_time, trophies (after battle), result,
        player_trophy_change, ordered chronologically.
    """
    stmt = (
        select(
            Battle.battle_time,
            Battle.player_starting_trophies,
            Battle.player_trophy_change,
            Battle.result,
        )
        .where(Battle.player_starting_trophies.isnot(None))
        .order_by(Battle.battle_time.asc())
    )
    stmt = _apply_ladder_filter(stmt, ladder_only)
    results = []
    for row in session.execute(stmt).all():
        d = row._asdict()
        change = d["player_trophy_change"] or 0
        d["trophies"] = (d["player_starting_trophies"] or 0) + change
        results.append(d)
    return results


def get_archetype_stats(session: Session, min_battles: int = 3, ladder_only: bool = False) -> list[dict]:
    """Cluster opponent decks into archetypes and show win rates.

    Args:
        session: SQLAlchemy session.
        min_battles: Minimum battles to include an archetype.
        ladder_only: If True, only include ladder (PvP) battles.

    Returns:
        List of dicts with archetype, total, wins, losses, win_rate.
    """
    stmt = select(Battle.opponent_deck, Battle.result).where(
        Battle.opponent_deck.isnot(None)
    )
    stmt = _apply_ladder_filter(stmt, ladder_only)

    archetype_data: dict[str, dict] = {}
    for row in session.execute(stmt).all():
        try:
            deck = json.loads(row.opponent_deck)
        except (json.JSONDecodeError, TypeError):
            continue
        archetype = classify_archetype(deck)
        if archetype not in archetype_data:
            archetype_data[archetype] = {"wins": 0, "losses": 0, "draws": 0}
        entry = archetype_data[archetype]
        if row.result == "win":
            entry["wins"] += 1
        elif row.result == "loss":
            entry["losses"] += 1
        else:
            entry["draws"] += 1

    results = []
    for archetype, data in archetype_data.items():
        total = data["wins"] + data["losses"] + data["draws"]
        if total >= min_battles:
            results.append({
                "archetype": archetype,
                "total": total,
                "wins": data["wins"],
                "losses": data["losses"],
                "win_rate": round(data["wins"] / total * 100, 1) if total > 0 else 0.0,
            })
    return sorted(results, key=lambda x: x["total"], reverse=True)


def get_snapshot_diff(session: Session) -> Optional[dict]:
    """Compare the two most recent player snapshots.

    Returns:
        Dict with field-level diffs, or None if fewer than 2 snapshots.
    """
    stmt = select(PlayerSnapshot).order_by(PlayerSnapshot.id.desc()).limit(2)
    rows = session.scalars(stmt).all()
    if len(rows) < 2:
        return None

    current, previous = rows[0], rows[1]
    diff_fields = [
        "trophies", "best_trophies", "wins", "losses", "battle_count",
        "three_crown_wins", "war_day_wins", "total_donations",
    ]
    diff: dict = {}
    for field in diff_fields:
        old_val = getattr(previous, field, None) or 0
        new_val = getattr(current, field, None) or 0
        diff[field] = new_val - old_val
    return diff
