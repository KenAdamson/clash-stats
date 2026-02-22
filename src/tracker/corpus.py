"""Training data corpus management (ADR-007).

Collects top-ladder player tags from the CR API and manages the
player_corpus table for batch replay scraping.
"""

import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from tracker.api import ClashRoyaleAPI
from tracker.models import Battle, PlayerCorpus

logger = logging.getLogger(__name__)


def update_top_ladder(
    session: Session,
    api: ClashRoyaleAPI,
    location_id: str = "global",
    limit: int = 200,
) -> int:
    """Fetch top-ladder player tags and upsert into player_corpus.

    Args:
        session: SQLAlchemy session.
        api: ClashRoyaleAPI client.
        location_id: Location ID or 'global'.
        limit: Number of players to fetch (max 200).

    Returns:
        Number of new players added.
    """
    players = api.get_top_players(location_id=location_id, limit=limit)
    logger.info("Fetched %d players from %s leaderboard.", len(players), location_id)

    added = 0
    for p in players:
        tag = p.get("tag", "").lstrip("#")
        if not tag:
            continue

        existing = session.get(PlayerCorpus, f"#{tag}")
        if existing:
            # Update trophy range and name
            trophies = p.get("trophies", 0)
            existing.player_name = p.get("name")
            if existing.trophy_range_high is None or trophies > existing.trophy_range_high:
                existing.trophy_range_high = trophies
            if existing.trophy_range_low is None or trophies < existing.trophy_range_low:
                existing.trophy_range_low = trophies
        else:
            trophies = p.get("trophies", 0)
            session.add(PlayerCorpus(
                player_tag=f"#{tag}",
                player_name=p.get("name"),
                source="top_ladder",
                trophy_range_low=trophies,
                trophy_range_high=trophies,
                active=1,
            ))
            added += 1

    session.commit()
    logger.info("Corpus update: %d new players, %d total.", added, len(players))
    return added


def add_manual_player(
    session: Session,
    player_tag: str,
    source: str = "manual",
    player_name: Optional[str] = None,
) -> bool:
    """Add a player to the corpus manually.

    Args:
        session: SQLAlchemy session.
        player_tag: Player tag (with or without #).
        source: Provenance label.
        player_name: Optional display name.

    Returns:
        True if added, False if already exists.
    """
    tag = f"#{player_tag.lstrip('#')}"
    if session.get(PlayerCorpus, tag):
        return False

    session.add(PlayerCorpus(
        player_tag=tag,
        player_name=player_name,
        source=source,
        active=1,
    ))
    session.commit()
    return True


def get_corpus_players(
    session: Session,
    active_only: bool = True,
    source: Optional[str] = None,
    limit: Optional[int] = None,
) -> list[PlayerCorpus]:
    """Get corpus players, ordered by least recently scraped.

    Args:
        session: SQLAlchemy session.
        active_only: Only return active players.
        source: Filter by source type.
        limit: Maximum players to return.

    Returns:
        List of PlayerCorpus objects.
    """
    stmt = select(PlayerCorpus)
    if active_only:
        stmt = stmt.where(PlayerCorpus.active == 1)
    if source:
        stmt = stmt.where(PlayerCorpus.source == source)

    # Prioritize players never scraped, then least recently scraped
    stmt = stmt.order_by(
        PlayerCorpus.last_scraped.is_(None).desc(),
        PlayerCorpus.last_scraped.asc(),
    )
    if limit:
        stmt = stmt.limit(limit)

    return list(session.scalars(stmt).all())


def mark_player_scraped(
    session: Session,
    player_tag: str,
    games: int = 0,
    replays: int = 0,
) -> None:
    """Update scraping stats for a corpus player.

    Args:
        session: SQLAlchemy session.
        player_tag: Player tag (with #).
        games: Number of new games scraped.
        replays: Number of new replays scraped.
    """
    player = session.get(PlayerCorpus, player_tag)
    if player:
        player.games_scraped = (player.games_scraped or 0) + games
        player.replays_scraped = (player.replays_scraped or 0) + replays
        player.last_scraped = datetime.utcnow()
        session.commit()


def get_corpus_stats(session: Session) -> dict:
    """Get summary statistics about the corpus.

    Returns:
        Dict with counts, source breakdown, and coverage stats.
    """
    total_players = session.scalar(
        select(func.count()).select_from(PlayerCorpus)
    ) or 0
    active_players = session.scalar(
        select(func.count()).select_from(PlayerCorpus).where(PlayerCorpus.active == 1)
    ) or 0

    # Games by corpus type
    corpus_counts = {}
    rows = session.execute(
        select(Battle.corpus, func.count()).group_by(Battle.corpus)
    ).all()
    for row in rows:
        corpus_counts[row[0] or "personal"] = row[1]

    # Source breakdown
    source_counts = {}
    rows = session.execute(
        select(PlayerCorpus.source, func.count()).group_by(PlayerCorpus.source)
    ).all()
    for row in rows:
        source_counts[row[0]] = row[1]

    # Replay coverage
    total_battles = session.scalar(
        select(func.count()).select_from(Battle)
    ) or 0
    battles_with_replays = session.scalar(
        select(func.count()).select_from(Battle).where(Battle.replay_fetched == 1)
    ) or 0

    return {
        "total_players": total_players,
        "active_players": active_players,
        "source_breakdown": source_counts,
        "battles_by_corpus": corpus_counts,
        "total_battles": total_battles,
        "battles_with_replays": battles_with_replays,
        "replay_coverage_pct": round(
            battles_with_replays / total_battles * 100, 1
        ) if total_battles > 0 else 0.0,
    }
