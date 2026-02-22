"""Batch replay scraping for the training data corpus (ADR-007).

Iterates over corpus players, fetches their battles from the CR API,
and scrapes replay data from RoyaleAPI.
"""

import asyncio
import logging
import time

from sqlalchemy.orm import Session

from tracker import analytics
from tracker.api import ClashRoyaleAPI
from tracker.corpus import get_corpus_players, mark_player_scraped
from tracker.models import Battle, PlayerCorpus
from tracker.replays import fetch_replays

logger = logging.getLogger(__name__)

# Rate limits (ADR-007 §7)
MAX_REPLAYS_PER_DAY = 1000
DELAY_BETWEEN_PLAYERS = 5  # seconds between switching players


def scrape_corpus_battles(
    session: Session,
    api: ClashRoyaleAPI,
    limit: int = 20,
) -> dict:
    """Fetch battles for corpus players from the CR API.

    This only collects battle data (no replay scraping).
    Uses the same dedup and storage as personal battles.

    Args:
        session: SQLAlchemy session.
        api: ClashRoyaleAPI client.
        limit: Maximum number of corpus players to process.

    Returns:
        Dict with total_players, total_new_battles counts.
    """
    players = get_corpus_players(session, active_only=True, limit=limit)
    logger.info("Scraping battles for %d corpus players.", len(players))

    total_new = 0
    players_processed = 0

    for player in players:
        tag = player.player_tag.lstrip("#")
        try:
            battles = api.get_battle_log(tag)
            new_count = 0

            for battle in battles:
                # Only store competitive ladder battles (ADR-007 §4)
                battle_type = battle.get("type", "")
                if battle_type not in ("PvP", "pathOfLegend", "riverRacePvP"):
                    continue

                # Trophy floor: 7000+ for regular ladder, any for Path of Legend
                # (PoL startingTrophies is the elo rating, ~3000+, already top-tier)
                if battle_type == "PvP":
                    team = battle.get("team", [{}])[0]
                    trophies = team.get("startingTrophies", 0)
                    if trophies and trophies < 7000:
                        continue

                battle_id, is_new = analytics.store_battle(
                    session, battle, player.player_tag, corpus="top_ladder"
                )
                if is_new:
                    new_count += 1

            mark_player_scraped(session, player.player_tag, games=new_count)
            total_new += new_count
            players_processed += 1

            if new_count > 0:
                logger.info(
                    "  %s (%s): %d new battles",
                    player.player_name or tag, tag, new_count,
                )

            time.sleep(DELAY_BETWEEN_PLAYERS)

        except Exception as e:
            logger.warning("Error scraping %s: %s", tag, e)
            continue

    logger.info(
        "Corpus battle scrape: %d players, %d new battles.",
        players_processed, total_new,
    )
    return {"total_players": players_processed, "total_new_battles": total_new}


async def scrape_corpus_replays(
    session: Session,
    browser_ws: str | None = None,
    state_path: str | None = None,
    limit: int = 20,
    replays_per_player: int = 5,
) -> dict:
    """Scrape replay data for corpus players from RoyaleAPI.

    Iterates over corpus players with unfetched battles and
    collects replay event data.

    Args:
        session: SQLAlchemy session.
        browser_ws: CDP endpoint URL for the Chromium browser.
        state_path: Path to RoyaleAPI session state JSON.
        limit: Maximum corpus players to process.
        replays_per_player: Max replays per player per run.

    Returns:
        Dict with total_players, total_replays counts.
    """
    players = get_corpus_players(session, active_only=True, limit=limit)
    logger.info("Scraping replays for up to %d corpus players.", len(players))

    total_replays = 0
    players_processed = 0

    for player in players:
        tag = player.player_tag.lstrip("#")

        # Check if this player has unfetched battles
        unfetched_count = (
            session.query(Battle)
            .filter(
                Battle.replay_fetched == 0,
                Battle.battle_type.in_(["PvP", "pathOfLegend", "riverRacePvP"]),
                Battle.player_tag.like(f"%{tag}%"),
                Battle.corpus == "top_ladder",
            )
            .count()
        )
        if unfetched_count == 0:
            continue

        try:
            count = await fetch_replays(
                session,
                tag,
                browser_ws=browser_ws,
                state_path=state_path,
                limit=replays_per_player,
            )
            if count > 0:
                mark_player_scraped(
                    session, player.player_tag, replays=count
                )
                total_replays += count
                logger.info(
                    "  %s (%s): %d replays",
                    player.player_name or tag, tag, count,
                )

            players_processed += 1

            if total_replays >= MAX_REPLAYS_PER_DAY:
                logger.info("Daily replay limit reached (%d).", MAX_REPLAYS_PER_DAY)
                break

        except Exception as e:
            logger.warning("Error scraping replays for %s: %s", tag, e)
            continue

    logger.info(
        "Corpus replay scrape: %d players, %d replays.",
        players_processed, total_replays,
    )
    return {"total_players": players_processed, "total_replays": total_replays}


def run_scrape_corpus_replays(
    session: Session,
    browser_ws: str | None = None,
    state_path: str | None = None,
    limit: int = 20,
    replays_per_player: int = 5,
) -> dict:
    """Synchronous wrapper for scrape_corpus_replays."""
    return asyncio.run(
        scrape_corpus_replays(
            session, browser_ws, state_path, limit, replays_per_player
        )
    )
