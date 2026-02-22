"""Batch scraping for the training data corpus (ADR-007).

Iterates over corpus players, fetches their battles from the CR API,
and scrapes replay data from RoyaleAPI.
"""

import asyncio
import logging
import time

from sqlalchemy.orm import Session

from tracker import analytics
from tracker.api import (
    AuthError,
    ClashRoyaleAPI,
    ConnectionError_,
    NotFoundError,
    RateLimitError,
    ServerError,
)
from tracker.corpus import get_corpus_players, mark_player_scraped
from tracker.metrics import (
    BATTLES_SCRAPED,
    CORPUS_PLAYERS_ACTIVE,
    CORPUS_PLAYERS_DEACTIVATED,
    REPLAYS_FETCHED,
    REPLAYS_FAILED,
    SCRAPE_RUNS,
)
from tracker.models import Battle
from tracker.replays import fetch_replays

logger = logging.getLogger(__name__)

# Rate limits
MAX_REPLAYS_PER_RUN = 5000  # per invocation; real throttle is RoyaleAPI/Cloudflare
DELAY_BETWEEN_PLAYERS = 5   # seconds between switching players


def scrape_corpus_battles(
    session: Session,
    api: ClashRoyaleAPI,
    limit: int = 20,
) -> dict:
    """Fetch battles for corpus players from the CR API.

    Handles errors by type:
    - 404: deactivate player permanently
    - 429/5xx/network: skip and retry next run
    - 401/403: stop entire run (API key issue)

    Args:
        session: SQLAlchemy session.
        api: ClashRoyaleAPI client.
        limit: Maximum number of corpus players to process.

    Returns:
        Dict with scrape statistics.
    """
    from sqlalchemy import select, func
    from tracker.models import PlayerCorpus
    total_active = session.scalar(
        select(func.count()).select_from(PlayerCorpus).where(PlayerCorpus.active == 1)
    ) or 0
    CORPUS_PLAYERS_ACTIVE.set(total_active)

    players = get_corpus_players(session, active_only=True, limit=limit)
    logger.info("Scraping battles for %d corpus players.", len(players))

    stats = {
        "total_players": 0,
        "total_new_battles": 0,
        "transient_errors": 0,
        "deactivated": 0,
        "auth_error": False,
    }

    for player in players:
        tag = player.player_tag.lstrip("#")
        try:
            battles = api.get_battle_log(tag)
            new_count = 0

            for battle in battles:
                battle_type = battle.get("type", "")
                if battle_type not in ("PvP", "pathOfLegend", "riverRacePvP"):
                    continue

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
                    BATTLES_SCRAPED.labels(corpus="top_ladder").inc()

            mark_player_scraped(session, player.player_tag, games=new_count)
            stats["total_new_battles"] += new_count
            stats["total_players"] += 1

            if new_count > 0:
                logger.info(
                    "  %s (%s): %d new battles",
                    player.player_name or tag, tag, new_count,
                )

            time.sleep(DELAY_BETWEEN_PLAYERS)

        except NotFoundError:
            logger.warning("Player %s not found (404), deactivating.", tag)
            player.active = 0
            session.commit()
            stats["deactivated"] += 1
            CORPUS_PLAYERS_DEACTIVATED.inc()

        except AuthError as e:
            logger.error("API auth failed — check CR_API_KEY: %s", e)
            stats["auth_error"] = True
            break  # stop entire run

        except (RateLimitError, ServerError, ConnectionError_) as e:
            logger.warning("Transient error for %s, skipping: %s", tag, e)
            stats["transient_errors"] += 1

        except Exception as e:
            logger.warning("Unexpected error for %s: %s", tag, e)
            stats["transient_errors"] += 1

    outcome = "failed" if stats["auth_error"] else (
        "partial" if stats["transient_errors"] > 0 else "success"
    )
    SCRAPE_RUNS.labels(scrape_type="battles", outcome=outcome).inc()

    logger.info(
        "Corpus battle scrape: %d players, %d new battles, "
        "%d transient errors, %d deactivated%s",
        stats["total_players"],
        stats["total_new_battles"],
        stats["transient_errors"],
        stats["deactivated"],
        " — STOPPED: auth error" if stats["auth_error"] else "",
    )
    return stats


async def scrape_corpus_replays(
    session: Session,
    browser_ws: str | None = None,
    state_path: str | None = None,
    limit: int = 20,
    replays_per_player: int = 5,
) -> dict:
    """Scrape replay data for corpus players from RoyaleAPI.

    Stops early if session expires (3 consecutive auth failures).

    Args:
        session: SQLAlchemy session.
        browser_ws: CDP endpoint URL for the Chromium browser.
        state_path: Path to RoyaleAPI session state JSON.
        limit: Maximum corpus players to process.
        replays_per_player: Max replays per player per run.

    Returns:
        Dict with scrape statistics.
    """
    players = get_corpus_players(session, active_only=True, limit=limit)
    logger.info("Scraping replays for up to %d corpus players.", len(players))

    stats = {
        "total_players": 0,
        "total_replays": 0,
        "transient_errors": 0,
        "session_expired": False,
    }

    consecutive_auth_failures = 0

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

            if count == -1:
                # Session expired
                consecutive_auth_failures += 1
                stats["transient_errors"] += 1
                if consecutive_auth_failures >= 3:
                    logger.error(
                        "RoyaleAPI session expired — 3 consecutive auth failures. "
                        "Stopping corpus replay scrape."
                    )
                    stats["session_expired"] = True
                    break
            else:
                consecutive_auth_failures = 0
                if count > 0:
                    mark_player_scraped(
                        session, player.player_tag, replays=count
                    )
                    stats["total_replays"] += count
                    logger.info(
                        "  %s (%s): %d replays",
                        player.player_name or tag, tag, count,
                    )

            stats["total_players"] += 1

            if stats["total_replays"] >= MAX_REPLAYS_PER_RUN:
                logger.info("Per-run replay limit reached (%d).", MAX_REPLAYS_PER_RUN)
                break

        except Exception as e:
            logger.warning("Error scraping replays for %s: %s", tag, e)
            stats["transient_errors"] += 1

    outcome = "failed" if stats["session_expired"] else (
        "partial" if stats["transient_errors"] > 0 else "success"
    )
    SCRAPE_RUNS.labels(scrape_type="replays", outcome=outcome).inc()

    logger.info(
        "Corpus replay scrape: %d players, %d replays, "
        "%d errors%s",
        stats["total_players"],
        stats["total_replays"],
        stats["transient_errors"],
        " — STOPPED: session expired" if stats["session_expired"] else "",
    )
    return stats


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
