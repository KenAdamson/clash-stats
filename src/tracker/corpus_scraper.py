"""Batch scraping for the training data corpus (ADR-007).

Iterates over corpus players, fetches their battles from the CR API,
and scrapes replay data from RoyaleAPI.
"""

import asyncio
import logging
import os
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
    CORPUS_PLAYERS_ACTIVE,
    CORPUS_PLAYERS_DEACTIVATED,
    REPLAYS_FETCHED,
    REPLAYS_FAILED,
    SCRAPE_RUNS,
    flush_metrics,
)

FLUSH_EVERY_N_PLAYERS = 5  # flush metrics to disk every N players
from tracker.models import Battle
from tracker.replays import (
    DEFAULT_BROWSER_CDP,
    DEFAULT_SESSION_PATH,
    fetch_replays,
    fetch_replays_for_player,
)

logger = logging.getLogger(__name__)

# Rate limits
MAX_REPLAYS_PER_RUN = 5000  # per invocation; real throttle is RoyaleAPI/Cloudflare
DELAY_BETWEEN_PLAYERS = 2   # seconds between CR API calls per player

# Replay staleness: battles older than this are unlikely to still be on RoyaleAPI.
# RoyaleAPI's battle page typically only shows the last ~24-48h of games.
STALE_REPLAY_DAYS = 2


def mark_stale_replays(session: Session, max_age_days: int = STALE_REPLAY_DAYS) -> int:
    """Mark old unfetched battles as permanently missed (replay_fetched=2).

    RoyaleAPI only keeps replays for recent battles. Unfetched battles older
    than max_age_days will never be found, so mark them to stop wasting
    pagination cycles looking for them.

    Args:
        session: SQLAlchemy session.
        max_age_days: Age threshold in days.

    Returns:
        Number of battles marked as stale.
    """
    from datetime import datetime, timedelta, timezone
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    # battle_time is stored as "20260228T200232.000Z" — lexicographic compare works
    cutoff_str = cutoff.strftime("%Y%m%dT%H%M%S.000Z")

    from sqlalchemy import update
    from sqlalchemy.exc import OperationalError
    try:
        result = session.execute(
            update(Battle)
            .where(
                Battle.replay_fetched == 0,
                Battle.battle_type.in_(["PvP", "pathOfLegend", "riverRacePvP"]),
                Battle.battle_time < cutoff_str,
            )
            .values(replay_fetched=2)
        )
        count = result.rowcount
        session.commit()
        if count > 0:
            logger.info("Marked %d stale battles as replay_fetched=2 (older than %d days).", count, max_age_days)
        return count
    except OperationalError as e:
        session.rollback()
        logger.warning("Stale replay marking skipped (database locked): %s", e)
        return 0


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

            mark_player_scraped(session, player.player_tag, games=new_count)
            stats["total_new_battles"] += new_count
            stats["total_players"] += 1

            if new_count > 0:
                logger.info(
                    "  %s (%s): %d new battles",
                    player.player_name or tag, tag, new_count,
                )

            if stats["total_players"] % FLUSH_EVERY_N_PLAYERS == 0:
                flush_metrics("corpus_scrape")
                logger.info(
                    "Progress: %d players, %d new battles so far",
                    stats["total_players"], stats["total_new_battles"],
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
    max_pages: int = 5,
    concurrency: int = 1,
) -> dict:
    """Scrape replay data for corpus players from RoyaleAPI.

    Opens `concurrency` browser tabs and processes players in parallel.
    Stops early if session expires (3 consecutive auth failures).

    Args:
        session: SQLAlchemy session.
        browser_ws: CDP endpoint URL for the Chromium browser.
        state_path: Path to RoyaleAPI session state JSON.
        limit: Maximum corpus players to process.
        replays_per_player: Max replays per player per run.
        max_pages: Pagination depth per player (1=fast/recent only).
        concurrency: Number of parallel browser tabs.

    Returns:
        Dict with scrape statistics.
    """
    from playwright.async_api import async_playwright

    cdp_url = browser_ws or os.environ.get("BROWSER_WS_URL", DEFAULT_BROWSER_CDP)
    save_path = state_path or os.environ.get(
        "ROYALEAPI_SESSION_PATH", DEFAULT_SESSION_PATH
    )

    # Mark battles too old for RoyaleAPI as permanently missed
    mark_stale_replays(session)

    # Filter to players with unfetched battles
    all_players = get_corpus_players(session, active_only=True, limit=limit)
    players = []
    for p in all_players:
        tag = p.player_tag.lstrip("#")
        unfetched_count = (
            session.query(Battle)
            .filter(
                Battle.replay_fetched == 0,
                Battle.battle_type.in_(["PvP", "pathOfLegend", "riverRacePvP"]),
                Battle.player_tag == f"#{tag}",
                Battle.corpus == "top_ladder",
            )
            .count()
        )
        if unfetched_count > 0:
            players.append(p)

    if not players:
        logger.info("No corpus players with unfetched battles.")
        return {"total_players": 0, "total_replays": 0, "transient_errors": 0,
                "session_expired": False}

    effective_concurrency = min(concurrency, len(players))
    logger.info(
        "Scraping replays for %d corpus players (%d concurrent tabs).",
        len(players), effective_concurrency,
    )

    stats = {
        "total_players": 0,
        "total_replays": 0,
        "transient_errors": 0,
        "session_expired": False,
    }
    consecutive_auth_failures = 0

    async with async_playwright() as pw:
        browser = await pw.chromium.connect_over_cdp(cdp_url)
        context = await browser.new_context(
            storage_state=save_path,
            viewport={"width": 1280, "height": 720},
        )

        # Create tab pool with resource blocking for speed
        pages = []
        for _ in range(effective_concurrency):
            page = await context.new_page()
            await page.route(
                "**/*.{png,jpg,jpeg,gif,svg,ico,woff,woff2,ttf,eot,css}",
                lambda route: route.abort(),
            )
            await page.route("**/analytics**", lambda route: route.abort())
            await page.route("**/ads**", lambda route: route.abort())
            await page.route("**/tracking**", lambda route: route.abort())
            await page.route("**/google-analytics**", lambda route: route.abort())
            await page.route("**/gtag**", lambda route: route.abort())
            await page.route("**/adsbygoogle**", lambda route: route.abort())
            pages.append(page)
        logger.info("Opened %d browser tabs (resources blocked).", len(pages))

        async def _worker(page, player, stagger_secs=0):
            """Scrape one player on one tab."""
            if stagger_secs > 0:
                await asyncio.sleep(stagger_secs)
            tag = player.player_tag.lstrip("#")
            count = await fetch_replays_for_player(
                session, page, tag,
                limit=replays_per_player,
                max_pages=max_pages,
            )
            return player, tag, count

        # Process players in batches of `concurrency`
        for batch_start in range(0, len(players), effective_concurrency):
            if stats["session_expired"]:
                break
            if stats["total_replays"] >= MAX_REPLAYS_PER_RUN:
                logger.info("Per-run replay limit reached (%d).", MAX_REPLAYS_PER_RUN)
                break

            batch = players[batch_start:batch_start + effective_concurrency]
            tasks = [
                _worker(pages[i], batch[i], stagger_secs=i * 2)
                for i in range(len(batch))
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    logger.warning("Worker error: %s", result)
                    stats["transient_errors"] += 1
                    continue

                player, tag, count = result

                if count == -1:
                    consecutive_auth_failures += 1
                    stats["transient_errors"] += 1
                    if consecutive_auth_failures >= 3:
                        logger.error(
                            "RoyaleAPI session expired — 3 consecutive auth "
                            "failures. Stopping corpus replay scrape."
                        )
                        stats["session_expired"] = True
                else:
                    consecutive_auth_failures = 0
                    if count > 0:
                        mark_player_scraped(session, player.player_tag, replays=count)
                        stats["total_replays"] += count
                        logger.info(
                            "  %s (%s): %d replays",
                            player.player_name or tag, tag, count,
                        )

                stats["total_players"] += 1

            # Periodic metric flush after each batch
            flush_metrics("corpus_replays")
            logger.info(
                "Progress: %d/%d players, %d replays so far",
                stats["total_players"], len(players), stats["total_replays"],
            )

        # Cleanup
        for page in pages:
            await page.close()
        await context.close()

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
    max_pages: int = 5,
    concurrency: int = 1,
) -> dict:
    """Synchronous wrapper for scrape_corpus_replays."""
    return asyncio.run(
        scrape_corpus_replays(
            session, browser_ws, state_path, limit,
            replays_per_player, max_pages, concurrency,
        )
    )


async def scrape_corpus_combined(
    session: Session,
    api: ClashRoyaleAPI,
    browser_ws: str | None = None,
    state_path: str | None = None,
    limit: int = 200,
    replays_per_player: int = 25,
    max_pages: int = 2,
    concurrency: int = 12,
) -> dict:
    """Fetch battles AND replays for corpus players in a single pass.

    For each batch of players: fetches battles from the CR API sequentially,
    then immediately dispatches the same batch to browser tabs for concurrent
    replay scraping. This maximizes the overlap between the CR API's battle
    window and RoyaleAPI's cached battles.

    Args:
        session: SQLAlchemy session.
        api: ClashRoyaleAPI client.
        browser_ws: CDP endpoint URL for the Chromium browser.
        state_path: Path to RoyaleAPI session state JSON.
        limit: Maximum corpus players to process.
        replays_per_player: Max replays per player per run.
        max_pages: Pagination depth on RoyaleAPI.
        concurrency: Number of parallel browser tabs.

    Returns:
        Dict with combined scrape statistics.
    """
    from playwright.async_api import async_playwright
    from sqlalchemy import select, func
    from tracker.models import PlayerCorpus

    cdp_url = browser_ws or os.environ.get("BROWSER_WS_URL", DEFAULT_BROWSER_CDP)
    save_path = state_path or os.environ.get(
        "ROYALEAPI_SESSION_PATH", DEFAULT_SESSION_PATH
    )

    # Mark stale replays once at start
    mark_stale_replays(session)

    # Set corpus gauge
    total_active = session.scalar(
        select(func.count()).select_from(PlayerCorpus).where(PlayerCorpus.active == 1)
    ) or 0
    CORPUS_PLAYERS_ACTIVE.set(total_active)

    players = get_corpus_players(session, active_only=True, limit=limit)
    if not players:
        logger.info("No active corpus players.")
        return {
            "total_players": 0, "total_new_battles": 0, "total_replays": 0,
            "deactivated": 0, "battle_errors": 0, "replay_errors": 0,
            "auth_error": False, "session_expired": False,
        }

    effective_concurrency = min(concurrency, len(players))
    logger.info(
        "Combined scrape: %d corpus players (%d concurrent tabs).",
        len(players), effective_concurrency,
    )

    stats = {
        "total_players": 0,
        "total_new_battles": 0,
        "total_replays": 0,
        "deactivated": 0,
        "battle_errors": 0,
        "replay_errors": 0,
        "auth_error": False,
        "session_expired": False,
    }
    consecutive_auth_failures = 0

    async with async_playwright() as pw:
        browser = await pw.chromium.connect_over_cdp(cdp_url)
        context = await browser.new_context(
            storage_state=save_path,
            viewport={"width": 1280, "height": 720},
        )

        # Create tab pool with resource blocking
        pages = []
        for _ in range(effective_concurrency):
            page = await context.new_page()
            await page.route(
                "**/*.{png,jpg,jpeg,gif,svg,ico,woff,woff2,ttf,eot,css}",
                lambda route: route.abort(),
            )
            await page.route("**/analytics**", lambda route: route.abort())
            await page.route("**/ads**", lambda route: route.abort())
            await page.route("**/tracking**", lambda route: route.abort())
            await page.route("**/google-analytics**", lambda route: route.abort())
            await page.route("**/gtag**", lambda route: route.abort())
            await page.route("**/adsbygoogle**", lambda route: route.abort())
            pages.append(page)
        logger.info("Opened %d browser tabs (resources blocked).", len(pages))

        async def _replay_worker(page, player, stagger_secs=0):
            """Scrape replays for one player on one tab."""
            if stagger_secs > 0:
                await asyncio.sleep(stagger_secs)
            tag = player.player_tag.lstrip("#")
            count = await fetch_replays_for_player(
                session, page, tag,
                limit=replays_per_player,
                max_pages=max_pages,
            )
            return player, tag, count

        # Process players in batches of `concurrency`
        for batch_start in range(0, len(players), effective_concurrency):
            if stats["auth_error"] or stats["session_expired"]:
                break
            if stats["total_replays"] >= MAX_REPLAYS_PER_RUN:
                logger.info("Per-run replay limit reached (%d).", MAX_REPLAYS_PER_RUN)
                break

            batch = players[batch_start:batch_start + effective_concurrency]

            # --- Phase 1: Fetch battles from CR API (sequential) ---
            batch_battles = 0
            for player in batch:
                if stats["auth_error"]:
                    break
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

                    mark_player_scraped(session, player.player_tag, games=new_count)
                    batch_battles += new_count
                    stats["total_new_battles"] += new_count

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
                    break

                except (RateLimitError, ServerError, ConnectionError_) as e:
                    logger.warning("Battle fetch transient error for %s: %s", tag, e)
                    stats["battle_errors"] += 1

                except Exception as e:
                    logger.warning("Battle fetch unexpected error for %s: %s", tag, e)
                    stats["battle_errors"] += 1

            if stats["auth_error"]:
                break

            logger.info(
                "Batch %d: %d new battles from %d players. Starting replay scrape...",
                batch_start // effective_concurrency + 1,
                batch_battles, len(batch),
            )

            # --- Phase 2: Scrape replays from RoyaleAPI (concurrent) ---
            # Filter to active players only (some may have been deactivated above)
            active_batch = [p for p in batch if p.active == 1]
            if not active_batch:
                continue

            tasks = [
                _replay_worker(pages[i], active_batch[i], stagger_secs=i * 2)
                for i in range(len(active_batch))
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    logger.warning("Replay worker error: %s", result)
                    stats["replay_errors"] += 1
                    continue

                player, tag, count = result

                if count == -1:
                    consecutive_auth_failures += 1
                    stats["replay_errors"] += 1
                    if consecutive_auth_failures >= 3:
                        logger.error(
                            "RoyaleAPI session expired — 3 consecutive auth "
                            "failures. Stopping combined scrape."
                        )
                        stats["session_expired"] = True
                else:
                    consecutive_auth_failures = 0
                    if count > 0:
                        mark_player_scraped(session, player.player_tag, replays=count)
                        stats["total_replays"] += count
                        logger.info(
                            "  %s (%s): %d replays",
                            player.player_name or tag, tag, count,
                        )

                stats["total_players"] += 1

            # Periodic metric flush after each batch
            flush_metrics("corpus_combined")
            logger.info(
                "Progress: %d/%d players, %d battles, %d replays so far",
                stats["total_players"], len(players),
                stats["total_new_battles"], stats["total_replays"],
            )

        # Cleanup
        for page in pages:
            await page.close()
        await context.close()

    # Record metrics
    if stats["auth_error"] or stats["session_expired"]:
        outcome = "failed"
    elif stats["battle_errors"] + stats["replay_errors"] > 0:
        outcome = "partial"
    else:
        outcome = "success"
    SCRAPE_RUNS.labels(scrape_type="combined", outcome=outcome).inc()

    logger.info(
        "Combined scrape: %d players, %d battles, %d replays, "
        "%d battle errors, %d replay errors%s%s",
        stats["total_players"],
        stats["total_new_battles"],
        stats["total_replays"],
        stats["battle_errors"],
        stats["replay_errors"],
        " — STOPPED: auth error" if stats["auth_error"] else "",
        " — STOPPED: session expired" if stats["session_expired"] else "",
    )
    return stats


def run_scrape_corpus_combined(
    session: Session,
    api: ClashRoyaleAPI,
    browser_ws: str | None = None,
    state_path: str | None = None,
    limit: int = 200,
    replays_per_player: int = 25,
    max_pages: int = 2,
    concurrency: int = 12,
) -> dict:
    """Synchronous wrapper for scrape_corpus_combined."""
    return asyncio.run(
        scrape_corpus_combined(
            session, api, browser_ws, state_path, limit,
            replays_per_player, max_pages, concurrency,
        )
    )
