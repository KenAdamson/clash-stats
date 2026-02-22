"""RoyaleAPI replay scraper and parser.

Fetches battle replay data from RoyaleAPI using Playwright (connecting to
a remote Chromium browser sidecar) and parses the structured HTML into
card placement events and elixir summaries.
"""

import asyncio
import json
import logging
import os
import re
import time
from pathlib import Path

from bs4 import BeautifulSoup
from sqlalchemy import text, update
from sqlalchemy.orm import Session

from tracker.models import Battle, ReplayEvent, ReplaySummary

logger = logging.getLogger(__name__)

ROYALEAPI_BASE = "https://royaleapi.com"
DEFAULT_BROWSER_CDP = "http://cr-browser:9223"
DEFAULT_SESSION_PATH = "/app/data/royaleapi_session.json"
FETCH_DELAY = 3  # seconds between page loads


# ---------------------------------------------------------------------------
# Pure HTML parsing (no Playwright, no DB — testable with static fixtures)
# ---------------------------------------------------------------------------

def parse_replay_html(html: str) -> dict:
    """Parse RoyaleAPI replay HTML into structured data.

    Args:
        html: Raw HTML string from the /data/replay endpoint.

    Returns:
        Dict with:
          - events: list of card placement event dicts
          - summaries: list of per-side elixir summary dicts
    """
    soup = BeautifulSoup(html, "html.parser")
    events = _parse_events(soup)
    summaries = _parse_summaries(soup)
    return {"events": events, "summaries": summaries}


def _parse_events(soup: BeautifulSoup) -> list[dict]:
    """Extract card placement events from map markers."""
    events = []
    markers = soup.select(".replay_map .marker")

    for marker in markers:
        side_code = marker.get("data-s", "")
        side = "team" if side_code == "t" else "opponent" if side_code == "o" else side_code

        card_name = marker.get("data-c", "")
        game_tick = _int_or_none(marker.get("data-t"))
        arena_x = _int_or_none(marker.get("data-x"))
        arena_y = _int_or_none(marker.get("data-y"))
        ability = _int_or_none(marker.get("data-i", "0"))

        # Play number is in the <span> child
        span = marker.find("span")
        play_number = _int_or_none(span.get_text(strip=True)) if span else 1

        if card_name and game_tick is not None:
            events.append({
                "side": side,
                "card_name": card_name,
                "game_tick": game_tick,
                "arena_x": arena_x or 0,
                "arena_y": arena_y or 0,
                "play_number": play_number or 1,
                "ability_used": ability or 0,
            })

    return events


def _parse_summaries(soup: BeautifulSoup) -> list[dict]:
    """Extract per-side elixir stats from the replay tables."""
    summaries = []
    tables = soup.select(".replay_elixir_table")

    # The HTML has two tables: first is team (blue), second is opponent (red)
    sides = ["team", "opponent"]

    for i, table in enumerate(tables[:2]):
        side = sides[i] if i < len(sides) else f"side_{i}"
        summary = {"side": side}

        rows = table.select("tr")
        for row in rows:
            title_td = row.select_one("td.title")
            if not title_td:
                continue
            label = title_td.get_text(strip=True).lower()

            count_td = row.select_one("td.count")
            elixir_td = row.select_one("td.elixir")

            count_val = _parse_number(count_td.get_text(strip=True)) if count_td else None
            elixir_val = _parse_number(elixir_td.get_text(strip=True)) if elixir_td else None

            if label == "total":
                summary["total_plays"] = count_val
                summary["total_elixir"] = elixir_val
            elif label == "troop":
                summary["troop_plays"] = count_val
                summary["troop_elixir"] = elixir_val
            elif label == "spell":
                summary["spell_plays"] = count_val
                summary["spell_elixir"] = elixir_val
            elif label == "building":
                summary["building_plays"] = count_val
                summary["building_elixir"] = elixir_val
            elif label == "ability":
                summary["ability_plays"] = count_val
                summary["ability_elixir"] = elixir_val
            elif label == "leaked":
                # Leaked row has elixir in a colspan td, no separate count
                summary["elixir_leaked"] = elixir_val

        summaries.append(summary)

    return summaries


def _int_or_none(val) -> int | None:
    """Convert to int or return None."""
    if val is None:
        return None
    try:
        return int(val)
    except (ValueError, TypeError):
        return None


def _parse_number(text: str) -> int | float | None:
    """Extract a number from text that may contain icons or whitespace."""
    if not text:
        return None
    # Strip non-numeric chars except dot and minus
    cleaned = re.sub(r"[^\d.\-]", "", text)
    if not cleaned:
        return None
    try:
        if "." in cleaned:
            return float(cleaned)
        return int(cleaned)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Database storage
# ---------------------------------------------------------------------------

def store_replay_data(session: Session, battle_id: str, data: dict) -> None:
    """Store parsed replay events and summaries in the database.

    Args:
        session: SQLAlchemy session.
        battle_id: The battle_id foreign key.
        data: Output from parse_replay_html().
    """
    for event in data.get("events", []):
        session.add(ReplayEvent(
            battle_id=battle_id,
            side=event["side"],
            card_name=event["card_name"],
            game_tick=event["game_tick"],
            arena_x=event["arena_x"],
            arena_y=event["arena_y"],
            play_number=event["play_number"],
            ability_used=event["ability_used"],
        ))

    for summary in data.get("summaries", []):
        session.add(ReplaySummary(
            battle_id=battle_id,
            side=summary["side"],
            total_plays=summary.get("total_plays"),
            total_elixir=summary.get("total_elixir"),
            troop_plays=summary.get("troop_plays"),
            troop_elixir=summary.get("troop_elixir"),
            spell_plays=summary.get("spell_plays"),
            spell_elixir=summary.get("spell_elixir"),
            building_plays=summary.get("building_plays"),
            building_elixir=summary.get("building_elixir"),
            ability_plays=summary.get("ability_plays"),
            ability_elixir=summary.get("ability_elixir"),
            elixir_leaked=summary.get("elixir_leaked"),
        ))

    # Mark battle as fetched
    session.execute(
        update(Battle).where(Battle.battle_id == battle_id).values(replay_fetched=1)
    )
    session.commit()


# ---------------------------------------------------------------------------
# Playwright scraper (connects to remote browser sidecar)
# ---------------------------------------------------------------------------

async def start_login(browser_ws: str | None = None) -> None:
    """Navigate the sidecar browser to RoyaleAPI login page.

    The user completes Google SSO by interacting with the browser
    via noVNC at http://<host>:6080. Call check_login_and_save()
    to poll for completion.

    Args:
        browser_ws: CDP endpoint URL for the Chromium browser.
    """
    from playwright.async_api import async_playwright

    cdp_url = browser_ws or os.environ.get("BROWSER_WS_URL", DEFAULT_BROWSER_CDP)

    async with async_playwright() as p:
        browser = await p.chromium.connect_over_cdp(cdp_url)
        # Use existing default context (the one visible on noVNC)
        contexts = browser.contexts
        if contexts:
            context = contexts[0]
            page = context.pages[0] if context.pages else await context.new_page()
        else:
            context = await browser.new_context(viewport={"width": 1280, "height": 720})
            page = await context.new_page()
        await page.goto(f"{ROYALEAPI_BASE}/login", wait_until="networkidle")
        logger.info("Navigated to RoyaleAPI login page. Complete login via noVNC.")
        # Don't close — the user interacts via noVNC.
        # The browser context stays open in the sidecar.


async def check_login_and_save(
    browser_ws: str | None = None,
    state_path: str | None = None,
) -> bool:
    """Check if RoyaleAPI login is complete and save session state.

    Args:
        browser_ws: CDP endpoint URL for the Chromium browser.
        state_path: Path to save the browser session state JSON.

    Returns:
        True if authenticated and session saved, False otherwise.
    """
    from playwright.async_api import async_playwright

    cdp_url = browser_ws or os.environ.get("BROWSER_WS_URL", DEFAULT_BROWSER_CDP)
    save_path = state_path or os.environ.get(
        "ROYALEAPI_SESSION_PATH", DEFAULT_SESSION_PATH
    )

    async with async_playwright() as p:
        browser = await p.chromium.connect_over_cdp(cdp_url)
        contexts = browser.contexts
        if not contexts:
            return False

        context = contexts[0]
        pages = context.pages
        if not pages:
            return False

        page = pages[0]
        url = page.url

        # Check if we're on a logged-in page (not /login)
        if "/login" in url or "accounts.google.com" in url:
            return False

        # Save the full browser state (cookies + localStorage)
        await context.storage_state(path=save_path)
        logger.info("RoyaleAPI session saved to %s", save_path)
        return True


async def fetch_replays(
    session: Session,
    player_tag: str,
    browser_ws: str | None = None,
    state_path: str | None = None,
    limit: int = 5,
) -> int:
    """Fetch replay data for battles that haven't been scraped yet.

    Navigates to the player's battles page on RoyaleAPI, extracts
    replay links, fetches each replay's HTML, parses it, and stores
    the results.

    Args:
        session: SQLAlchemy session.
        player_tag: Player tag (with or without #).
        browser_ws: WebSocket URL for the Playwright server.
        state_path: Path to the saved session state JSON.
        limit: Maximum number of replays to fetch per run.

    Returns:
        Number of replays successfully fetched and stored.
    """
    from playwright.async_api import async_playwright

    cdp_url = browser_ws or os.environ.get("BROWSER_WS_URL", DEFAULT_BROWSER_CDP)
    save_path = state_path or os.environ.get(
        "ROYALEAPI_SESSION_PATH", DEFAULT_SESSION_PATH
    )

    if not Path(save_path).exists():
        logger.warning(
            "No RoyaleAPI session found at %s. "
            "Run --replay-login first.",
            save_path,
        )
        return 0

    # Get battles that need replay data
    tag_clean = player_tag.lstrip("#")
    unfetched = (
        session.query(Battle)
        .filter(Battle.replay_fetched == 0, Battle.battle_type.in_(["PvP", "riverRacePvP"]))
        .filter(Battle.player_tag.like(f"%{tag_clean}%"))
        .order_by(Battle.battle_time.desc())
        .limit(limit)
        .all()
    )

    if not unfetched:
        logger.info("No unfetched battles found.")
        return 0

    logger.info("Found %d battles to fetch replays for.", len(unfetched))
    fetched_count = 0

    async with async_playwright() as p:
        browser = await p.chromium.connect_over_cdp(cdp_url)

        # Prefer the existing default context (already authenticated from
        # noVNC login). Fall back to a new context with saved session state.
        contexts = browser.contexts
        if contexts:
            context = contexts[0]
            page = context.pages[0] if context.pages else await context.new_page()
            owns_context = False
        else:
            context = await browser.new_context(
                storage_state=save_path,
                viewport={"width": 1280, "height": 720},
            )
            page = await context.new_page()
            owns_context = True

        # Navigate to player battles page to discover replay links
        battles_url = f"{ROYALEAPI_BASE}/player/{tag_clean}/battles"
        logger.info("Navigating to %s", battles_url)
        await page.goto(battles_url, wait_until="load", timeout=60000)
        # Wait for Cloudflare challenge to resolve if present
        await _wait_for_cloudflare(page)

        # Check if we got redirected to login
        if "/login" in page.url:
            logger.warning(
                "RoyaleAPI session expired. Re-run --replay-login."
            )
            if owns_context:
                await context.close()
            return 0

        # Wait for replay links to appear in DOM
        await page.wait_for_timeout(3000)

        # Extract all replay links from the battles page
        replay_links = await _extract_replay_links(page)
        logger.info("Found %d replay links on battles page.", len(replay_links))

        # Match replay links to unfetched battles
        for battle in unfetched:
            link = _match_replay_link(battle, replay_links)
            if not link:
                logger.debug(
                    "No replay link found for battle %s", battle.battle_id
                )
                # Don't mark as fetched — the battle may appear on
                # a future page load. Only skip for this run.
                continue

            try:
                replay_url = f"{ROYALEAPI_BASE}{link}"
                logger.info("Fetching replay: %s", replay_url)
                resp = await page.goto(replay_url, wait_until="load", timeout=60000)
                await _wait_for_cloudflare(page)

                if resp and resp.status == 200:
                    html = await _extract_replay_html(page)
                    data = parse_replay_html(html)

                    if data["events"]:
                        store_replay_data(session, battle.battle_id, data)
                        fetched_count += 1
                        logger.info(
                            "Stored %d events for battle %s",
                            len(data["events"]),
                            battle.battle_id,
                        )
                    else:
                        logger.warning(
                            "No events found in replay for %s",
                            battle.battle_id,
                        )
                        session.execute(
                            update(Battle)
                            .where(Battle.battle_id == battle.battle_id)
                            .values(replay_fetched=1)
                        )
                        session.commit()
                else:
                    logger.warning(
                        "Failed to fetch replay for %s: HTTP %s",
                        battle.battle_id,
                        resp.status if resp else "no response",
                    )

                # Rate limit
                await asyncio.sleep(FETCH_DELAY)

            except Exception as e:
                logger.error(
                    "Error fetching replay for %s: %s",
                    battle.battle_id,
                    e,
                )

        if owns_context:
            await context.close()

    return fetched_count


async def _extract_replay_html(page) -> str:
    """Extract replay HTML from the page.

    The /data/replay endpoint returns JSON: {"success": true, "html": "..."}
    where the HTML is entity-encoded inside a <pre> tag. This function
    extracts and decodes it. Falls back to raw page content if not JSON.
    """
    content = await page.content()

    # Try to extract JSON from <pre> tag
    pre_match = re.search(r"<pre[^>]*>(.*?)</pre>", content, re.DOTALL)
    if pre_match:
        import html as html_module
        raw_json = html_module.unescape(pre_match.group(1))
        try:
            data = json.loads(raw_json)
            if isinstance(data, dict) and data.get("success") and "html" in data:
                return data["html"]
        except (json.JSONDecodeError, KeyError):
            pass

    # Fallback: return raw page content
    return content


async def _wait_for_cloudflare(page, timeout: int = 15000) -> None:
    """Wait for Cloudflare challenge to resolve if present."""
    try:
        title = await page.title()
        if "just a moment" in title.lower():
            logger.info("Cloudflare challenge detected, waiting...")
            await page.wait_for_function(
                "() => !document.title.toLowerCase().includes('just a moment')",
                timeout=timeout,
            )
            logger.info("Cloudflare challenge passed.")
    except Exception:
        pass  # Not a Cloudflare page, or timed out — continue anyway


async def _extract_replay_links(page) -> list[dict]:
    """Extract replay link data from the player's battles page.

    RoyaleAPI stores replay metadata as data-* attributes on
    .replay_button elements. We extract these and build the
    /data/replay URL from the attributes.

    Returns list of dicts with: url, tag, team_tags, opponent_tags,
    team_crowns, opponent_crowns.
    """
    links = []
    buttons = await page.query_selector_all(".replay_button")

    for btn in buttons:
        replay_tag = await btn.get_attribute("data-replay")
        team_tags = await btn.get_attribute("data-team-tags")
        opponent_tags = await btn.get_attribute("data-opponent-tags")
        team_crowns = await btn.get_attribute("data-team-crowns")
        opponent_crowns = await btn.get_attribute("data-opponent-crowns")

        if not replay_tag or not team_tags:
            continue

        url = (
            f"/data/replay?tag={replay_tag}"
            f"&team_tags={team_tags}"
            f"&opponent_tags={opponent_tags}"
            f"&team_crowns={team_crowns}"
            f"&opponent_crowns={opponent_crowns}"
        )

        links.append({
            "url": url,
            "tag": replay_tag,
            "team_tags": team_tags or "",
            "opponent_tags": opponent_tags or "",
            "team_crowns": int(team_crowns) if team_crowns else 0,
            "opponent_crowns": int(opponent_crowns) if opponent_crowns else 0,
        })

    return links


def _parse_replay_url(url: str) -> dict | None:
    """Parse a RoyaleAPI replay URL into its components."""
    try:
        from urllib.parse import parse_qs, urlparse
        parsed = urlparse(url)
        params = parse_qs(parsed.query)

        return {
            "url": url,
            "tag": params.get("tag", [""])[0],
            "team_tags": params.get("team_tags", [""])[0],
            "opponent_tags": params.get("opponent_tags", [""])[0],
            "team_crowns": int(params.get("team_crowns", [0])[0]),
            "opponent_crowns": int(params.get("opponent_crowns", [0])[0]),
        }
    except (ValueError, IndexError):
        return None


def _match_replay_link(battle: Battle, links: list[dict]) -> str | None:
    """Match a stored battle to a replay link by player/opponent tags and crowns.

    Returns the replay URL path if matched, None otherwise.
    """
    player_tag = (battle.player_tag or "").lstrip("#")
    opponent_tag = (battle.opponent_tag or "").lstrip("#")
    player_crowns = battle.player_crowns
    opponent_crowns = battle.opponent_crowns

    for link in links:
        link_team = link.get("team_tags", "").lstrip("#")
        link_opp = link.get("opponent_tags", "").lstrip("#")
        link_tc = link.get("team_crowns")
        link_oc = link.get("opponent_crowns")

        if (
            link_team == player_tag
            and link_opp == opponent_tag
            and link_tc == player_crowns
            and link_oc == opponent_crowns
        ):
            return link["url"]

    return None


def run_fetch_replays(
    session: Session,
    player_tag: str,
    browser_ws: str | None = None,
    state_path: str | None = None,
    limit: int = 5,
) -> int:
    """Synchronous wrapper for fetch_replays."""
    return asyncio.run(
        fetch_replays(session, player_tag, browser_ws, state_path, limit)
    )


def run_start_login(browser_ws: str | None = None) -> None:
    """Synchronous wrapper for start_login."""
    asyncio.run(start_login(browser_ws))


def run_check_login(
    browser_ws: str | None = None,
    state_path: str | None = None,
) -> bool:
    """Synchronous wrapper for check_login_and_save."""
    return asyncio.run(check_login_and_save(browser_ws, state_path))
