"""HTTP-based replay fetcher — no browser required.

Replaces Playwright-based replay scraping with direct HTTP requests using
saved RoyaleAPI session cookies. 10-50x faster than browser XHR since we
skip page navigation, Cloudflare challenge waits, and DOM rendering.

The browser sidecar is still needed for the initial Cloudflare challenge
solve (manual login via noVNC), but all subsequent replay fetching uses
plain HTTP with the saved cf_clearance cookie.

Uses urllib.request (not aiohttp) because Cloudflare fingerprints the TLS
stack — aiohttp's SSL context triggers 403s while Python's built-in ssl
module passes cleanly.

Flow:
  1. GET /player/{tag}/battles — extract replay tags from HTML
  2. GET /player/{tag}/battles/scroll/{cursor}/type/all — paginate
  3. GET /data/replay?tag=...&team_tags=... — fetch replay JSON
  4. parse_replay_html() + store_replay_data() — existing pipeline
"""

import json
import logging
import random
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

from sqlalchemy.orm import Session

from tracker.metrics import (
    RATE_LIMIT_BACKOFF,
    REPLAYS_FAILED,
    REPLAYS_FETCHED,
)
from tracker.models import Battle
from tracker.replays import (
    ROYALEAPI_BASE,
    DEFAULT_SESSION_PATH,
    SessionExpiredError,
    parse_replay_html,
    store_replay_data,
)

logger = logging.getLogger(__name__)

# Tuning knobs
MAX_CONCURRENT = 8           # concurrent HTTP requests — Cloudflare limit is between 8-10
REPLAY_DELAY = 0.25          # seconds between replay fetches (rate limiting)
BATCH_PAGE_DELAY = 0.2       # seconds between battle page fetches
STARTUP_JITTER = 1.0         # max seconds of random jitter per thread at startup
REQUEST_TIMEOUT = 15         # seconds per HTTP request
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

# Regex patterns for extracting replay data from battle page HTML
_REPLAY_BUTTON_RE_ALT = re.compile(
    r'data-replay="(?P<tag>[^"]+)"',
)
_TEAM_TAGS_RE = re.compile(r'data-team-tags="(?P<team>[^"]*)"')
_OPP_TAGS_RE = re.compile(r'data-opponent-tags="(?P<opp>[^"]*)"')
_TEAM_CROWNS_RE = re.compile(r'data-team-crowns="(?P<tc>[^"]*)"')
_OPP_CROWNS_RE = re.compile(r'data-opponent-crowns="(?P<oc>[^"]*)"')

# Pagination: extract data-index values for scroll API cursor
_DATA_INDEX_RE = re.compile(r'data-index="(\d+)"')


def _load_cookies(state_path: str) -> dict[str, str]:
    """Load RoyaleAPI cookies from Playwright session state file.

    Args:
        state_path: Path to royaleapi_session.json.

    Returns:
        Dict of cookie name → value for royaleapi.com domain.
    """
    with open(state_path) as f:
        data = json.load(f)
    cookies = {}
    for c in data.get("cookies", []):
        domain = c.get("domain", "")
        if "royaleapi" in domain:
            cookies[c["name"]] = c["value"]
    return cookies


def _build_cookie_header(cookies: dict[str, str]) -> str:
    """Build Cookie header string from dict."""
    return "; ".join(f"{k}={v}" for k, v in cookies.items())


def _http_get(url: str, cookie_header: str) -> tuple[int, str]:
    """Make an HTTP GET request using urllib.

    Args:
        url: Full URL.
        cookie_header: Pre-built Cookie header string.

    Returns:
        (status_code, response_body)
    """
    req = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT,
        "Referer": f"{ROYALEAPI_BASE}/",
        "Accept": "text/html,application/xhtml+xml,application/json,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Cookie": cookie_header,
    })
    try:
        resp = urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT)
        body = resp.read().decode("utf-8", errors="replace")
        # Check for login redirect
        if "/login" in resp.url:
            raise SessionExpiredError("Redirected to login")
        return resp.status, body
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if e.fp else ""
        return e.code, body


def _extract_replay_links_from_html(html: str) -> list[dict]:
    """Extract replay link data from battle page HTML using regex.

    Replaces Playwright DOM scraping of .replay_button elements.

    Args:
        html: Raw HTML of a RoyaleAPI battles page.

    Returns:
        List of dicts with tag, team_tags, opponent_tags, team/opponent_crowns.
    """
    links = []
    for match in _REPLAY_BUTTON_RE_ALT.finditer(html):
        start = match.start()
        elem_start = html.rfind("<", 0, start)
        if elem_start == -1:
            continue
        elem_end = html.find(">", match.end())
        if elem_end == -1:
            continue
        elem = html[elem_start:elem_end + 1]

        tag = match.group("tag")
        team_m = _TEAM_TAGS_RE.search(elem)
        opp_m = _OPP_TAGS_RE.search(elem)
        tc_m = _TEAM_CROWNS_RE.search(elem)
        oc_m = _OPP_CROWNS_RE.search(elem)

        if not team_m:
            continue

        links.append({
            "tag": tag,
            "team_tags": team_m.group("team") if team_m else "",
            "opponent_tags": opp_m.group("opp") if opp_m else "",
            "team_crowns": int(tc_m.group("tc")) if tc_m and tc_m.group("tc") else 0,
            "opponent_crowns": int(oc_m.group("oc")) if oc_m and oc_m.group("oc") else 0,
        })

    return links


def _extract_last_data_index(html: str) -> Optional[str]:
    """Extract the last data-index value for scroll pagination cursor."""
    matches = _DATA_INDEX_RE.findall(html)
    return matches[-1] if matches else None


def _match_battle_to_link(
    battle: Battle,
    links: list[dict],
) -> Optional[dict]:
    """Match a stored battle to a replay link by tags and crowns."""
    player_tag = (battle.player_tag or "").lstrip("#")
    opponent_tag = (battle.opponent_tag or "").lstrip("#")

    for link in links:
        if (
            link["team_tags"].lstrip("#") == player_tag
            and link["opponent_tags"].lstrip("#") == opponent_tag
            and link["team_crowns"] == battle.player_crowns
            and link["opponent_crowns"] == battle.opponent_crowns
        ):
            return link
    return None


def _build_replay_url(link: dict) -> str:
    """Build the /data/replay URL from a link dict."""
    params = urlencode({
        "tag": link["tag"],
        "team_tags": link["team_tags"],
        "opponent_tags": link["opponent_tags"],
        "team_crowns": link["team_crowns"],
        "opponent_crowns": link["opponent_crowns"],
    })
    return f"{ROYALEAPI_BASE}/data/replay?{params}"


def _fetch_replay_with_retry(
    url: str,
    cookie_header: str,
    battle_id: str,
    max_retries: int = 8,
) -> tuple[int, str]:
    """Fetch a single replay with exponential backoff on 429s.

    Returns:
        (status_code, replay_html)
    """
    total_backoff = 0.0
    for attempt in range(max_retries):
        status, body = _http_get(url, cookie_header)

        if status in (429, 403):
            is_cloudflare = "just a moment" in body.lower() if body else False
            wait = min(2 ** attempt, 32) + random.uniform(0, 1)
            total_backoff += wait
            logger.warning(
                "%d on %s — attempt %d/%d, backoff %.1fs%s",
                status, battle_id[:12], attempt + 1, max_retries, wait,
                " (Cloudflare)" if is_cloudflare else "",
            )
            time.sleep(wait)
            continue

        if total_backoff > 0:
            RATE_LIMIT_BACKOFF.observe(total_backoff)

        if status == 200 and body:
            try:
                data = json.loads(body)
                if isinstance(data, dict) and data.get("success") and "html" in data:
                    return status, data["html"]
                return status, ""
            except json.JSONDecodeError:
                if "just a moment" in body.lower():
                    return 403, ""
                return status, body

        return status, ""

    RATE_LIMIT_BACKOFF.observe(total_backoff)
    REPLAYS_FAILED.labels(error_type="rate_limited").inc()
    return status, ""


def fetch_replays_http(
    db_session: Session,
    player_tag: str,
    state_path: str = DEFAULT_SESSION_PATH,
    limit: int = 25,
    max_pages: int = 20,
) -> int:
    """Fetch replays for a player using plain HTTP — no browser needed.

    Args:
        db_session: SQLAlchemy session.
        player_tag: Player tag without #.
        state_path: Path to royaleapi_session.json.
        limit: Max unfetched battles to process.
        max_pages: Max battle pages to paginate.

    Returns:
        Number of replays fetched, or -1 if session expired.
    """
    tag_clean = player_tag.lstrip("#")

    unfetched = (
        db_session.query(Battle)
        .filter(
            Battle.replay_fetched == 0,
            Battle.battle_type.in_(["PvP", "pathOfLegend", "riverRacePvP"]),
            Battle.player_tag == f"#{tag_clean}",
        )
        .order_by(Battle.battle_time.desc())
        .limit(limit)
        .all()
    )

    if not unfetched:
        return 0

    if not Path(state_path).exists():
        logger.warning("No session file at %s — cannot fetch replays", state_path)
        return 0

    cookies = _load_cookies(state_path)
    if "cf_clearance" not in cookies:
        logger.warning("No cf_clearance cookie — browser login required")
        return 0

    cookie_header = _build_cookie_header(cookies)
    stats = {"fetched": 0, "no_link": 0, "failed": 0, "empty": 0}

    # Build unmatched set for early pagination exit
    unmatched = set()
    for b in unfetched:
        key = (
            (b.player_tag or "").lstrip("#"),
            (b.opponent_tag or "").lstrip("#"),
            b.player_crowns,
            b.opponent_crowns,
        )
        unmatched.add(key)

    # Phase 1: Collect replay links from battle pages
    # Page 1: GET /player/{tag}/battles
    # Page 2+: GET /player/{tag}/battles/scroll/{data-index}/type/all
    all_links: list[dict] = []
    scroll_cursor = None

    for page_num in range(max_pages):
        if scroll_cursor:
            url = f"{ROYALEAPI_BASE}/player/{tag_clean}/battles/scroll/{scroll_cursor}/type/all"
        else:
            url = f"{ROYALEAPI_BASE}/player/{tag_clean}/battles"

        try:
            status, html = _http_get(url, cookie_header)
        except SessionExpiredError:
            logger.error("Session expired fetching battles for %s", tag_clean)
            return -1
        except Exception as e:
            logger.warning("Failed fetching battle page %d for %s: %s", page_num + 1, tag_clean, e)
            break

        if status != 200:
            logger.warning("Battle page %d returned %d for %s", page_num + 1, status, tag_clean)
            if status == 403 and "just a moment" in html.lower():
                logger.error("Cloudflare challenge — cf_clearance expired")
                return -1
            break

        links = _extract_replay_links_from_html(html)
        all_links.extend(links)

        for link in links:
            key = (
                link["team_tags"].lstrip("#"),
                link["opponent_tags"].lstrip("#"),
                link["team_crowns"],
                link["opponent_crowns"],
            )
            unmatched.discard(key)

        logger.info(
            "Page %d: %d replay links for %s (total: %d, %d unmatched)",
            page_num + 1, len(links), tag_clean, len(all_links), len(unmatched),
        )

        if not unmatched:
            logger.info("All unfetched battles matched — stopping pagination for %s", tag_clean)
            break

        next_cursor = _extract_last_data_index(html)
        if not next_cursor or next_cursor == scroll_cursor:
            break
        scroll_cursor = next_cursor

        time.sleep(BATCH_PAGE_DELAY)

    if not all_links:
        logger.info("No replay links found for %s", tag_clean)
        return 0

    # Phase 2: Fetch individual replays concurrently
    # Results are (battle_id, parsed_data | "empty" | None)
    parsed_results: list[tuple[str, Optional[dict]]] = []

    _thread_started = set()  # track which threads have done their startup jitter

    def _process_one(battle: Battle) -> tuple[str, Optional[dict]]:
        """Fetch and parse one replay. Returns (battle_id, parsed_data)."""
        # Startup jitter: first request per thread gets a random delay
        # to spread threads across different Cloudflare rate limit buckets
        tid = id(battle)  # unique per call
        import threading
        thread_id = threading.current_thread().ident
        if thread_id not in _thread_started:
            _thread_started.add(thread_id)
            time.sleep(random.uniform(0, STARTUP_JITTER))

        link = _match_battle_to_link(battle, all_links)
        if not link:
            stats["no_link"] += 1
            return battle.battle_id, None

        url = _build_replay_url(link)
        time.sleep(REPLAY_DELAY * random.uniform(0.5, 1.5))

        try:
            status, html = _fetch_replay_with_retry(url, cookie_header, battle.battle_id)
        except Exception as e:
            logger.warning("Error fetching replay %s: %s", battle.battle_id[:12], e)
            stats["failed"] += 1
            REPLAYS_FAILED.labels(error_type="http_error").inc()
            return battle.battle_id, None

        if status == 403:
            stats["failed"] += 1
            REPLAYS_FAILED.labels(error_type="http_403").inc()
            return battle.battle_id, None

        if status != 200 or not html:
            stats["failed"] += 1
            REPLAYS_FAILED.labels(error_type="http_error").inc()
            return battle.battle_id, None

        try:
            data = parse_replay_html(html)
        except Exception as e:
            logger.warning("Parse error for %s: %s", battle.battle_id[:12], e)
            stats["failed"] += 1
            return battle.battle_id, None

        if data.get("events"):
            stats["fetched"] += 1
            REPLAYS_FETCHED.labels(source="http").inc()
            return battle.battle_id, data
        else:
            stats["empty"] += 1
            return battle.battle_id, {"empty": True}

    # Run replay fetches in thread pool
    with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as pool:
        futures = {pool.submit(_process_one, b): b for b in unfetched}
        for future in as_completed(futures):
            parsed_results.append(future.result())

    # Store results in main thread (SQLAlchemy sessions aren't thread-safe)
    for bid, data in parsed_results:
        if data is None:
            continue
        if data.get("empty"):
            db_session.execute(
                Battle.__table__.update()
                .where(Battle.battle_id == bid)
                .values(replay_fetched=1)
            )
        else:
            store_replay_data(db_session, bid, data)

    db_session.commit()

    logger.info(
        "HTTP replay fetch for %s: %d fetched, %d empty, %d no_link, %d failed (of %d)",
        tag_clean, stats["fetched"], stats["empty"], stats["no_link"],
        stats["failed"], len(unfetched),
    )
    return stats["fetched"]


def run_fetch_replays_http(
    db_session: Session,
    player_tag: str,
    state_path: str = DEFAULT_SESSION_PATH,
    limit: int = 25,
) -> int:
    """Synchronous entry point for HTTP replay fetching.

    Args:
        db_session: SQLAlchemy session.
        player_tag: Player tag (with or without #).
        state_path: Path to session file.
        limit: Max unfetched battles to process.

    Returns:
        Number of replays fetched, or -1 if session expired.
    """
    return fetch_replays_http(db_session, player_tag, state_path, limit)
