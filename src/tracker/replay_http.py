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

import contextlib
import fcntl
import json
import logging
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from http.cookies import SimpleCookie
from pathlib import Path
from typing import Callable, Optional
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
# 2026-06-04: dropped 8 -> 4 after sustained 100% 429 rate-limiting (started
# 06-02). Isolated requests still return 200, so it was purely our burst rate
# crossing Cloudflare's threshold; 8 concurrent + the every-1-min corpus cron +
# 8-retry amplification kept us flagged. 4 backs us under the limit.
MAX_CONCURRENT = 4           # concurrent HTTP requests — Cloudflare throttles above this
REPLAY_DELAY = 0.5           # seconds between replay fetches (rate limiting)
BATCH_PAGE_DELAY = 0.2       # seconds between battle page fetches
STARTUP_JITTER = 1.0         # max seconds of random jitter per thread at startup
REQUEST_TIMEOUT = 15         # seconds per HTTP request
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)

# Global request-rate cap across ALL threads. RoyaleAPI's Cloudflare issues a
# bot-challenge (cf-mitigated: challenge, no Retry-After) once sustained req/min
# crosses a threshold. Pre-fiasco the slow hardware was an accidental governor —
# requests completed slowly enough to stay under the limit. The faster 64GB/NVMe
# box removed that governor, so 8-concurrent now bursts past the threshold. This
# caps the average request rate explicitly, independent of concurrency or
# hardware speed. Tune via env without a rebuild.
REQUESTS_PER_SEC = float(os.environ.get("ROYALEAPI_REQUESTS_PER_SEC", "2.0"))


class _RateLimiter:
    """Thread-safe global rate limiter.

    Reserves a time slot under a short lock, then sleeps outside the lock until
    the slot — so N threads get staggered start times averaging REQUESTS_PER_SEC
    without serializing on the lock during the wait.
    """

    def __init__(self, rate_per_sec: float):
        self._min_interval = 1.0 / rate_per_sec if rate_per_sec > 0 else 0.0
        self._lock = threading.Lock()
        self._next = 0.0

    def acquire(self) -> None:
        if self._min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            slot = now if now >= self._next else self._next
            self._next = slot + self._min_interval
        wait = slot - time.monotonic()
        if wait > 0:
            time.sleep(wait)


_RATE_LIMITER = _RateLimiter(REQUESTS_PER_SEC)


# Cross-process serialization for RoyaleAPI scraping. The per-process rate
# limiter only governs one process; the cron runs several replay-fetching
# processes (personal + corpus) that would otherwise stack their rates and
# re-trip the Cloudflare challenge. A single flock funnels all of them so at
# most one process scrapes at a time — restoring the implicit serialization the
# old shared cr-browser used to provide. flock auto-releases on process death,
# so a crashed cron job can't wedge the pipeline.
ROYALEAPI_LOCK_PATH = os.environ.get(
    "ROYALEAPI_SCRAPE_LOCK", "/app/data/.royaleapi_scrape.lock"
)


@contextlib.contextmanager
def _royaleapi_serialize():
    """Hold an exclusive flock for the duration of one RoyaleAPI scrape pass."""
    fd = os.open(ROYALEAPI_LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o644)
    t0 = time.monotonic()
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        waited = time.monotonic() - t0
        if waited > 1.0:
            logger.info("Waited %.1fs for RoyaleAPI scrape lock", waited)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


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


def _http_get(url: str, cookie_header: str) -> tuple[int, str, list[str]]:
    """Make an HTTP GET request using urllib.

    Args:
        url: Full URL.
        cookie_header: Pre-built Cookie header string.

    Returns:
        (status_code, response_body, set_cookie_headers)
    """
    _RATE_LIMITER.acquire()  # global rate cap — gate every RoyaleAPI request
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
        set_cookies = resp.headers.get_all("Set-Cookie") or []
        # Check for login redirect
        if "/login" in resp.url:
            raise SessionExpiredError("Redirected to login")
        return resp.status, body, set_cookies
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if e.fp else ""
        set_cookies = e.headers.get_all("Set-Cookie") if e.headers else []
        return e.code, body, set_cookies or []


# RoyaleAPI cookies worth persisting back to the session file. The login
# session is a sliding-expiry cookie: every authenticated response carries a
# Set-Cookie that bumps its 7-day expiry forward. Capturing and re-saving it
# keeps the login alive indefinitely (as the old Playwright storage_state
# save-back used to) instead of letting it age out 7 days after login.
RENEWABLE_COOKIE_NAMES = ("__royaleapi_session_v2", "cf_clearance", "NB_SRVID")


def _parse_renewed_cookies(set_cookie_headers: list[str]) -> dict[str, dict]:
    """Parse Set-Cookie headers into {name: {value, expires}} for tracked cookies.

    ``expires`` is a unix timestamp (float) or None when the header gives no
    explicit lifetime.
    """
    renewed: dict[str, dict] = {}
    for raw in set_cookie_headers:
        try:
            jar = SimpleCookie()
            jar.load(raw)
        except Exception:
            continue
        for name, morsel in jar.items():
            if name not in RENEWABLE_COOKIE_NAMES or not morsel.value:
                continue
            expires = None
            max_age = morsel["max-age"]
            if max_age:
                try:
                    expires = time.time() + float(max_age)
                except ValueError:
                    expires = None
            elif morsel["expires"]:
                try:
                    # e.g. "Wed, 04 Jun 2026 07:24:28 GMT"
                    from email.utils import parsedate_to_datetime
                    expires = parsedate_to_datetime(morsel["expires"]).timestamp()
                except Exception:
                    expires = None
            renewed[name] = {"value": morsel.value, "expires": expires}
    return renewed


def _persist_session_cookies(state_path: str, renewed: dict[str, dict]) -> None:
    """Atomically merge renewed cookie values/expiries into the session file.

    Updates the Playwright storage_state JSON in place (value + expires) for any
    tracked cookie that was renewed, then rewrites via temp-file + os.replace so
    a concurrent reader never sees a half-written file (the SQLite corruption
    lesson applies to this file too).
    """
    if not renewed:
        return
    try:
        with open(state_path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.warning("Cannot read session file to persist cookies: %s", e)
        return

    cookies = data.get("cookies", [])
    by_name = {c.get("name"): c for c in cookies}
    changed = False
    for name, info in renewed.items():
        existing = by_name.get(name)
        if existing is None:
            continue  # only renew cookies already established by login
        if existing.get("value") != info["value"]:
            existing["value"] = info["value"]
            changed = True
        if info["expires"] and existing.get("expires") != info["expires"]:
            existing["expires"] = info["expires"]
            changed = True

    if not changed:
        return

    tmp = f"{state_path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, state_path)
        logger.info("Persisted renewed session cookies: %s", ", ".join(sorted(renewed)))
    except OSError as e:
        logger.warning("Failed to persist renewed cookies: %s", e)
        try:
            os.unlink(tmp)
        except OSError:
            pass


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
    on_cookies: Optional[Callable[[list[str]], None]] = None,
) -> tuple[int, str]:
    """Fetch a single replay with exponential backoff on 429s.

    Returns:
        (status_code, replay_html)
    """
    total_backoff = 0.0
    for attempt in range(max_retries):
        status, body, set_cookies = _http_get(url, cookie_header)
        if on_cookies and set_cookies:
            on_cookies(set_cookies)

        if status in (429, 403):
            is_cloudflare = "just a moment" in body.lower() if body else False
            # Start backoff at 8s — empirically 69% of retries succeed after
            # ~8s. Skips 3 wasted requests that just add traffic without
            # results. Subsequent retries double from there.
            wait = min(8 * max(1, 2 ** (attempt - 1)), 64) + random.uniform(0, 1)
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
    """Fetch replays for a player, serialized across processes.

    Thin wrapper holding the cross-process RoyaleAPI lock so concurrent cron
    jobs (personal + corpus) take turns rather than stacking their request
    rates. Per-player granularity keeps them interleaving fairly.
    """
    with _royaleapi_serialize():
        return _fetch_replays_http_impl(
            db_session, player_tag, state_path, limit, max_pages
        )


def _fetch_replays_http_impl(
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

    # Accumulate renewed session cookies seen on authenticated responses, then
    # persist once at the end so the sliding 7-day login window keeps advancing.
    # Thread-safe: replay fetches run in a pool, so the page loop (main thread)
    # and worker threads both feed _capture_cookies.
    _renewed: dict[str, dict] = {}
    _renewed_lock = threading.Lock()

    def _capture_cookies(set_cookie_headers: list[str]) -> None:
        parsed = _parse_renewed_cookies(set_cookie_headers)
        if parsed:
            with _renewed_lock:
                _renewed.update(parsed)

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
            status, html, set_cookies = _http_get(url, cookie_header)
            if set_cookies:
                _capture_cookies(set_cookies)
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
            status, html = _fetch_replay_with_retry(
                url, cookie_header, battle.battle_id, on_cookies=_capture_cookies,
            )
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

    # Persist any renewed session cookies so the login window slides forward.
    with _renewed_lock:
        _persist_session_cookies(state_path, dict(_renewed))

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
