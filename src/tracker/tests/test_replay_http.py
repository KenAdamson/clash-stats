"""Tests for replay_http session-cookie persistence.

RoyaleAPI's login cookie (__royaleapi_session_v2) is a sliding-expiry session:
every authenticated response carries a Set-Cookie that bumps its ~7-day expiry
forward. The old Playwright scraper persisted those renewals via storage_state;
the pure-HTTP scraper must do the same explicitly or the login ages out 7 days
after login. These tests lock in that behavior so a future refactor can't
silently drop it (the regression that took the replay pipeline down in 06/2026).
"""

import fcntl
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

import tracker.replay_http as rh
from tracker.replay_http import (
    _parse_renewed_cookies,
    _persist_session_cookies,
    _RateLimiter,
    RENEWABLE_COOKIE_NAMES,
)


# ---------------------------------------------------------------------------
# _royaleapi_serialize (cross-process scrape lock)
# ---------------------------------------------------------------------------

def test_royaleapi_lock_is_exclusive(tmp_path, monkeypatch):
    """While the lock is held, an independent fd cannot acquire it; once
    released, it can. Proves the cross-process mutex actually excludes."""
    lockpath = str(tmp_path / "scrape.lock")
    monkeypatch.setattr(rh, "ROYALEAPI_LOCK_PATH", lockpath)

    with rh._royaleapi_serialize():
        fd = os.open(lockpath, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(fd)

    # Released — now acquirable.
    fd = os.open(lockpath, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)  # must not raise
        fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# _RateLimiter
# ---------------------------------------------------------------------------

def test_rate_limiter_spaces_sequential_calls():
    """N acquires at R/s take ~(N-1)/R seconds (first is immediate)."""
    rl = _RateLimiter(20)  # 20/s -> 0.05s interval
    start = time.monotonic()
    for _ in range(5):
        rl.acquire()
    elapsed = time.monotonic() - start
    # 4 gaps * 0.05s = 0.2s expected; allow scheduling slack.
    assert 0.18 < elapsed < 0.45


def test_rate_limiter_disabled_is_immediate():
    """rate<=0 disables limiting (no sleeps)."""
    rl = _RateLimiter(0)
    start = time.monotonic()
    for _ in range(100):
        rl.acquire()
    assert time.monotonic() - start < 0.05


def test_rate_limiter_caps_across_threads():
    """Concurrent threads share the global cap: total time respects the rate."""
    rl = _RateLimiter(20)  # 20/s
    start = time.monotonic()
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(lambda _: rl.acquire(), range(16)))
    elapsed = time.monotonic() - start
    # 16 acquires at 20/s -> ~15*0.05 = 0.75s floor regardless of 8 threads.
    assert elapsed > 0.6


# ---------------------------------------------------------------------------
# _parse_renewed_cookies
# ---------------------------------------------------------------------------

def test_parse_extracts_tracked_cookies_with_maxage():
    """Max-Age yields an expiry ~that many seconds out; value captured."""
    headers = [
        "__royaleapi_session_v2=NEWVAL; Max-Age=604800; path=/; secure; HttpOnly",
        "cf_clearance=freshcf; Max-Age=31536000; path=/",
    ]
    parsed = _parse_renewed_cookies(headers)
    assert set(parsed) == {"__royaleapi_session_v2", "cf_clearance"}
    assert parsed["__royaleapi_session_v2"]["value"] == "NEWVAL"
    days = (parsed["__royaleapi_session_v2"]["expires"] - time.time()) / 86400
    assert 6.9 < days < 7.1


def test_parse_handles_expires_date_form():
    """An Expires HTTP-date is parsed to a unix timestamp."""
    # Far-future fixed date so the test isn't time-sensitive.
    headers = ["__royaleapi_session_v2=V; expires=Wed, 01 Jan 2031 00:00:00 GMT; path=/"]
    parsed = _parse_renewed_cookies(headers)
    assert parsed["__royaleapi_session_v2"]["value"] == "V"
    assert parsed["__royaleapi_session_v2"]["expires"] > time.time()


def test_parse_ignores_untracked_cookies():
    """Cookies outside RENEWABLE_COOKIE_NAMES are not captured."""
    parsed = _parse_renewed_cookies(["random_analytics=abc; path=/"])
    assert parsed == {}
    # Sanity: the tracked set is what we expect.
    assert "__royaleapi_session_v2" in RENEWABLE_COOKIE_NAMES


def test_parse_tolerates_empty_and_garbage():
    assert _parse_renewed_cookies([]) == {}
    # Malformed header must not raise.
    assert _parse_renewed_cookies(["=;=;garbage"]) == {}


def test_parse_skips_empty_value():
    """A cleared cookie (empty value) is not treated as a renewal."""
    parsed = _parse_renewed_cookies(["__royaleapi_session_v2=; Max-Age=0; path=/"])
    assert "__royaleapi_session_v2" not in parsed


# ---------------------------------------------------------------------------
# _persist_session_cookies
# ---------------------------------------------------------------------------

def _write_state(tmp_path, cookies):
    p = tmp_path / "session.json"
    p.write_text(json.dumps({"cookies": cookies}))
    return str(p)


def test_persist_updates_value_and_expires(tmp_path):
    old_exp = time.time() + 100
    path = _write_state(tmp_path, [
        {"name": "__royaleapi_session_v2", "value": "OLD",
         "domain": ".royaleapi.com", "path": "/", "expires": old_exp},
    ])
    new_exp = time.time() + 604800
    _persist_session_cookies(path, {
        "__royaleapi_session_v2": {"value": "NEW", "expires": new_exp},
    })
    saved = {c["name"]: c for c in json.load(open(path))["cookies"]}
    assert saved["__royaleapi_session_v2"]["value"] == "NEW"
    assert saved["__royaleapi_session_v2"]["expires"] == new_exp


def test_persist_only_touches_existing_cookies(tmp_path):
    """A renewed cookie not already in the file is not added (login establishes it)."""
    path = _write_state(tmp_path, [
        {"name": "cf_clearance", "value": "cf", "domain": ".royaleapi.com", "path": "/"},
    ])
    _persist_session_cookies(path, {
        "__royaleapi_session_v2": {"value": "NEW", "expires": time.time() + 100},
        "cf_clearance": {"value": "cf2", "expires": None},
    })
    saved = {c["name"]: c for c in json.load(open(path))["cookies"]}
    assert "__royaleapi_session_v2" not in saved   # not established by login here
    assert saved["cf_clearance"]["value"] == "cf2"  # existing one updated


def test_persist_atomic_no_temp_file_left(tmp_path):
    path = _write_state(tmp_path, [
        {"name": "cf_clearance", "value": "cf", "domain": ".royaleapi.com", "path": "/"},
    ])
    _persist_session_cookies(path, {"cf_clearance": {"value": "cf2", "expires": None}})
    leftovers = [f for f in os.listdir(tmp_path) if f != "session.json"]
    assert leftovers == []
    # File remains valid JSON.
    json.load(open(path))


def test_persist_noop_when_nothing_changed(tmp_path):
    """If the renewed value matches what's stored, the file is left untouched."""
    path = _write_state(tmp_path, [
        {"name": "cf_clearance", "value": "cf", "domain": ".royaleapi.com", "path": "/"},
    ])
    before = os.stat(path).st_mtime_ns
    _persist_session_cookies(path, {"cf_clearance": {"value": "cf", "expires": None}})
    after = os.stat(path).st_mtime_ns
    assert before == after  # no rewrite


def test_persist_empty_renewed_is_safe(tmp_path):
    path = _write_state(tmp_path, [
        {"name": "cf_clearance", "value": "cf", "domain": ".royaleapi.com", "path": "/"},
    ])
    _persist_session_cookies(path, {})  # must not raise or rewrite
    assert json.load(open(path))["cookies"][0]["value"] == "cf"


def test_persist_missing_file_is_safe(tmp_path):
    """A non-existent session file is handled gracefully (logged, no raise)."""
    _persist_session_cookies(str(tmp_path / "nope.json"),
                             {"cf_clearance": {"value": "x", "expires": None}})
