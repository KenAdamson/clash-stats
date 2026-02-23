"""Clash Royale API client."""

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request

from tracker.metrics import API_LATENCY, API_REQUESTS
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

logger = logging.getLogger(__name__)

DEFAULT_API_URL = "https://api.clashroyale.com/v1"


# ---------------------------------------------------------------------------
# Structured exception hierarchy
# ---------------------------------------------------------------------------

class APIError(Exception):
    """Base for all CR API errors."""

    def __init__(self, message: str, status_code: int | None = None, body: str = ""):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class RateLimitError(APIError):
    """429 Too Many Requests."""


class AuthError(APIError):
    """401 Unauthorized or 403 Forbidden."""


class NotFoundError(APIError):
    """404 Not Found (player tag doesn't exist)."""


class ServerError(APIError):
    """5xx server-side errors."""


class ConnectionError_(APIError):
    """Network/timeout errors."""


def _classify_http_error(code: int, reason: str, body: str) -> APIError:
    """Map HTTP status code to the appropriate exception class."""
    msg = f"API Error {code}: {reason}"
    if body:
        msg += f"\n{body}"

    if code == 429:
        return RateLimitError(msg, status_code=code, body=body)
    elif code in (401, 403):
        return AuthError(msg, status_code=code, body=body)
    elif code == 404:
        return NotFoundError(msg, status_code=code, body=body)
    elif code >= 500:
        return ServerError(msg, status_code=code, body=body)
    else:
        return APIError(msg, status_code=code, body=body)


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------

class ClashRoyaleAPI:
    """CR API client with retry and structured error handling."""

    def __init__(self, api_key: str, base_url: str = DEFAULT_API_URL):
        self.api_key = api_key
        self.base_url = base_url

    @retry(
        retry=retry_if_exception_type((RateLimitError, ServerError, ConnectionError_)),
        wait=wait_exponential_jitter(initial=2, max=60, jitter=3),
        stop=stop_after_attempt(4),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    def _request(self, endpoint: str) -> dict:
        """Make authenticated API request with retry on transient errors.

        Args:
            endpoint: API path (e.g., /players/%23TAG).

        Returns:
            Parsed JSON response.

        Raises:
            RateLimitError: On 429 (retried first).
            AuthError: On 401/403 (not retried).
            NotFoundError: On 404 (not retried).
            ServerError: On 5xx (retried first).
            ConnectionError_: On network/timeout (retried first).
        """
        url = f"{self.base_url}{endpoint}"

        req = urllib.request.Request(url)
        req.add_header("Authorization", f"Bearer {self.api_key}")
        req.add_header("Accept", "application/json")

        endpoint_label = endpoint.split("?")[0].strip("/").split("/")[0]
        start = time.monotonic()
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                data = json.loads(response.read().decode("utf-8"))
                elapsed = time.monotonic() - start
                API_REQUESTS.labels(endpoint=endpoint_label, status="200").inc()
                API_LATENCY.labels(endpoint=endpoint_label).observe(elapsed)
                logger.debug("API %s → 200 (%.1fs)", endpoint_label, elapsed)
                return data
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else ""
            API_REQUESTS.labels(
                endpoint=endpoint_label, status=str(e.code)
            ).inc()
            raise _classify_http_error(e.code, e.reason, error_body)
        except urllib.error.URLError as e:
            API_REQUESTS.labels(
                endpoint=endpoint_label, status="conn_error"
            ).inc()
            raise ConnectionError_(
                f"Connection Error: {e.reason}", status_code=None
            )

    def get_player(self, player_tag: str) -> dict:
        """Get player profile.

        Args:
            player_tag: Player tag with or without '#' prefix.

        Returns:
            Player profile dict from the API.
        """
        encoded_tag = urllib.parse.quote(
            f"#{player_tag}" if not player_tag.startswith("#") else player_tag
        )
        return self._request(f"/players/{encoded_tag}")

    def get_battle_log(self, player_tag: str) -> list:
        """Get last 25 battles.

        Args:
            player_tag: Player tag with or without '#' prefix.

        Returns:
            List of battle dicts from the API.
        """
        encoded_tag = urllib.parse.quote(
            f"#{player_tag}" if not player_tag.startswith("#") else player_tag
        )
        return self._request(f"/players/{encoded_tag}/battlelog")

    def get_top_players(self, location_id: str = "global", limit: int = 200) -> list:
        """Get top-ranked Path of Legend players.

        Args:
            location_id: Location ID or 'global' for global leaderboard.
            limit: Number of players to return (max 200).

        Returns:
            List of player ranking dicts from the API.
        """
        resp = self._request(
            f"/locations/{location_id}/pathoflegend/players?limit={limit}"
        )
        return resp.get("items", []) if isinstance(resp, dict) else resp
