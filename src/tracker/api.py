"""Clash Royale API client."""

import json
import urllib.error
import urllib.parse
import urllib.request

DEFAULT_API_URL = "https://api.clashroyale.com/v1"


class ClashRoyaleAPI:
    """Simple CR API client using stdlib only."""

    def __init__(self, api_key: str, base_url: str = DEFAULT_API_URL):
        self.api_key = api_key
        self.base_url = base_url

    def _request(self, endpoint: str) -> dict:
        """Make authenticated API request.

        Args:
            endpoint: API path (e.g., /players/%23TAG).

        Returns:
            Parsed JSON response.

        Raises:
            Exception: On HTTP or connection errors.
        """
        url = f"{self.base_url}{endpoint}"

        req = urllib.request.Request(url)
        req.add_header("Authorization", f"Bearer {self.api_key}")
        req.add_header("Accept", "application/json")

        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else ""
            raise Exception(f"API Error {e.code}: {e.reason}\n{error_body}")
        except urllib.error.URLError as e:
            raise Exception(f"Connection Error: {e.reason}")

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
