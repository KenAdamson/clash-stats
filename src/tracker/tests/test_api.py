"""Tests for the CR API client."""

import urllib.error
from unittest.mock import patch, MagicMock

import pytest

from tracker.api import ClashRoyaleAPI


class TestClashRoyaleAPI:
    @pytest.fixture
    def mock_urlopen(self):
        with patch("urllib.request.urlopen") as m:
            response = MagicMock()
            response.__enter__ = lambda s: s
            response.__exit__ = MagicMock(return_value=False)
            m.return_value = response
            yield m, response

    def test_player_tag_encoding(self, mock_urlopen):
        mock_urlopen_fn, response = mock_urlopen
        response.read.return_value = b'{"tag": "#L90009GPP"}'
        ClashRoyaleAPI("fake-key").get_player("L90009GPP")
        called_req = mock_urlopen_fn.call_args[0][0]
        assert "%23L90009GPP" in called_req.full_url

    def test_auth_header_set(self, mock_urlopen):
        mock_urlopen_fn, response = mock_urlopen
        response.read.return_value = b"[]"
        ClashRoyaleAPI("my-secret-key").get_battle_log("L90009GPP")
        called_req = mock_urlopen_fn.call_args[0][0]
        assert called_req.get_header("Authorization") == "Bearer my-secret-key"

    def test_http_error_raises(self):
        api = ClashRoyaleAPI("fake-key")
        with patch("urllib.request.urlopen") as m:
            m.side_effect = urllib.error.HTTPError(
                "https://example.com", 403, "Forbidden", {}, None
            )
            with pytest.raises(Exception, match="403"):
                api.get_player("L90009GPP")

    def test_url_error_raises(self):
        api = ClashRoyaleAPI("fake-key")
        with patch("urllib.request.urlopen") as m:
            m.side_effect = urllib.error.URLError("DNS lookup failed")
            with pytest.raises(Exception, match="Connection Error"):
                api.get_player("L90009GPP")
