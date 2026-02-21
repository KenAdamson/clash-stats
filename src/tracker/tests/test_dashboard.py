"""Tests for the Flask dashboard endpoints."""

import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from tracker import analytics
from tracker.dashboard import create_app
from tracker.models import Base
from tracker.tests.conftest import (
    make_battle,
    make_player_profile,
    seed_reporting_db,
)


@pytest.fixture
def app(tmp_path):
    """Create a Flask test app with a seeded database."""
    db_file = tmp_path / "test.db"
    db_path = str(db_file)

    # Create tables directly (skip Alembic for test speed)
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    Base.metadata.create_all(engine)

    # Seed data
    with Session(engine) as session:
        seed_reporting_db(session)
    engine.dispose()

    app = create_app(db_path)
    app.config["TESTING"] = True
    return app


@pytest.fixture
def client(app):
    """Flask test client."""
    return app.test_client()


@pytest.fixture
def empty_app(tmp_path):
    """Flask test app with an empty database."""
    db_file = tmp_path / "empty.db"
    db_path = str(db_file)
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    Base.metadata.create_all(engine)
    engine.dispose()

    app = create_app(db_path)
    app.config["TESTING"] = True
    return app


@pytest.fixture
def empty_client(empty_app):
    """Flask test client for empty database."""
    return empty_app.test_client()


class TestIndex:
    """Tests for the dashboard page."""

    def test_index_returns_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert b"Clash Royale Analytics" in resp.data

    def test_index_includes_chart_js(self, client):
        resp = client.get("/")
        assert b"chart.js" in resp.data

    def test_index_includes_dashboard_js(self, client):
        resp = client.get("/")
        assert b"dashboard.js" in resp.data


class TestOverviewAPI:
    """Tests for /api/overview."""

    def test_overview_returns_json(self, client):
        resp = client.get("/api/overview")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "tracked" in data
        assert "api_stats" in data
        assert "snapshot_diff" in data

    def test_overview_tracked_stats(self, client):
        data = client.get("/api/overview").get_json()
        tracked = data["tracked"]
        assert tracked["total"] > 0
        assert tracked["wins"] + tracked["losses"] + tracked["draws"] == tracked["total"]

    def test_overview_api_stats(self, client):
        data = client.get("/api/overview").get_json()
        api = data["api_stats"]
        assert api["name"] == "KrylarPrime"
        assert api["trophies"] == 10900

    def test_overview_empty_db(self, empty_client):
        data = empty_client.get("/api/overview").get_json()
        assert data["tracked"]["total"] == 0
        assert data["api_stats"] == {}
        assert data["snapshot_diff"] is None


class TestTrophyHistoryAPI:
    """Tests for /api/trophy-history."""

    def test_trophy_history_returns_list(self, client):
        resp = client.get("/api/trophy-history")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, list)
        assert len(data) > 0

    def test_trophy_history_has_required_fields(self, client):
        data = client.get("/api/trophy-history").get_json()
        entry = data[0]
        assert "battle_time" in entry
        assert "trophies" in entry
        assert "result" in entry

    def test_trophy_history_empty_db(self, empty_client):
        data = empty_client.get("/api/trophy-history").get_json()
        assert data == []


class TestMatchupsAPI:
    """Tests for /api/matchups."""

    def test_matchups_returns_json(self, client):
        resp = client.get("/api/matchups")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "card_matchups" in data
        assert "archetypes" in data

    def test_card_matchups_structure(self, client):
        data = client.get("/api/matchups").get_json()
        matchups = data["card_matchups"]
        assert isinstance(matchups, list)
        if matchups:
            m = matchups[0]
            assert "card_name" in m
            assert "times_faced" in m
            assert "win_rate" in m

    def test_archetypes_structure(self, client):
        data = client.get("/api/matchups").get_json()
        archetypes = data["archetypes"]
        assert isinstance(archetypes, list)
        if archetypes:
            a = archetypes[0]
            assert "archetype" in a
            assert "total" in a
            assert "win_rate" in a

    def test_matchups_empty_db(self, empty_client):
        data = empty_client.get("/api/matchups").get_json()
        assert data["card_matchups"] == []
        assert data["archetypes"] == []


class TestRecentAPI:
    """Tests for /api/recent."""

    def test_recent_returns_list(self, client):
        resp = client.get("/api/recent")
        assert resp.status_code == 200
        data = resp.get_json()
        assert isinstance(data, list)

    def test_recent_has_battle_fields(self, client):
        data = client.get("/api/recent").get_json()
        assert len(data) > 0
        b = data[0]
        assert "result" in b
        assert "player_crowns" in b
        assert "opponent_crowns" in b
        assert "opponent_name" in b
        assert "player_trophy_change" in b

    def test_recent_limit(self, client):
        data = client.get("/api/recent").get_json()
        assert len(data) <= 25

    def test_recent_empty_db(self, empty_client):
        data = empty_client.get("/api/recent").get_json()
        assert data == []


class TestStreaksAPI:
    """Tests for /api/streaks."""

    def test_streaks_returns_json(self, client):
        resp = client.get("/api/streaks")
        assert resp.status_code == 200
        data = resp.get_json()
        assert "streaks" in data
        assert "rolling_35" in data
        assert "rolling_10" in data
        assert "crown_distribution" in data
        assert "time_of_day" in data

    def test_streaks_data(self, client):
        data = client.get("/api/streaks").get_json()
        streaks = data["streaks"]
        assert "current_streak" in streaks
        assert "longest_win_streak" in streaks

    def test_rolling_stats(self, client):
        data = client.get("/api/streaks").get_json()
        r35 = data["rolling_35"]
        assert "total" in r35
        assert "win_rate" in r35
        assert "wins" in r35
        assert r35["total"] > 0

    def test_crown_distribution(self, client):
        data = client.get("/api/streaks").get_json()
        dist = data["crown_distribution"]
        assert "win" in dist
        assert "loss" in dist

    def test_time_of_day(self, client):
        data = client.get("/api/streaks").get_json()
        tod = data["time_of_day"]
        assert isinstance(tod, list)

    def test_streaks_empty_db(self, empty_client):
        data = empty_client.get("/api/streaks").get_json()
        assert data["streaks"]["current_streak"] is None
        assert data["rolling_35"]["total"] == 0


class TestStaticFiles:
    """Tests for static file serving."""

    def test_css_served(self, client):
        resp = client.get("/static/style.css")
        assert resp.status_code == 200
        assert b"--bg" in resp.data

    def test_js_served(self, client):
        resp = client.get("/static/dashboard.js")
        assert resp.status_code == 200
        assert b"fetchAll" in resp.data
