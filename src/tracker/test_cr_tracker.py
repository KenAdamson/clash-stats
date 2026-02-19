"""
Tests for cr_tracker.py

Covers:
- Database schema initialization
- Battle ID generation (dedup hashing)
- Deck hash generation
- Battle storage and deduplication
- Player snapshot storage
- All analytics queries
- Reporting functions (no-crash verification)
- CLI argument parsing
- Edge cases (empty db, missing fields)
"""

import json
import os
import sys
import tempfile
import urllib.error
from io import StringIO
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, os.path.dirname(__file__))
import cr_tracker


# =============================================================================
# TEST FIXTURES — realistic CR API response shapes
# =============================================================================

def make_card(name: str, level: int = 14, max_level: int = 14,
              elixir: int = 4, evolution_level: int = 0) -> dict:
    """Build a card dict matching the CR API shape."""
    card = {
        "name": name,
        "id": hash(name) & 0xFFFFFFFF,
        "level": level,
        "maxLevel": max_level,
        "elixirCost": elixir,
        "iconUrls": {},
    }
    if evolution_level:
        card["evolutionLevel"] = evolution_level
    return card


# Ken's actual deck
PLAYER_DECK = [
    make_card("P.E.K.K.A", level=16, elixir=7),
    make_card("Witch", level=16, elixir=5, evolution_level=1),
    make_card("Executioner", level=15, elixir=5, evolution_level=1),
    make_card("Graveyard", level=15, elixir=5),
    make_card("Miner", level=15, elixir=3),
    make_card("Arrows", level=14, elixir=3),
    make_card("Goblin Curse", level=14, elixir=2),
    make_card("Bats", level=15, elixir=2),
]

# Typical opponent deck
OPPONENT_DECK = [
    make_card("Hog Rider", level=16, elixir=4),
    make_card("Musketeer", level=14, elixir=4),
    make_card("Ice Spirit", level=14, elixir=1),
    make_card("Skeletons", level=14, elixir=1),
    make_card("Cannon", level=14, elixir=3),
    make_card("Fireball", level=14, elixir=4),
    make_card("The Log", level=14, elixir=2),
    make_card("Ice Golem", level=14, elixir=2),
]


def make_battle(battle_time: str = "20260214T180000.000Z",
                player_tag: str = "#L90009GPP",
                player_crowns: int = 3,
                opponent_tag: str = "#OPPONENT1",
                opponent_name: str = "xXDarkLordXx",
                opponent_crowns: int = 1,
                trophy_change: int = 30,
                battle_type: str = "PvP",
                player_deck: list = None,
                opponent_deck: list = None) -> dict:
    """Build a battle dict matching the CR API shape."""
    return {
        "type": battle_type,
        "battleTime": battle_time,
        "isLadderTournament": True,
        "arena": {"id": 54000061, "name": "Arena 22"},
        "gameMode": {"id": 72000006, "name": "Ladder"},
        "team": [{
            "tag": player_tag,
            "name": "KrylarPrime",
            "crowns": player_crowns,
            "startingTrophies": 10900,
            "trophyChange": trophy_change,
            "kingTowerHitPoints": 8000 if player_crowns < 3 else 0,
            "princessTowersHitPoints": [3000] if player_crowns < 2 else [],
            "cards": player_deck or PLAYER_DECK,
            "elixirLeaked": 0.5,
        }],
        "opponent": [{
            "tag": opponent_tag,
            "name": opponent_name,
            "crowns": opponent_crowns,
            "startingTrophies": 10850,
            "trophyChange": -trophy_change,
            "kingTowerHitPoints": 0 if opponent_crowns == 0 else 5000,
            "princessTowersHitPoints": [],
            "cards": opponent_deck or OPPONENT_DECK,
            "elixirLeaked": 1.3,
        }],
    }


def make_player_profile(tag: str = "#L90009GPP", trophies: int = 10900) -> dict:
    """Build a player profile dict matching the CR API shape."""
    return {
        "tag": tag,
        "name": "KrylarPrime",
        "expLevel": 61,
        "trophies": trophies,
        "bestTrophies": 10954,
        "wins": 1500,
        "losses": 1343,
        "battleCount": 2843,
        "threeCrownWins": 1094,
        "challengeCardsWon": 500,
        "challengeMaxWins": 12,
        "tournamentBattleCount": 0,
        "tournamentCardsWon": 0,
        "warDayWins": 120,
        "totalDonations": 50000,
        "clan": {"tag": "#CLAN123", "name": "TestClan"},
        "arena": {"id": 54000061, "name": "Arena 22"},
    }


# =============================================================================
# SHARED FIXTURES
# =============================================================================

@pytest.fixture
def db(tmp_path):
    """Fresh SQLite database for each test."""
    db_path = str(tmp_path / "test.db")
    database = cr_tracker.BattleDatabase(db_path)
    yield database
    database.close()


@pytest.fixture
def db_path(tmp_path):
    """Just a temp DB path (for CLI tests that manage their own connection)."""
    return str(tmp_path / "test.db")


def seed_battles(db, specs: list[dict]):
    """Insert multiple battles from a list of keyword overrides."""
    for i, spec in enumerate(specs):
        spec.setdefault("battle_time", f"2026021{4 + i // 10}T{18 + i % 10:02d}0000.000Z")
        db.store_battle(make_battle(**spec), "#L90009GPP")


# =============================================================================
# SCHEMA
# =============================================================================

class TestSchema:
    def test_creates_tables(self, db):
        cursor = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row["name"] for row in cursor.fetchall()}
        assert "battles" in tables
        assert "player_snapshots" in tables
        assert "deck_cards" in tables

    def test_idempotent(self, db):
        db._init_schema()
        db._init_schema()

    def test_indexes_exist(self, db):
        cursor = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        )
        indexes = {row["name"] for row in cursor.fetchall()}
        assert "idx_battles_player_tag" in indexes
        assert "idx_battles_battle_time" in indexes
        assert "idx_battles_result" in indexes
        assert "idx_deck_cards_card_name" in indexes


# =============================================================================
# BATTLE ID GENERATION
# =============================================================================

class TestBattleId:
    def test_deterministic(self, db):
        battle = make_battle()
        assert db._generate_battle_id(battle) == db._generate_battle_id(battle)

    def test_different_for_different_times(self, db):
        b1 = make_battle(battle_time="20260214T180000.000Z")
        b2 = make_battle(battle_time="20260214T190000.000Z")
        assert db._generate_battle_id(b1) != db._generate_battle_id(b2)

    def test_different_for_different_crowns(self, db):
        b1 = make_battle(player_crowns=3, opponent_crowns=1)
        b2 = make_battle(player_crowns=2, opponent_crowns=1)
        assert db._generate_battle_id(b1) != db._generate_battle_id(b2)

    def test_is_32_hex_chars(self, db):
        bid = db._generate_battle_id(make_battle())
        assert len(bid) == 32
        int(bid, 16)  # raises ValueError if not hex


# =============================================================================
# DECK HASH
# =============================================================================

class TestDeckHash:
    def test_ignores_level(self, db):
        d1 = [make_card("P.E.K.K.A", level=14), make_card("Witch", level=14)]
        d2 = [make_card("P.E.K.K.A", level=16), make_card("Witch", level=16)]
        assert db._generate_deck_hash(d1) == db._generate_deck_hash(d2)

    def test_ignores_order(self, db):
        d1 = [make_card("Witch"), make_card("P.E.K.K.A")]
        d2 = [make_card("P.E.K.K.A"), make_card("Witch")]
        assert db._generate_deck_hash(d1) == db._generate_deck_hash(d2)

    def test_different_for_different_cards(self, db):
        d1 = [make_card("Witch"), make_card("P.E.K.K.A")]
        d2 = [make_card("Witch"), make_card("Hog Rider")]
        assert db._generate_deck_hash(d1) != db._generate_deck_hash(d2)


# =============================================================================
# BATTLE STORAGE
# =============================================================================

class TestBattleStorage:
    def test_returns_new(self, db):
        battle_id, is_new = db.store_battle(make_battle(), "#L90009GPP")
        assert is_new
        assert len(battle_id) == 32

    def test_deduplicates(self, db):
        battle = make_battle()
        _, is_new1 = db.store_battle(battle, "#L90009GPP")
        _, is_new2 = db.store_battle(battle, "#L90009GPP")
        assert is_new1
        assert not is_new2

    def test_total_increments(self, db):
        assert db.get_total_battles() == 0
        db.store_battle(make_battle(battle_time="20260214T180000.000Z"), "#L90009GPP")
        assert db.get_total_battles() == 1
        db.store_battle(make_battle(battle_time="20260214T190000.000Z"), "#L90009GPP")
        assert db.get_total_battles() == 2

    @pytest.mark.parametrize("p_crowns,o_crowns,expected", [
        (3, 1, "win"),
        (1, 3, "loss"),
        (1, 1, "draw"),
    ])
    def test_result_classification(self, db, p_crowns, o_crowns, expected):
        tc = 30 if expected == "win" else (-30 if expected == "loss" else 0)
        db.store_battle(
            make_battle(player_crowns=p_crowns, opponent_crowns=o_crowns, trophy_change=tc),
            "#L90009GPP",
        )
        row = db.conn.execute("SELECT result FROM battles").fetchone()
        assert row["result"] == expected

    def test_crown_differential(self, db):
        db.store_battle(make_battle(player_crowns=3, opponent_crowns=1), "#L90009GPP")
        row = db.conn.execute("SELECT crown_differential FROM battles").fetchone()
        assert row["crown_differential"] == 2

    def test_deck_cards_created(self, db):
        db.store_battle(make_battle(), "#L90009GPP")
        player_cards = db.conn.execute(
            "SELECT COUNT(*) FROM deck_cards WHERE is_player_deck = 1"
        ).fetchone()[0]
        opp_cards = db.conn.execute(
            "SELECT COUNT(*) FROM deck_cards WHERE is_player_deck = 0"
        ).fetchone()[0]
        assert player_cards == 8
        assert opp_cards == 8

    def test_raw_json_preserved(self, db):
        battle = make_battle()
        db.store_battle(battle, "#L90009GPP")
        row = db.conn.execute("SELECT raw_json FROM battles").fetchone()
        restored = json.loads(row["raw_json"])
        assert restored["battleTime"] == battle["battleTime"]

    def test_preserves_trophy_data(self, db):
        db.store_battle(make_battle(trophy_change=30), "#L90009GPP")
        row = db.conn.execute(
            "SELECT player_starting_trophies, player_trophy_change FROM battles"
        ).fetchone()
        assert row["player_starting_trophies"] == 10900
        assert row["player_trophy_change"] == 30


# =============================================================================
# PLAYER SNAPSHOTS
# =============================================================================

class TestPlayerSnapshots:
    def test_store_snapshot(self, db):
        db.store_player_snapshot(make_player_profile())
        row = db.conn.execute("SELECT * FROM player_snapshots").fetchone()
        assert row["player_tag"] == "#L90009GPP"
        assert row["trophies"] == 10900
        assert row["wins"] == 1500
        assert row["three_crown_wins"] == 1094

    def test_store_multiple(self, db):
        db.store_player_snapshot(make_player_profile(trophies=10900))
        db.store_player_snapshot(make_player_profile(trophies=10930))
        count = db.conn.execute("SELECT COUNT(*) FROM player_snapshots").fetchone()[0]
        assert count == 2

    def test_get_latest(self, db):
        db.store_player_snapshot(make_player_profile(trophies=10900))
        db.store_player_snapshot(make_player_profile(trophies=10930))
        stats = db.get_all_time_api_stats()
        assert stats["trophies"] == 10930

    def test_get_latest_empty_db(self, db):
        assert db.get_all_time_api_stats() == {}


# =============================================================================
# ANALYTICS QUERIES
# =============================================================================

class TestAnalytics:
    def test_overall_stats(self, db):
        seed_battles(db, [
            {"player_crowns": 3, "opponent_crowns": 1, "trophy_change": 30},
            {"player_crowns": 1, "opponent_crowns": 3, "trophy_change": -30},
            {"player_crowns": 3, "opponent_crowns": 0, "trophy_change": 31},
        ])
        stats = db.get_overall_stats()
        assert stats["total"] == 3
        assert stats["wins"] == 2
        assert stats["losses"] == 1
        assert stats["three_crowns"] == 2

    def test_stats_by_battle_type(self, db):
        seed_battles(db, [
            {"battle_type": "PvP", "player_crowns": 3, "opponent_crowns": 1},
            {"battle_type": "PvP", "player_crowns": 1, "opponent_crowns": 3, "trophy_change": -30},
            {"battle_type": "clanWarWarDay", "player_crowns": 3, "opponent_crowns": 0},
        ])
        by_type = {t["battle_type"]: t for t in db.get_stats_by_battle_type()}
        assert by_type["PvP"]["total"] == 2
        assert by_type["clanWarWarDay"]["wins"] == 1

    def test_deck_stats_min_battles(self, db):
        seed_battles(db, [
            {"player_crowns": 3, "opponent_crowns": 1},
            {"player_crowns": 3, "opponent_crowns": 0},
            {"player_crowns": 1, "opponent_crowns": 3, "trophy_change": -30},
        ])
        assert len(db.get_deck_stats(min_battles=3)) == 1
        assert len(db.get_deck_stats(min_battles=4)) == 0

    def test_deck_stats_win_rate(self, db):
        seed_battles(db, [
            {"player_crowns": 3, "opponent_crowns": 1},
            {"player_crowns": 3, "opponent_crowns": 0},
            {"player_crowns": 1, "opponent_crowns": 3, "trophy_change": -30},
        ])
        decks = db.get_deck_stats(min_battles=1)
        assert len(decks) == 1
        assert decks[0]["win_rate"] == pytest.approx(66.7, abs=0.1)

    def test_crown_distribution(self, db):
        seed_battles(db, [
            {"player_crowns": 3, "opponent_crowns": 1},
            {"player_crowns": 3, "opponent_crowns": 0},
            {"player_crowns": 1, "opponent_crowns": 3, "trophy_change": -30},
        ])
        dist = db.get_crown_distribution()
        assert dist["win"].get(3, 0) == 2
        assert dist["loss"].get(1, 0) == 1

    def test_card_matchup_stats(self, db):
        seed_battles(db, [
            {"player_crowns": 3, "opponent_crowns": 1},
            {"player_crowns": 3, "opponent_crowns": 0},
            {"player_crowns": 1, "opponent_crowns": 3, "trophy_change": -30},
        ])
        matchups = db.get_card_matchup_stats(min_battles=1)
        hog = [m for m in matchups if m["card_name"] == "Hog Rider"]
        assert len(hog) == 1
        assert hog[0]["times_faced"] == 3
        assert hog[0]["win_rate"] == pytest.approx(66.7, abs=0.1)

    def test_recent_battles_ordering(self, db):
        seed_battles(db, [
            {"battle_time": "20260214T180000.000Z", "opponent_name": "First"},
            {"battle_time": "20260214T200000.000Z", "opponent_name": "Last"},
            {"battle_time": "20260214T190000.000Z", "opponent_name": "Middle"},
        ])
        recent = db.get_recent_battles(limit=3)
        assert [r["opponent_name"] for r in recent] == ["Last", "Middle", "First"]

    def test_recent_battles_limit(self, db):
        seed_battles(db, [
            {"battle_time": "20260214T180000.000Z"},
            {"battle_time": "20260214T190000.000Z"},
            {"battle_time": "20260214T200000.000Z"},
        ])
        assert len(db.get_recent_battles(limit=2)) == 2

    def test_time_of_day_stats(self, db):
        seed_battles(db, [
            {"battle_time": "20260214T180000.000Z", "player_crowns": 3, "opponent_crowns": 1},
            {"battle_time": "20260214T180500.000Z", "player_crowns": 3, "opponent_crowns": 0},
            {"battle_time": "20260214T220000.000Z", "player_crowns": 1, "opponent_crowns": 3,
             "trophy_change": -30},
        ])
        hours = {t["hour"]: t for t in db.get_time_of_day_stats()}
        assert 18 in hours
        assert hours[18]["total"] == 2
        assert hours[18]["wins"] == 2


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    def test_empty_db_overall_stats(self, db):
        stats = db.get_overall_stats()
        assert stats["total"] == 0

    def test_empty_db_deck_stats(self, db):
        assert db.get_deck_stats() == []

    def test_empty_db_matchups(self, db):
        assert db.get_card_matchup_stats() == []

    def test_battle_with_missing_fields(self, db):
        sparse_battle = {
            "type": "PvP",
            "battleTime": "20260214T180000.000Z",
            "team": [{"tag": "#ABC", "crowns": 1, "cards": []}],
            "opponent": [{"tag": "#DEF", "crowns": 0, "cards": []}],
        }
        battle_id, is_new = db.store_battle(sparse_battle, "#ABC")
        assert is_new

    def test_battle_with_no_trophy_change(self, db):
        battle = make_battle()
        battle["team"][0].pop("trophyChange", None)
        db.store_battle(battle, "#L90009GPP")
        row = db.conn.execute("SELECT player_trophy_change FROM battles").fetchone()
        assert row["player_trophy_change"] is None


# =============================================================================
# API CLIENT (mocked HTTP)
# =============================================================================

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

        cr_tracker.ClashRoyaleAPI("fake-key").get_player("L90009GPP")

        called_req = mock_urlopen_fn.call_args[0][0]
        assert "%23L90009GPP" in called_req.full_url

    def test_auth_header_set(self, mock_urlopen):
        mock_urlopen_fn, response = mock_urlopen
        response.read.return_value = b'[]'

        cr_tracker.ClashRoyaleAPI("my-secret-key").get_battle_log("L90009GPP")

        called_req = mock_urlopen_fn.call_args[0][0]
        assert called_req.get_header("Authorization") == "Bearer my-secret-key"

    def test_http_error_raises(self):
        api = cr_tracker.ClashRoyaleAPI("fake-key")
        with patch("urllib.request.urlopen") as m:
            m.side_effect = urllib.error.HTTPError(
                "https://example.com", 403, "Forbidden", {}, None
            )
            with pytest.raises(Exception, match="403"):
                api.get_player("L90009GPP")

    def test_url_error_raises(self):
        api = cr_tracker.ClashRoyaleAPI("fake-key")
        with patch("urllib.request.urlopen") as m:
            m.side_effect = urllib.error.URLError("DNS lookup failed")
            with pytest.raises(Exception, match="Connection Error"):
                api.get_player("L90009GPP")


# =============================================================================
# REPORTING (smoke tests — verify no crashes)
# =============================================================================

def _seed_reporting_db(db):
    """Populate DB with enough data for all reports."""
    db.store_player_snapshot(make_player_profile())
    times = [f"2026021{4 + i // 10}T{18 + i % 6:02d}0000.000Z" for i in range(10)]
    for i, t in enumerate(times):
        crowns = 3 if i % 3 != 2 else 1
        opp_crowns = 0 if crowns == 3 else 3
        tc = 30 if crowns > opp_crowns else -30
        db.store_battle(
            make_battle(battle_time=t, player_crowns=crowns,
                        opponent_crowns=opp_crowns, trophy_change=tc,
                        opponent_name=f"Opp_{i}"),
            "#L90009GPP",
        )


class TestReporting:
    def test_print_overall_stats_with_data(self, db, capsys):
        _seed_reporting_db(db)
        cr_tracker.print_overall_stats(db)
        assert "TRACKED BATTLES" in capsys.readouterr().out

    def test_print_overall_stats_empty(self, db, capsys):
        cr_tracker.print_overall_stats(db)
        assert "No battles tracked" in capsys.readouterr().out

    def test_print_deck_stats(self, db, capsys):
        _seed_reporting_db(db)
        cr_tracker.print_deck_stats(db)
        assert "DECK PERFORMANCE" in capsys.readouterr().out

    def test_print_crown_distribution(self, db, capsys):
        _seed_reporting_db(db)
        cr_tracker.print_crown_distribution(db)
        assert "CROWN DISTRIBUTION" in capsys.readouterr().out

    def test_print_matchup_stats(self, db, capsys):
        _seed_reporting_db(db)
        cr_tracker.print_matchup_stats(db)
        assert "CARD MATCHUP" in capsys.readouterr().out

    def test_print_recent_battles(self, db, capsys):
        _seed_reporting_db(db)
        cr_tracker.print_recent_battles(db, limit=5)
        assert "LAST 5 BATTLES" in capsys.readouterr().out

    def test_print_recent_battles_empty(self, db, capsys):
        cr_tracker.print_recent_battles(db)
        assert "No battles tracked" in capsys.readouterr().out


# =============================================================================
# CLI
# =============================================================================

class TestCLI:
    def test_stats_on_empty_db(self, db_path):
        with patch("sys.argv", ["cr_tracker.py", "--stats", "--db", db_path]):
            assert cr_tracker.main() == 0

    def test_fetch_without_credentials_fails(self, db_path, capsys):
        env = {k: v for k, v in os.environ.items()
               if k not in ("CR_API_KEY", "CR_PLAYER_TAG")}
        with patch.dict(os.environ, env, clear=True):
            with patch("sys.argv", ["cr_tracker.py", "--fetch", "--db", db_path]):
                assert cr_tracker.main() == 1
        assert "required" in capsys.readouterr().out

    def test_default_shows_help(self, db_path, capsys):
        with patch("sys.argv", ["cr_tracker.py", "--db", db_path]):
            assert cr_tracker.main() == 0
        assert "Battles tracked" in capsys.readouterr().out

    def test_recent_flag(self, db_path):
        with patch("sys.argv", ["cr_tracker.py", "--recent", "5", "--db", db_path]):
            assert cr_tracker.main() == 0

    def test_multiple_flags(self, db_path):
        with patch("sys.argv", [
            "cr_tracker.py", "--stats", "--crowns", "--matchups", "--db", db_path,
        ]):
            assert cr_tracker.main() == 0


# =============================================================================
# FETCH INTEGRATION (mocked API)
# =============================================================================

class TestFetchAndStore:
    def test_stores_battles_and_snapshot(self, db, capsys):
        battles = [make_battle(battle_time=f"20260214T{18 + i:02d}0000.000Z") for i in range(5)]
        with patch.object(cr_tracker.ClashRoyaleAPI, "get_player", return_value=make_player_profile()):
            with patch.object(cr_tracker.ClashRoyaleAPI, "get_battle_log", return_value=battles):
                cr_tracker.fetch_and_store("fake-key", "L90009GPP", db)
        assert db.get_total_battles() == 5
        assert db.get_all_time_api_stats().get("player_tag") is not None
        assert "5 NEW" in capsys.readouterr().out

    def test_deduplicates_on_second_run(self, db, capsys):
        battles = [make_battle(battle_time=f"20260214T{18 + i:02d}0000.000Z") for i in range(3)]
        with patch.object(cr_tracker.ClashRoyaleAPI, "get_player", return_value=make_player_profile()):
            with patch.object(cr_tracker.ClashRoyaleAPI, "get_battle_log", return_value=battles):
                cr_tracker.fetch_and_store("fake-key", "L90009GPP", db)
                capsys.readouterr()  # discard first run output
                cr_tracker.fetch_and_store("fake-key", "L90009GPP", db)
        assert db.get_total_battles() == 3
        assert "0 NEW" in capsys.readouterr().out

    def test_handles_api_error_gracefully(self, db, capsys):
        with patch.object(cr_tracker.ClashRoyaleAPI, "get_player",
                          side_effect=Exception("API Error 403: Forbidden")):
            cr_tracker.fetch_and_store("fake-key", "L90009GPP", db)
        assert "Error fetching player" in capsys.readouterr().out
        assert db.get_total_battles() == 0
