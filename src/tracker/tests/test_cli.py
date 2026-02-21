"""Tests for CLI argument parsing and command dispatch."""

import os
from unittest.mock import patch

from tracker import cli
from tracker.tests.conftest import make_battle, make_player_profile


class TestCLI:
    def test_stats_on_empty_db(self, db_path):
        with patch("sys.argv", ["clash-stats", "--stats", "--db", db_path]):
            assert cli.main() == 0

    def test_fetch_without_credentials_fails(self, db_path, capsys):
        env = {k: v for k, v in os.environ.items()
               if k not in ("CR_API_KEY", "CR_PLAYER_TAG")}
        with patch.dict(os.environ, env, clear=True):
            with patch("sys.argv", ["clash-stats", "--fetch", "--db", db_path]):
                assert cli.main() == 1
        assert "required" in capsys.readouterr().out

    def test_default_shows_help(self, db_path, capsys):
        with patch("sys.argv", ["clash-stats", "--db", db_path]):
            assert cli.main() == 0
        assert "Battles tracked" in capsys.readouterr().out

    def test_recent_flag(self, db_path):
        with patch("sys.argv", ["clash-stats", "--recent", "5", "--db", db_path]):
            assert cli.main() == 0

    def test_multiple_flags(self, db_path):
        with patch("sys.argv", [
            "clash-stats", "--stats", "--crowns", "--matchups", "--db", db_path,
        ]):
            assert cli.main() == 0

    def test_streaks_flag(self, db_path):
        with patch("sys.argv", ["clash-stats", "--streaks", "--db", db_path]):
            assert cli.main() == 0

    def test_rolling_flag(self, db_path):
        with patch("sys.argv", ["clash-stats", "--rolling", "20", "--db", db_path]):
            assert cli.main() == 0

    def test_trophy_history_flag(self, db_path):
        with patch("sys.argv", ["clash-stats", "--trophy-history", "--db", db_path]):
            assert cli.main() == 0

    def test_archetypes_flag(self, db_path):
        with patch("sys.argv", ["clash-stats", "--archetypes", "--db", db_path]):
            assert cli.main() == 0

    def test_export_json_flag(self, db_path):
        with patch("sys.argv", ["clash-stats", "--stats", "--export", "json", "--db", db_path]):
            assert cli.main() == 0

    def test_export_csv_flag(self, db_path):
        with patch("sys.argv", ["clash-stats", "--matchups", "--export", "csv", "--db", db_path]):
            assert cli.main() == 0


class TestFetchAndStore:
    def test_stores_battles_and_snapshot(self, db_path, capsys):
        from tracker.database import init_db, get_session
        from tracker import analytics

        engine = init_db(db_path)
        session = get_session(engine)
        battles = [make_battle(battle_time=f"20260214T{18 + i:02d}0000.000Z") for i in range(5)]
        with patch.object(cli.ClashRoyaleAPI, "get_player", return_value=make_player_profile()):
            with patch.object(cli.ClashRoyaleAPI, "get_battle_log", return_value=battles):
                cli.fetch_and_store("fake-key", "L90009GPP", session)
        assert analytics.get_total_battles(session) == 5
        assert analytics.get_all_time_api_stats(session).get("player_tag") is not None
        assert "5 NEW" in capsys.readouterr().out
        session.close()
        engine.dispose()

    def test_deduplicates_on_second_run(self, db_path, capsys):
        from tracker.database import init_db, get_session
        from tracker import analytics

        engine = init_db(db_path)
        session = get_session(engine)
        battles = [make_battle(battle_time=f"20260214T{18 + i:02d}0000.000Z") for i in range(3)]
        with patch.object(cli.ClashRoyaleAPI, "get_player", return_value=make_player_profile()):
            with patch.object(cli.ClashRoyaleAPI, "get_battle_log", return_value=battles):
                cli.fetch_and_store("fake-key", "L90009GPP", session)
                capsys.readouterr()
                cli.fetch_and_store("fake-key", "L90009GPP", session)
        assert analytics.get_total_battles(session) == 3
        assert "0 NEW" in capsys.readouterr().out
        session.close()
        engine.dispose()

    def test_handles_api_error_gracefully(self, db_path, capsys):
        from tracker.database import init_db, get_session
        from tracker import analytics

        engine = init_db(db_path)
        session = get_session(engine)
        with patch.object(cli.ClashRoyaleAPI, "get_player",
                          side_effect=Exception("API Error 403: Forbidden")):
            cli.fetch_and_store("fake-key", "L90009GPP", session)
        assert "Error fetching player" in capsys.readouterr().out
        assert analytics.get_total_battles(session) == 0
        session.close()
        engine.dispose()

    def test_snapshot_diff_shown_in_fetch(self, db_path, capsys):
        from tracker.database import init_db, get_session
        from tracker import analytics

        engine = init_db(db_path)
        session = get_session(engine)

        p1 = make_player_profile(trophies=10900)
        p1["wins"] = 1500
        analytics.store_player_snapshot(session, p1)

        p2 = make_player_profile(trophies=10930)
        p2["wins"] = 1502
        battles = [make_battle()]
        with patch.object(cli.ClashRoyaleAPI, "get_player", return_value=p2):
            with patch.object(cli.ClashRoyaleAPI, "get_battle_log", return_value=battles):
                cli.fetch_and_store("fake-key", "L90009GPP", session)
        output = capsys.readouterr().out
        assert "Since last fetch" in output
        assert "+30 trophies" in output
        session.close()
        engine.dispose()
