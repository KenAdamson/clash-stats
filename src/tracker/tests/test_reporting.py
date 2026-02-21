"""Smoke tests for reporting functions — verify no crashes."""

from tracker import reporting
from tracker.tests.conftest import seed_reporting_db


class TestReporting:
    def test_print_overall_stats_with_data(self, session, capsys):
        seed_reporting_db(session)
        reporting.print_overall_stats(session)
        assert "TRACKED BATTLES" in capsys.readouterr().out

    def test_print_overall_stats_empty(self, session, capsys):
        reporting.print_overall_stats(session)
        assert "No battles tracked" in capsys.readouterr().out

    def test_print_deck_stats(self, session, capsys):
        seed_reporting_db(session)
        reporting.print_deck_stats(session)
        assert "DECK PERFORMANCE" in capsys.readouterr().out

    def test_print_crown_distribution(self, session, capsys):
        seed_reporting_db(session)
        reporting.print_crown_distribution(session)
        assert "CROWN DISTRIBUTION" in capsys.readouterr().out

    def test_print_matchup_stats(self, session, capsys):
        seed_reporting_db(session)
        reporting.print_matchup_stats(session)
        assert "CARD MATCHUP" in capsys.readouterr().out

    def test_print_recent_battles(self, session, capsys):
        seed_reporting_db(session)
        reporting.print_recent_battles(session, limit=5)
        assert "LAST 5 BATTLES" in capsys.readouterr().out

    def test_print_recent_battles_empty(self, session, capsys):
        reporting.print_recent_battles(session)
        assert "No battles tracked" in capsys.readouterr().out

    def test_print_streaks(self, session, capsys):
        seed_reporting_db(session)
        reporting.print_streaks(session)
        assert "STREAK ANALYSIS" in capsys.readouterr().out

    def test_print_streaks_empty(self, session, capsys):
        reporting.print_streaks(session)
        assert "No battles tracked" in capsys.readouterr().out

    def test_print_rolling_stats(self, session, capsys):
        seed_reporting_db(session)
        reporting.print_rolling_stats(session, 35)
        assert "rolling window" in capsys.readouterr().out

    def test_print_rolling_stats_empty(self, session, capsys):
        reporting.print_rolling_stats(session, 35)
        assert "No battles tracked" in capsys.readouterr().out

    def test_print_trophy_history(self, session, capsys):
        seed_reporting_db(session)
        reporting.print_trophy_history(session)
        assert "TROPHY PROGRESSION" in capsys.readouterr().out

    def test_print_trophy_history_empty(self, session, capsys):
        reporting.print_trophy_history(session)
        assert "No trophy data" in capsys.readouterr().out

    def test_print_archetype_stats(self, session, capsys):
        seed_reporting_db(session)
        reporting.print_archetype_stats(session)
        assert "ARCHETYPE" in capsys.readouterr().out

    def test_print_archetype_stats_empty(self, session, capsys):
        reporting.print_archetype_stats(session)
        assert "Not enough data" in capsys.readouterr().out
