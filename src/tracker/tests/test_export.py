"""Tests for export functionality."""

import json

from tracker import analytics
from tracker.export import export_data
from tracker.tests.conftest import seed_battles


class TestExport:
    def test_export_json(self, session, tmp_path):
        seed_battles(session, [
            {"player_crowns": 3, "opponent_crowns": 1, "trophy_change": 30},
            {"player_crowns": 1, "opponent_crowns": 3, "trophy_change": -30},
        ])
        out_file = str(tmp_path / "export.json")
        data = analytics.get_card_matchup_stats(session, min_battles=1)
        export_data(data, "json", out_file)

        with open(out_file) as f:
            loaded = json.load(f)
        assert isinstance(loaded, list)
        assert len(loaded) > 0
        assert "card_name" in loaded[0]

    def test_export_csv(self, session, tmp_path):
        seed_battles(session, [
            {"player_crowns": 3, "opponent_crowns": 1, "trophy_change": 30},
            {"player_crowns": 1, "opponent_crowns": 3, "trophy_change": -30},
        ])
        out_file = str(tmp_path / "export.csv")
        data = analytics.get_card_matchup_stats(session, min_battles=1)
        export_data(data, "csv", out_file)

        with open(out_file) as f:
            lines = f.readlines()
        assert len(lines) > 1
        assert "card_name" in lines[0]

    def test_export_json_to_stdout(self, session, capsys):
        seed_battles(session, [
            {"player_crowns": 3, "opponent_crowns": 1, "trophy_change": 30},
        ])
        data = analytics.get_rolling_stats(session, 10)
        export_data(data, "json")
        output = capsys.readouterr().out
        loaded = json.loads(output)
        assert isinstance(loaded, list)

    def test_export_empty_data(self, tmp_path):
        out_file = str(tmp_path / "empty.csv")
        export_data([], "csv", out_file)
        with open(out_file) as f:
            assert f.read() == ""
