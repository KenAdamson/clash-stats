"""Tests for analytics queries — streaks, rolling, archetypes, trophy history, etc."""

import pytest

from tracker import analytics
from tracker.archetypes import classify_archetype
from tracker.tests.conftest import (
    OPPONENT_DECK, make_battle, make_card, make_player_profile, seed_battles,
)


class TestOverallStats:
    def test_overall_stats(self, session):
        seed_battles(session, [
            {"player_crowns": 3, "opponent_crowns": 1, "trophy_change": 30},
            {"player_crowns": 1, "opponent_crowns": 3, "trophy_change": -30},
            {"player_crowns": 3, "opponent_crowns": 0, "trophy_change": 31},
        ])
        stats = analytics.get_overall_stats(session)
        assert stats["total"] == 3
        assert stats["wins"] == 2
        assert stats["losses"] == 1
        assert stats["three_crowns"] == 2

    def test_stats_by_battle_type(self, session):
        seed_battles(session, [
            {"battle_type": "PvP", "player_crowns": 3, "opponent_crowns": 1},
            {"battle_type": "PvP", "player_crowns": 1, "opponent_crowns": 3, "trophy_change": -30},
            {"battle_type": "clanWarWarDay", "player_crowns": 3, "opponent_crowns": 0},
        ])
        by_type = {t["battle_type"]: t for t in analytics.get_stats_by_battle_type(session)}
        assert by_type["PvP"]["total"] == 2
        assert by_type["clanWarWarDay"]["wins"] == 1

    def test_deck_stats_min_battles(self, session):
        seed_battles(session, [
            {"player_crowns": 3, "opponent_crowns": 1},
            {"player_crowns": 3, "opponent_crowns": 0},
            {"player_crowns": 1, "opponent_crowns": 3, "trophy_change": -30},
        ])
        assert len(analytics.get_deck_stats(session, min_battles=3)) == 1
        assert len(analytics.get_deck_stats(session, min_battles=4)) == 0

    def test_deck_stats_win_rate(self, session):
        seed_battles(session, [
            {"player_crowns": 3, "opponent_crowns": 1},
            {"player_crowns": 3, "opponent_crowns": 0},
            {"player_crowns": 1, "opponent_crowns": 3, "trophy_change": -30},
        ])
        decks = analytics.get_deck_stats(session, min_battles=1)
        assert len(decks) == 1
        assert decks[0]["win_rate"] == pytest.approx(66.7, abs=0.1)

    def test_crown_distribution(self, session):
        seed_battles(session, [
            {"player_crowns": 3, "opponent_crowns": 1},
            {"player_crowns": 3, "opponent_crowns": 0},
            {"player_crowns": 1, "opponent_crowns": 3, "trophy_change": -30},
        ])
        dist = analytics.get_crown_distribution(session)
        assert dist["win"].get(3, 0) == 2
        assert dist["loss"].get(1, 0) == 1

    def test_card_matchup_stats(self, session):
        seed_battles(session, [
            {"player_crowns": 3, "opponent_crowns": 1},
            {"player_crowns": 3, "opponent_crowns": 0},
            {"player_crowns": 1, "opponent_crowns": 3, "trophy_change": -30},
        ])
        matchups = analytics.get_card_matchup_stats(session, min_battles=1)
        hog = [m for m in matchups if m["card_name"] == "Hog Rider"]
        assert len(hog) == 1
        assert hog[0]["times_faced"] == 3
        assert hog[0]["win_rate"] == pytest.approx(66.7, abs=0.1)

    def test_recent_battles_ordering(self, session):
        seed_battles(session, [
            {"battle_time": "20260214T180000.000Z", "opponent_name": "First"},
            {"battle_time": "20260214T200000.000Z", "opponent_name": "Last"},
            {"battle_time": "20260214T190000.000Z", "opponent_name": "Middle"},
        ])
        recent = analytics.get_recent_battles(session, limit=3)
        assert [r["opponent_name"] for r in recent] == ["Last", "Middle", "First"]

    def test_recent_battles_limit(self, session):
        seed_battles(session, [
            {"battle_time": "20260214T180000.000Z"},
            {"battle_time": "20260214T190000.000Z"},
            {"battle_time": "20260214T200000.000Z"},
        ])
        assert len(analytics.get_recent_battles(session, limit=2)) == 2

    def test_time_of_day_stats(self, session):
        seed_battles(session, [
            {"battle_time": "20260214T180000.000Z", "player_crowns": 3, "opponent_crowns": 1},
            {"battle_time": "20260214T180500.000Z", "player_crowns": 3, "opponent_crowns": 0},
            {"battle_time": "20260214T220000.000Z", "player_crowns": 1, "opponent_crowns": 3,
             "trophy_change": -30},
        ])
        hours = {t["hour"]: t for t in analytics.get_time_of_day_stats(session)}
        assert 18 in hours
        assert hours[18]["total"] == 2
        assert hours[18]["wins"] == 2


class TestStreaks:
    def test_empty_db(self, session):
        data = analytics.get_streaks(session)
        assert data["current_streak"] is None
        assert data["streaks"] == []

    def test_single_win(self, session):
        seed_battles(session, [{"player_crowns": 3, "opponent_crowns": 1}])
        data = analytics.get_streaks(session)
        assert data["current_streak"]["type"] == "win"
        assert data["current_streak"]["length"] == 1

    def test_win_streak(self, session):
        seed_battles(session, [
            {"player_crowns": 3, "opponent_crowns": 1, "trophy_change": 30},
            {"player_crowns": 3, "opponent_crowns": 0, "trophy_change": 31},
            {"player_crowns": 2, "opponent_crowns": 1, "trophy_change": 29},
        ])
        data = analytics.get_streaks(session)
        assert data["longest_win_streak"]["length"] == 3
        assert data["current_streak"]["type"] == "win"

    def test_alternating(self, session):
        seed_battles(session, [
            {"player_crowns": 3, "opponent_crowns": 1, "trophy_change": 30},
            {"player_crowns": 1, "opponent_crowns": 3, "trophy_change": -30},
            {"player_crowns": 3, "opponent_crowns": 0, "trophy_change": 31},
        ])
        data = analytics.get_streaks(session)
        assert all(s["length"] == 1 for s in data["streaks"])
        assert len(data["streaks"]) == 3

    def test_loss_streak_trophies(self, session):
        seed_battles(session, [
            {"player_crowns": 1, "opponent_crowns": 3, "trophy_change": -30},
            {"player_crowns": 0, "opponent_crowns": 3, "trophy_change": -31},
        ])
        data = analytics.get_streaks(session)
        ls = data["longest_loss_streak"]
        assert ls["length"] == 2
        assert ls["start_trophies"] == 10900
        assert ls["end_trophies"] == 10900 + (-31)


class TestRollingStats:
    def test_empty_db(self, session):
        stats = analytics.get_rolling_stats(session, 35)
        assert stats["total"] == 0
        assert stats["win_rate"] == 0.0

    def test_window_limits_results(self, session):
        seed_battles(session, [
            {"player_crowns": 3, "opponent_crowns": 1, "trophy_change": 30},
            {"player_crowns": 3, "opponent_crowns": 0, "trophy_change": 31},
            {"player_crowns": 1, "opponent_crowns": 3, "trophy_change": -30},
            {"player_crowns": 3, "opponent_crowns": 1, "trophy_change": 30},
            {"player_crowns": 1, "opponent_crowns": 3, "trophy_change": -30},
        ])
        stats = analytics.get_rolling_stats(session, 3)
        assert stats["total"] == 3

    def test_larger_window_than_db(self, session):
        seed_battles(session, [
            {"player_crowns": 3, "opponent_crowns": 1, "trophy_change": 30},
            {"player_crowns": 3, "opponent_crowns": 0, "trophy_change": 31},
        ])
        stats = analytics.get_rolling_stats(session, 100)
        assert stats["total"] == 2
        assert stats["wins"] == 2
        assert stats["win_rate"] == 100.0

    def test_trophy_change_sum(self, session):
        seed_battles(session, [
            {"player_crowns": 3, "opponent_crowns": 1, "trophy_change": 30},
            {"player_crowns": 1, "opponent_crowns": 3, "trophy_change": -30},
        ])
        stats = analytics.get_rolling_stats(session, 10)
        assert stats["trophy_change"] == 0


class TestTrophyHistory:
    def test_empty_db(self, session):
        assert analytics.get_trophy_history(session) == []

    def test_chronological_order(self, session):
        seed_battles(session, [
            {"battle_time": "20260214T180000.000Z", "trophy_change": 30},
            {"battle_time": "20260214T200000.000Z", "trophy_change": -30},
            {"battle_time": "20260214T190000.000Z", "trophy_change": 31},
        ])
        history = analytics.get_trophy_history(session)
        times = [h["battle_time"] for h in history]
        assert times == sorted(times)

    def test_trophies_calculated(self, session):
        seed_battles(session, [{"trophy_change": 30}])
        history = analytics.get_trophy_history(session)
        assert history[0]["trophies"] == 10900 + 30


class TestArchetypes:
    def test_hog_classified(self):
        assert classify_archetype(OPPONENT_DECK) == "Hog Cycle"

    def test_unknown_deck(self):
        unknown_deck = [
            make_card("Arrows"), make_card("Fireball"),
            make_card("The Log"), make_card("Zap"),
            make_card("Tornado"), make_card("Rocket"),
            make_card("Lightning"), make_card("Freeze"),
        ]
        assert classify_archetype(unknown_deck) == "Unknown"

    def test_archetype_stats_aggregation(self, session):
        seed_battles(session, [
            {"player_crowns": 3, "opponent_crowns": 1, "trophy_change": 30},
            {"player_crowns": 3, "opponent_crowns": 0, "trophy_change": 31},
            {"player_crowns": 1, "opponent_crowns": 3, "trophy_change": -30},
        ])
        stats = analytics.get_archetype_stats(session, min_battles=1)
        hog = [s for s in stats if s["archetype"] == "Hog Cycle"]
        assert len(hog) == 1
        assert hog[0]["total"] == 3
        assert hog[0]["wins"] == 2

    def test_archetype_stats_empty_db(self, session):
        assert analytics.get_archetype_stats(session) == []


class TestSnapshotDiff:
    def test_single_snapshot_returns_none(self, session):
        analytics.store_player_snapshot(session, make_player_profile(trophies=10900))
        assert analytics.get_snapshot_diff(session) is None

    def test_two_snapshots_diff(self, session):
        p1 = make_player_profile(trophies=10900)
        p1["wins"] = 1500
        p1["losses"] = 1343
        analytics.store_player_snapshot(session, p1)

        p2 = make_player_profile(trophies=10947)
        p2["wins"] = 1503
        p2["losses"] = 1344
        analytics.store_player_snapshot(session, p2)

        diff = analytics.get_snapshot_diff(session)
        assert diff is not None
        assert diff["trophies"] == 47
        assert diff["wins"] == 3
        assert diff["losses"] == 1

    def test_no_change(self, session):
        profile = make_player_profile()
        analytics.store_player_snapshot(session, profile)
        analytics.store_player_snapshot(session, profile)
        diff = analytics.get_snapshot_diff(session)
        assert diff["trophies"] == 0
        assert diff["wins"] == 0
