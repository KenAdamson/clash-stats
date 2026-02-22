"""Tests for the RoyaleAPI replay parser and storage."""

import json
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from tracker import analytics
from tracker.models import Base, Battle, ReplayEvent, ReplaySummary
from tracker.replays import parse_replay_html, store_replay_data, _parse_replay_url
from tracker.tests.conftest import make_battle, seed_battles

FIXTURE_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def replay_html():
    """Load the sample replay HTML fixture."""
    return (FIXTURE_DIR / "replay_sample.html").read_text()


@pytest.fixture
def session():
    """Fresh in-memory SQLAlchemy session with all tables."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s


class TestParseReplayHTML:
    """Tests for the pure HTML parser."""

    def test_parses_all_events(self, replay_html):
        result = parse_replay_html(replay_html)
        events = result["events"]
        # 16 team + 10 opponent = 26 total markers
        assert len(events) == 26

    def test_team_events_count(self, replay_html):
        result = parse_replay_html(replay_html)
        team_events = [e for e in result["events"] if e["side"] == "team"]
        assert len(team_events) == 16

    def test_opponent_events_count(self, replay_html):
        result = parse_replay_html(replay_html)
        opp_events = [e for e in result["events"] if e["side"] == "opponent"]
        assert len(opp_events) == 10

    def test_first_event_is_opponent_elixir_golem(self, replay_html):
        result = parse_replay_html(replay_html)
        # Events are in DOM order — first marker is opponent's elixir-golem
        first = result["events"][0]
        assert first["side"] == "opponent"
        assert first["card_name"] == "elixir-golem"
        assert first["game_tick"] == 132
        assert first["arena_x"] == 8499
        assert first["arena_y"] == 500
        assert first["play_number"] == 1

    def test_team_pekka_first_play(self, replay_html):
        result = parse_replay_html(replay_html)
        # Second marker is team's pekka
        pekka = result["events"][1]
        assert pekka["side"] == "team"
        assert pekka["card_name"] == "pekka"
        assert pekka["game_tick"] == 226
        assert pekka["arena_x"] == 7499
        assert pekka["arena_y"] == 31499
        assert pekka["play_number"] == 1

    def test_play_number_increments(self, replay_html):
        result = parse_replay_html(replay_html)
        pekka_plays = [
            e for e in result["events"]
            if e["card_name"] == "pekka" and e["side"] == "team"
        ]
        assert len(pekka_plays) == 3
        assert pekka_plays[0]["play_number"] == 1
        assert pekka_plays[1]["play_number"] == 2
        assert pekka_plays[2]["play_number"] == 3

    def test_miner_plays(self, replay_html):
        result = parse_replay_html(replay_html)
        miner_plays = [
            e for e in result["events"]
            if e["card_name"] == "miner" and e["side"] == "team"
        ]
        assert len(miner_plays) == 3
        # Check coordinates differ between plays
        positions = [(m["arena_x"], m["arena_y"]) for m in miner_plays]
        assert len(set(positions)) > 1  # Miner placed in different spots

    def test_ability_used_default_zero(self, replay_html):
        result = parse_replay_html(replay_html)
        for event in result["events"]:
            assert event["ability_used"] == 0

    def test_last_event_is_miner(self, replay_html):
        result = parse_replay_html(replay_html)
        last = result["events"][-1]
        assert last["card_name"] == "miner"
        assert last["game_tick"] == 2881
        assert last["play_number"] == 3

    def test_events_sorted_by_tick(self, replay_html):
        """Events should be in chronological order (DOM order)."""
        result = parse_replay_html(replay_html)
        ticks = [e["game_tick"] for e in result["events"]]
        assert ticks == sorted(ticks)


class TestParseSummaries:
    """Tests for the elixir summary parser."""

    def test_two_summaries(self, replay_html):
        result = parse_replay_html(replay_html)
        assert len(result["summaries"]) == 2

    def test_team_summary(self, replay_html):
        result = parse_replay_html(replay_html)
        team = result["summaries"][0]
        assert team["side"] == "team"
        assert team["total_plays"] == 16
        assert team["total_elixir"] == 66
        assert team["troop_plays"] == 12
        assert team["troop_elixir"] == 54
        assert team["spell_plays"] == 4
        assert team["spell_elixir"] == 12
        assert team["building_plays"] == 0
        assert team["building_elixir"] == 0
        assert team["ability_plays"] == 0
        assert team["ability_elixir"] == 0
        assert team["elixir_leaked"] == 1.75

    def test_opponent_summary(self, replay_html):
        result = parse_replay_html(replay_html)
        opp = result["summaries"][1]
        assert opp["side"] == "opponent"
        assert opp["total_plays"] == 10
        assert opp["total_elixir"] == 41
        assert opp["troop_plays"] == 7
        assert opp["troop_elixir"] == 30
        assert opp["spell_plays"] == 3
        assert opp["spell_elixir"] == 11
        assert opp["elixir_leaked"] == 22.24


class TestParseReplayUrl:
    """Tests for replay URL parsing."""

    def test_parse_full_url(self):
        url = "/replay?tag=00PY999ULVUC&team_tags=L90009GPP&opponent_tags=VVLUPLJG0&team_crowns=3&opponent_crowns=0"
        result = _parse_replay_url(url)
        assert result["tag"] == "00PY999ULVUC"
        assert result["team_tags"] == "L90009GPP"
        assert result["opponent_tags"] == "VVLUPLJG0"
        assert result["team_crowns"] == 3
        assert result["opponent_crowns"] == 0

    def test_parse_with_extra_params(self):
        url = "/replay?tag=ABC123&team_tags=X&opponent_tags=Y&team_crowns=1&opponent_crowns=2&overlay=1&lang=en"
        result = _parse_replay_url(url)
        assert result["tag"] == "ABC123"
        assert result["team_crowns"] == 1
        assert result["opponent_crowns"] == 2


class TestStoreReplayData:
    """Tests for database storage of parsed replay data."""

    def test_stores_events(self, session, replay_html):
        # Create a battle to link to
        battle = make_battle()
        analytics.store_battle(session, battle, "#L90009GPP")

        # Get the battle_id that was generated
        b = session.query(Battle).first()
        data = parse_replay_html(replay_html)
        store_replay_data(session, b.battle_id, data)

        events = session.query(ReplayEvent).all()
        assert len(events) == 26

    def test_stores_summaries(self, session, replay_html):
        battle = make_battle()
        analytics.store_battle(session, battle, "#L90009GPP")
        b = session.query(Battle).first()

        data = parse_replay_html(replay_html)
        store_replay_data(session, b.battle_id, data)

        summaries = session.query(ReplaySummary).all()
        assert len(summaries) == 2

    def test_marks_battle_as_fetched(self, session, replay_html):
        battle = make_battle()
        analytics.store_battle(session, battle, "#L90009GPP")
        b = session.query(Battle).first()
        assert b.replay_fetched == 0

        data = parse_replay_html(replay_html)
        store_replay_data(session, b.battle_id, data)

        session.refresh(b)
        assert b.replay_fetched == 1

    def test_event_foreign_key(self, session, replay_html):
        battle = make_battle()
        analytics.store_battle(session, battle, "#L90009GPP")
        b = session.query(Battle).first()

        data = parse_replay_html(replay_html)
        store_replay_data(session, b.battle_id, data)

        event = session.query(ReplayEvent).first()
        assert event.battle_id == b.battle_id

    def test_summary_elixir_values(self, session, replay_html):
        battle = make_battle()
        analytics.store_battle(session, battle, "#L90009GPP")
        b = session.query(Battle).first()

        data = parse_replay_html(replay_html)
        store_replay_data(session, b.battle_id, data)

        team_summary = (
            session.query(ReplaySummary)
            .filter_by(battle_id=b.battle_id, side="team")
            .first()
        )
        assert team_summary.total_elixir == 66
        assert team_summary.elixir_leaked == 1.75


class TestEmptyReplay:
    """Tests for edge cases with empty or malformed HTML."""

    def test_empty_html(self):
        result = parse_replay_html("")
        assert result["events"] == []
        assert result["summaries"] == []

    def test_no_markers(self):
        html = '<div class="replay_map"><div class="markers"></div></div>'
        result = parse_replay_html(html)
        assert result["events"] == []

    def test_no_elixir_tables(self):
        html = '<div class="battle_replay"></div>'
        result = parse_replay_html(html)
        assert result["summaries"] == []
