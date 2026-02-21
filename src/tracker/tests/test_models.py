"""Tests for ORM models, schema, battle ID/hash generation, and battle storage."""

import json
import pytest

from tracker import analytics
from tracker.models import Battle, DeckCard, PlayerSnapshot
from tracker.tests.conftest import (
    OPPONENT_DECK, PLAYER_DECK, make_battle, make_card, make_player_profile,
    seed_battles,
)


# =============================================================================
# SCHEMA
# =============================================================================

class TestSchema:
    def test_creates_tables(self, session):
        from sqlalchemy import inspect
        insp = inspect(session.bind)
        tables = set(insp.get_table_names())
        assert "battles" in tables
        assert "player_snapshots" in tables
        assert "deck_cards" in tables

    def test_columns_on_deck_cards(self, session):
        from sqlalchemy import inspect
        insp = inspect(session.bind)
        columns = {c["name"] for c in insp.get_columns("deck_cards")}
        assert "evolution_level" in columns
        assert "star_level" in columns

    def test_columns_on_battles(self, session):
        from sqlalchemy import inspect
        insp = inspect(session.bind)
        columns = {c["name"] for c in insp.get_columns("battles")}
        assert "player_elixir_leaked" in columns
        assert "opponent_elixir_leaked" in columns
        assert "battle_duration" in columns


# =============================================================================
# BATTLE ID GENERATION
# =============================================================================

class TestBattleId:
    def test_deterministic(self):
        battle = make_battle()
        assert analytics.generate_battle_id(battle) == analytics.generate_battle_id(battle)

    def test_different_for_different_times(self):
        b1 = make_battle(battle_time="20260214T180000.000Z")
        b2 = make_battle(battle_time="20260214T190000.000Z")
        assert analytics.generate_battle_id(b1) != analytics.generate_battle_id(b2)

    def test_different_for_different_crowns(self):
        b1 = make_battle(player_crowns=3, opponent_crowns=1)
        b2 = make_battle(player_crowns=2, opponent_crowns=1)
        assert analytics.generate_battle_id(b1) != analytics.generate_battle_id(b2)

    def test_is_32_hex_chars(self):
        bid = analytics.generate_battle_id(make_battle())
        assert len(bid) == 32
        int(bid, 16)


# =============================================================================
# DECK HASH
# =============================================================================

class TestDeckHash:
    def test_ignores_level(self):
        d1 = [make_card("P.E.K.K.A", level=14), make_card("Witch", level=14)]
        d2 = [make_card("P.E.K.K.A", level=16), make_card("Witch", level=16)]
        assert analytics.generate_deck_hash(d1) == analytics.generate_deck_hash(d2)

    def test_ignores_order(self):
        d1 = [make_card("Witch"), make_card("P.E.K.K.A")]
        d2 = [make_card("P.E.K.K.A"), make_card("Witch")]
        assert analytics.generate_deck_hash(d1) == analytics.generate_deck_hash(d2)

    def test_different_for_different_cards(self):
        d1 = [make_card("Witch"), make_card("P.E.K.K.A")]
        d2 = [make_card("Witch"), make_card("Hog Rider")]
        assert analytics.generate_deck_hash(d1) != analytics.generate_deck_hash(d2)

    def test_different_for_different_evolutions(self):
        d1 = [make_card("Witch", evolution_level=0), make_card("P.E.K.K.A")]
        d2 = [make_card("Witch", evolution_level=1), make_card("P.E.K.K.A")]
        assert analytics.generate_deck_hash(d1) != analytics.generate_deck_hash(d2)

    def test_no_evo_key_matches_zero(self):
        d1 = [{"name": "Witch"}, {"name": "P.E.K.K.A"}]
        d2 = [make_card("Witch", evolution_level=0), make_card("P.E.K.K.A", evolution_level=0)]
        assert analytics.generate_deck_hash(d1) == analytics.generate_deck_hash(d2)


# =============================================================================
# BATTLE STORAGE
# =============================================================================

class TestBattleStorage:
    def test_returns_new(self, session):
        battle_id, is_new = analytics.store_battle(session, make_battle(), "#L90009GPP")
        assert is_new
        assert len(battle_id) == 32

    def test_deduplicates(self, session):
        battle = make_battle()
        _, is_new1 = analytics.store_battle(session, battle, "#L90009GPP")
        _, is_new2 = analytics.store_battle(session, battle, "#L90009GPP")
        assert is_new1
        assert not is_new2

    def test_total_increments(self, session):
        assert analytics.get_total_battles(session) == 0
        analytics.store_battle(session, make_battle(battle_time="20260214T180000.000Z"), "#L90009GPP")
        assert analytics.get_total_battles(session) == 1
        analytics.store_battle(session, make_battle(battle_time="20260214T190000.000Z"), "#L90009GPP")
        assert analytics.get_total_battles(session) == 2

    @pytest.mark.parametrize("p_crowns,o_crowns,expected", [
        (3, 1, "win"),
        (1, 3, "loss"),
        (1, 1, "draw"),
    ])
    def test_result_classification(self, session, p_crowns, o_crowns, expected):
        tc = 30 if expected == "win" else (-30 if expected == "loss" else 0)
        analytics.store_battle(
            session,
            make_battle(player_crowns=p_crowns, opponent_crowns=o_crowns, trophy_change=tc),
            "#L90009GPP",
        )
        battle = session.query(Battle).first()
        assert battle.result == expected

    def test_crown_differential(self, session):
        analytics.store_battle(session, make_battle(player_crowns=3, opponent_crowns=1), "#L90009GPP")
        battle = session.query(Battle).first()
        assert battle.crown_differential == 2

    def test_deck_cards_created(self, session):
        analytics.store_battle(session, make_battle(), "#L90009GPP")
        player_cards = session.query(DeckCard).filter_by(is_player_deck=1).count()
        opp_cards = session.query(DeckCard).filter_by(is_player_deck=0).count()
        assert player_cards == 8
        assert opp_cards == 8

    def test_raw_json_preserved(self, session):
        battle = make_battle()
        analytics.store_battle(session, battle, "#L90009GPP")
        stored = session.query(Battle).first()
        restored = json.loads(stored.raw_json)
        assert restored["battleTime"] == battle["battleTime"]

    def test_preserves_trophy_data(self, session):
        analytics.store_battle(session, make_battle(trophy_change=30), "#L90009GPP")
        battle = session.query(Battle).first()
        assert battle.player_starting_trophies == 10900
        assert battle.player_trophy_change == 30

    def test_elixir_leaked_stored(self, session):
        analytics.store_battle(session, make_battle(), "#L90009GPP")
        battle = session.query(Battle).first()
        assert battle.player_elixir_leaked == pytest.approx(0.5)
        assert battle.opponent_elixir_leaked == pytest.approx(1.3)

    def test_battle_duration_stored(self, session):
        battle = make_battle()
        battle["battleDuration"] = 180
        analytics.store_battle(session, battle, "#L90009GPP")
        stored = session.query(Battle).first()
        assert stored.battle_duration == 180

    def test_evolution_level_in_deck_cards(self, session):
        analytics.store_battle(session, make_battle(), "#L90009GPP")
        card = session.query(DeckCard).filter_by(card_name="Witch", is_player_deck=1).first()
        assert card.evolution_level == 1

    def test_star_level_default_zero(self, session):
        analytics.store_battle(session, make_battle(), "#L90009GPP")
        card = session.query(DeckCard).filter_by(card_name="P.E.K.K.A", is_player_deck=1).first()
        assert card.star_level == 0


# =============================================================================
# PLAYER SNAPSHOTS
# =============================================================================

class TestPlayerSnapshots:
    def test_store_snapshot(self, session):
        analytics.store_player_snapshot(session, make_player_profile())
        snapshot = session.query(PlayerSnapshot).first()
        assert snapshot.player_tag == "#L90009GPP"
        assert snapshot.trophies == 10900
        assert snapshot.wins == 1500

    def test_store_multiple(self, session):
        analytics.store_player_snapshot(session, make_player_profile(trophies=10900))
        analytics.store_player_snapshot(session, make_player_profile(trophies=10930))
        assert session.query(PlayerSnapshot).count() == 2

    def test_get_latest(self, session):
        analytics.store_player_snapshot(session, make_player_profile(trophies=10900))
        analytics.store_player_snapshot(session, make_player_profile(trophies=10930))
        stats = analytics.get_all_time_api_stats(session)
        assert stats["trophies"] == 10930

    def test_get_latest_empty_db(self, session):
        assert analytics.get_all_time_api_stats(session) == {}


# =============================================================================
# EDGE CASES
# =============================================================================

class TestEdgeCases:
    def test_empty_db_overall_stats(self, session):
        stats = analytics.get_overall_stats(session)
        assert stats["total"] == 0

    def test_empty_db_deck_stats(self, session):
        assert analytics.get_deck_stats(session) == []

    def test_empty_db_matchups(self, session):
        assert analytics.get_card_matchup_stats(session) == []

    def test_battle_with_missing_fields(self, session):
        sparse_battle = {
            "type": "PvP",
            "battleTime": "20260214T180000.000Z",
            "team": [{"tag": "#ABC", "crowns": 1, "cards": []}],
            "opponent": [{"tag": "#DEF", "crowns": 0, "cards": []}],
        }
        battle_id, is_new = analytics.store_battle(session, sparse_battle, "#ABC")
        assert is_new

    def test_battle_with_no_trophy_change(self, session):
        battle = make_battle()
        battle["team"][0].pop("trophyChange", None)
        analytics.store_battle(session, battle, "#L90009GPP")
        stored = session.query(Battle).first()
        assert stored.player_trophy_change is None
