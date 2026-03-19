"""Shared test fixtures — realistic CR API response shapes."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from tracker.models import Base
from tracker import analytics

# Import all ORM models so Base.metadata.create_all creates their tables
import tracker.ml.wp_storage  # noqa: F401
import tracker.ml.storage  # noqa: F401
import tracker.ml.cvae_storage  # noqa: F401


# =============================================================================
# CARD AND BATTLE BUILDERS
# =============================================================================

def make_card(
    name: str,
    level: int = 14,
    max_level: int = 14,
    elixir: int = 4,
    evolution_level: int = 0,
) -> dict:
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


def make_battle(
    battle_time: str = "20260214T180000.000Z",
    player_tag: str = "#L90009GPP",
    player_crowns: int = 3,
    opponent_tag: str = "#OPPONENT1",
    opponent_name: str = "xXDarkLordXx",
    opponent_crowns: int = 1,
    trophy_change: int = 30,
    battle_type: str = "PvP",
    player_deck: list = None,
    opponent_deck: list = None,
) -> dict:
    """Build a battle dict matching the CR API shape."""
    return {
        "type": battle_type,
        "battleTime": battle_time,
        "isLadderTournament": True,
        "arena": {"id": 54000061, "name": "Arena 22"},
        "gameMode": {"id": 72000006, "name": "Ladder"},
        "team": [
            {
                "tag": player_tag,
                "name": "KrylarPrime",
                "crowns": player_crowns,
                "startingTrophies": 10900,
                "trophyChange": trophy_change,
                "kingTowerHitPoints": 8000 if player_crowns < 3 else 0,
                "princessTowersHitPoints": [3000] if player_crowns < 2 else [],
                "cards": player_deck or PLAYER_DECK,
                "elixirLeaked": 0.5,
            }
        ],
        "opponent": [
            {
                "tag": opponent_tag,
                "name": opponent_name,
                "crowns": opponent_crowns,
                "startingTrophies": 10850,
                "trophyChange": -trophy_change,
                "kingTowerHitPoints": 0 if opponent_crowns == 0 else 5000,
                "princessTowersHitPoints": [],
                "cards": opponent_deck or OPPONENT_DECK,
                "elixirLeaked": 1.3,
            }
        ],
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
# SESSION FIXTURES
# =============================================================================

@pytest.fixture
def session():
    """Fresh in-memory SQLAlchemy session for each test."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(engine)
    with Session(engine) as s:
        yield s
    engine.dispose()


@pytest.fixture
def db_path(tmp_path):
    """Temp DB path for CLI tests that manage their own connection."""
    return str(tmp_path / "test.db")


def seed_battles(session: Session, specs: list[dict]) -> None:
    """Insert multiple battles from a list of keyword overrides."""
    for i, spec in enumerate(specs):
        spec.setdefault(
            "battle_time", f"2026021{4 + i // 10}T{18 + i % 10:02d}0000.000Z"
        )
        analytics.store_battle(session, make_battle(**spec), "#L90009GPP")


def seed_reporting_db(session: Session) -> None:
    """Populate DB with enough data for all reports."""
    analytics.store_player_snapshot(session, make_player_profile())
    times = [f"2026021{4 + i // 10}T{18 + i % 6:02d}0000.000Z" for i in range(10)]
    for i, t in enumerate(times):
        crowns = 3 if i % 3 != 2 else 1
        opp_crowns = 0 if crowns == 3 else 3
        tc = 30 if crowns > opp_crowns else -30
        analytics.store_battle(
            session,
            make_battle(
                battle_time=t,
                player_crowns=crowns,
                opponent_crowns=opp_crowns,
                trophy_change=tc,
                opponent_name=f"Opp_{i}",
            ),
            "#L90009GPP",
        )
