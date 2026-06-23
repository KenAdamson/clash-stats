"""Tests for the deck-invariant pilot fingerprint (smurf pillar 3)."""

import pytest

from tracker.models import Battle, DeckCard, PlayerDim, ReplayEvent, PilotFingerprint
from tracker.ml import pilot_fingerprint as pf


# ---------------------------------------------------------------------------
# Pure feature math
# ---------------------------------------------------------------------------

def _game(team, opp_ticks):
    """team: list of (tick, card, arena_y); opp_ticks: list of opponent play ticks."""
    evs = [("opponent", t, "x", 20000) for t in opp_ticks]
    evs += [("team", t, c, ay) for (t, c, ay) in team]
    return evs


def test_slug_bridges_display_to_replay():
    assert pf.slug("Baby Dragon") == "baby-dragon"
    assert pf.slug("P.E.K.K.A") == "pekka"
    assert pf.slug("Mini P.E.K.K.A") == "mini-pekka"


def test_compute_fingerprint_deterministic():
    costs = {"a": 4}
    # 10 identical games: team plays every 100 ticks (gap/cost = 100/4 = 25),
    # each preceded by an opp play 10 ticks earlier (reaction 10, fast+defensive).
    games = []
    for _ in range(10):
        team = [(t, "a", 10000) for t in (100, 200, 300, 400, 500)]
        opp = [90, 190, 290, 390, 490]
        games.append(_game(team, opp))

    fp = pf.compute_fingerprint(games, costs)
    assert fp is not None
    assert fp["elixir_pace"] == pytest.approx(25.0)
    assert fp["reaction"] == pytest.approx(10.0)
    assert fp["def_reaction"] == pytest.approx(10.0)       # all answers in own half
    assert fp["fast_react_frac"] == pytest.approx(1.0)     # all <= 50 ticks
    assert fp["pace_consistency"] == pytest.approx(0.0)     # perfectly regular
    assert fp["throughput"] == pytest.approx(50.0)          # 20 elixir / 400 span * 1000
    assert fp["n_games"] == 10


def test_compute_fingerprint_thin_returns_none():
    costs = {"a": 4}
    assert pf.compute_fingerprint([_game([(100, "a", 10000)], [90])], costs) is None


# ---------------------------------------------------------------------------
# Integration: refresh + behavioral match over a seeded SQLite db
# ---------------------------------------------------------------------------

def _seed_player(session, tag, n_games, trophies, *, gap=100, react=10, ay=10000):
    """Seed n_games battles (replay_fetched) + their team/opp replay events."""
    session.add(DeckCard(battle_id="seed", card_name="a",
                         card_level=14, card_max_level=14, card_elixir=4))
    for g in range(n_games):
        bid = f"{tag}-{g}"
        session.add(Battle(
            battle_id=bid, player_tag=tag, corpus="top_ladder",
            replay_fetched=1, player_starting_trophies=trophies,
            battle_time=None, opponent_tag=None,
        ))
        for k, base in enumerate((100, 200, 300, 400, 500)):
            session.add(ReplayEvent(battle_id=bid, side="opponent",
                                    card_name="x", game_tick=base - react,
                                    arena_x=8000, arena_y=20000))
            session.add(ReplayEvent(battle_id=bid, side="team",
                                    card_name="a", game_tick=base,
                                    arena_x=8000, arena_y=ay))
    session.commit()


def test_refresh_and_behavioral_match(session, monkeypatch):
    monkeypatch.setattr(pf, "MIN_GAMES", 8)   # keep the seed light

    # DeckCard needs a real battle row for its FK-free insert; seed players first.
    _seed_player(session, "#TWIN_A", 10, trophies=12000)
    _seed_player(session, "#TWIN_B", 10, trophies=12000)
    # a behaviorally different pilot: slow, non-defensive, irregular
    _seed_player(session, "#OTHER", 10, trophies=3000, gap=100, react=200, ay=20000)

    written = pf.refresh_pilot_fingerprints(session, batch=100)
    assert written == 3
    assert session.get(PilotFingerprint, "#TWIN_A") is not None

    # player_dim row for a low-trophy account that behaves like the 12k twins
    session.add(PlayerDim(player_tag="#TWIN_A", latest_trophies=2500, games=10))
    session.commit()

    matched = pf.compute_behavioral_match(session, k=2)
    assert matched >= 1
    pd = session.get(PlayerDim, "#TWIN_A")
    # nearest pilot is TWIN_B (12k); behavioral_gap = ~12000 - 2500 > 0 (skill-smurf)
    assert pd.behavioral_neighbor_trophy is not None
    assert pd.behavioral_gap >= 5000   # plays like a much-higher-trophy pilot

    neighbors = pf.nearest_pilots(session, "#TWIN_A", k=2)
    assert neighbors[0]["player_tag"] == "#TWIN_B"
