"""Tests for corpus_hygiene bot detection — efficiency rule + badge corroborator."""

import pickle

import pytest

from tracker.corpus import corpus_hygiene
from tracker.models import PlayerCorpus


class FakeAPI:
    """Returns canned /players profiles keyed by tag."""

    def __init__(self, profiles):
        self.profiles = profiles

    def get_player(self, tag):
        return self.profiles[tag]


def _profile(bc, best, clan=None, badges=None):
    p = {"battleCount": bc, "bestTrophies": best, "badges": badges or []}
    if clan:
        p["clan"] = {"tag": clan}
    return p


def _rich_badges(n_masteries=100, event=True, years_played=True):
    """A human-shaped badge sheet."""
    badges = [{"name": f"MasteryCard{i}"} for i in range(n_masteries)]
    badges += [{"name": n} for n in ("BattleWins", "CollectionLevel", "ClanDonations")]
    if years_played:
        badges.append({"name": "YearsPlayed"})
    if event:
        badges += [{"name": n} for n in ("GoblinJourney2024", "2v2", "Draft")]
    return badges


@pytest.fixture
def corpus_session(session, tmp_path):
    """Session pre-loaded with one account per detection scenario."""
    accounts = {
        # strict-rule bot: huge bc, eff 0.22, clanless
        "#BOT1": _profile(50000, 11000),
        # gray-zone + skeletal badge sheet -> badge rule fires
        "#GRAYBOT": _profile(9000, 4500, badges=[{"name": "BattleWins"}] * 1),
        # gray-zone + no YearsPlayed + no event badges -> badge rule fires
        "#YOUNGBOT": _profile(8000, 4400, badges=_rich_badges(event=False, years_played=False)),
        # gray-zone but human-shaped badges -> spared
        "#GRINDER": _profile(9000, 4500, badges=_rich_badges()),
        # gray-zone, skeletal, but IN A CLAN -> spared
        "#CLANNED": _profile(9000, 4500, clan="#C1", badges=[]),
        # efficient human -> spared by both rules
        "#HUMAN": _profile(3000, 6300, badges=_rich_badges()),
    }
    for tag in accounts:
        session.add(PlayerCorpus(player_tag=tag, active=1, source="top_ladder"))
    session.commit()
    return session, FakeAPI(accounts), str(tmp_path / "enrich.pkl")


def test_strict_and_badge_rules(corpus_session):
    session, api, cache_path = corpus_session
    corpus_hygiene(session, api, dormant_days=99999, min_trophy=0, cache_path=cache_path)

    flagged = {
        r.player_tag: (r.active, r.source)
        for r in session.query(PlayerCorpus).all()
    }
    assert flagged["#BOT1"] == (0, "bot")        # strict efficiency rule
    assert flagged["#GRAYBOT"] == (0, "bot")     # skeletal badge sheet
    assert flagged["#YOUNGBOT"] == (0, "bot")    # no YearsPlayed + no events
    assert flagged["#GRINDER"][0] == 1           # human badges spare gray zone
    assert flagged["#CLANNED"][0] == 1           # clan gate spares
    assert flagged["#HUMAN"][0] == 1


def test_badge_fields_cached(corpus_session):
    session, api, cache_path = corpus_session
    corpus_hygiene(session, api, dormant_days=99999, min_trophy=0, cache_path=cache_path)

    with open(cache_path, "rb") as f:
        cache = pickle.load(f)
    v = cache["#GRINDER"]
    assert v["n_badges"] == len(_rich_badges())
    assert v["has_years_played"] is True
    assert v["n_event_badges"] == 3  # GoblinJourney2024, 2v2, Draft


def test_stale_entries_self_heal_via_reenrichment(corpus_session):
    """Pre-badge-era cache entries are re-enriched and then judged."""
    session, api, cache_path = corpus_session
    with open(cache_path, "wb") as f:
        pickle.dump({"#GRAYBOT": {"bc": 9000, "best": 4500, "clan": None}}, f)

    corpus_hygiene(session, api, dormant_days=99999, min_trophy=0, cache_path=cache_path)
    row = session.query(PlayerCorpus).filter_by(player_tag="#GRAYBOT").one()
    assert row.active == 0  # re-enriched -> skeletal badges -> flagged
    with open(cache_path, "rb") as f:
        assert "n_badges" in pickle.load(f)["#GRAYBOT"]


def test_reenrichment_failure_keeps_stale_entry_and_spares(corpus_session):
    """If re-enrichment fails, the stale entry survives and the badge rule
    stays silent (no n_badges -> unknown -> spared)."""
    session, api, cache_path = corpus_session
    stale = {"bc": 9000, "best": 4500, "clan": None}
    with open(cache_path, "wb") as f:
        pickle.dump({"#GRAYBOT": dict(stale)}, f)
    del api.profiles["#GRAYBOT"]  # fetch now raises

    corpus_hygiene(session, api, dormant_days=99999, min_trophy=0, cache_path=cache_path)
    row = session.query(PlayerCorpus).filter_by(player_tag="#GRAYBOT").one()
    assert row.active == 1
    with open(cache_path, "rb") as f:
        assert pickle.load(f)["#GRAYBOT"] == stale  # not clobbered to None
