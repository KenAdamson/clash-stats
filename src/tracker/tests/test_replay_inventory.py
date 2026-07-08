"""Tests for inventory-driven replay-job player selection."""

from datetime import datetime, timedelta, timezone

from tracker.corpus_scraper import players_with_replay_inventory
from tracker.models import Battle, PlayerCorpus


def _battle(i, tag, hours_ago, fetched=0, corpus="top_ladder", btype="PvP"):
    return Battle(
        battle_id=f"b{i:04d}",
        battle_time=datetime.now(timezone.utc).replace(tzinfo=None)
        - timedelta(hours=hours_ago),
        battle_type=btype,
        result="win",
        player_tag=tag,
        corpus=corpus,
        replay_fetched=fetched,
    )


def test_inventory_selection_orders_by_unfetched_count(session):
    session.add_all([
        PlayerCorpus(player_tag="#BIG", active=1, source="network"),
        PlayerCorpus(player_tag="#SMALL", active=1, source="network"),
        PlayerCorpus(player_tag="#INACTIVE", active=0, source="network"),
        PlayerCorpus(player_tag="#PRIO_DEAD", active=1, source="priority"),
    ])
    i = 0
    for _ in range(5):
        i += 1
        session.add(_battle(i, "#BIG", hours_ago=5))
    session.add(_battle(i + 1, "#SMALL", hours_ago=5))
    # inactive player's battles don't count
    session.add(_battle(i + 2, "#INACTIVE", hours_ago=5))
    # too old (outside window)
    session.add(_battle(i + 3, "#SMALL", hours_ago=90))
    # already fetched
    session.add(_battle(i + 4, "#SMALL", hours_ago=5, fetched=1))
    # wrong corpus / type excluded
    session.add(_battle(i + 5, "#SMALL", hours_ago=5, corpus="alt"))
    session.add(_battle(i + 6, "#SMALL", hours_ago=5, btype="pathOfLegend"))
    session.commit()

    players = players_with_replay_inventory(session, limit=8)
    tags = [p.player_tag for p in players]
    # battle-less priority account can never occupy a slot
    assert "#PRIO_DEAD" not in tags
    assert "#INACTIVE" not in tags
    assert tags == ["#BIG", "#SMALL"]  # ordered by inventory size


def test_inventory_empty_when_caught_up(session):
    session.add(PlayerCorpus(player_tag="#DONE", active=1, source="network"))
    session.add(_battle(1, "#DONE", hours_ago=5, fetched=1))
    session.commit()
    assert players_with_replay_inventory(session, limit=8) == []
