"""Tests for score-then-limit polling selection with exploration floor."""

from unittest.mock import patch

from tracker.corpus import get_corpus_players
from tracker.models import PlayerCorpus


def _pool(session, n=20):
    """n active players; #P00 is priority, rest network."""
    for i in range(n):
        session.add(PlayerCorpus(
            player_tag=f"#P{i:02d}", active=1,
            source="priority" if i == 0 else "network",
        ))
    session.commit()


def test_score_then_limit_membership(session):
    """High-scored players enter the batch even from deep FIFO positions."""
    _pool(session)
    # score players in REVERSE of tag order so FIFO and score disagree
    fake_scores = [(f"#P{i:02d}", i / 100.0) for i in range(20)]
    with patch("tracker.ml.activity_model.score_corpus_players",
               return_value=fake_scores):
        players = get_corpus_players(session, limit=10, prioritize_active=True)

    tags = [p.player_tag for p in players]
    assert len(tags) == 10
    assert tags[0] == "#P00"                      # priority leads
    assert "#P19" in tags and "#P18" in tags      # top-scored made the cut
    # exploration floor: at least one low-score FIFO pick present
    n_explore = max(1, int(10 * 0.2))
    scored_set = {f"#P{i:02d}" for i in range(13, 20)}  # top 7 scored
    assert len([t for t in tags if t not in scored_set and t != "#P00"]) >= n_explore


def test_fifo_fallback_when_model_missing(session):
    """Scoring failure degrades to FIFO with the limit still applied."""
    _pool(session)
    with patch("tracker.ml.activity_model.score_corpus_players",
               side_effect=RuntimeError("no model")):
        players = get_corpus_players(session, limit=5, prioritize_active=True)
    assert len(players) == 5
    assert players[0].player_tag == "#P00"  # priority-first survives


def test_plain_fifo_unchanged(session):
    """Without prioritize_active the SQL limit path is untouched."""
    _pool(session)
    players = get_corpus_players(session, limit=5)
    assert len(players) == 5
    assert players[0].player_tag == "#P00"
