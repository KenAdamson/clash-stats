"""Tilt detection from battle patterns and TCN embeddings.

Two detection layers:
  1. Heuristic — works immediately after fetch using elixir leak,
     crown differential, and consecutive losses. No replay data needed.
  2. Embedding-based — when TCN embeddings exist, measures distance to
     known tilt cluster centroids in 128-dim space for higher confidence.

Tilt clusters identified from TCN Phase 1 analysis:
  Core (0% WR, 12+ elixir leaked): C5, C10, C11, C12, C13, C14
  Extended (~5% WR, 5+ leaked): C7, C25, C28, C32, C35, C36
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from tracker.models import Battle

logger = logging.getLogger(__name__)

# Heuristic thresholds (calibrated from cluster analysis)
LEAK_SEVERE = 12.0       # Core tilt clusters average 12-20
LEAK_ELEVATED = 6.0      # Extended tilt clusters average 5-7
CONSECUTIVE_LOSS_WARN = 3
CONSECUTIVE_LOSS_SEVERE = 5
LOOKBACK_GAMES = 10      # Check last N games for pattern

# TCN tilt cluster IDs (from cluster profiling)
CORE_TILT_CLUSTERS = {5, 10, 11, 12, 13, 14}
EXTENDED_TILT_CLUSTERS = {7, 25, 28, 32, 35, 36}

# Embedding distance threshold (calibrated: games within this distance
# of a tilt centroid are considered tilt-pattern matches)
TILT_DISTANCE_THRESHOLD = 15.0


@dataclass
class TiltStatus:
    """Result of tilt detection analysis."""

    level: str                    # "none", "warning", "tilting", "severe"
    consecutive_losses: int
    recent_record: str            # e.g. "2W-5L"
    avg_leak_recent: float
    max_leak_recent: float
    tilt_game_count: int          # games in lookback matching tilt pattern
    embedding_matches: int        # games near tilt centroids (0 if no embeddings)
    message: str


def _get_recent_battles(
    session: Session, n: int = LOOKBACK_GAMES
) -> list[dict]:
    """Load the last N PvP ladder battles with relevant fields."""
    rows = session.execute(
        select(
            Battle.battle_id,
            Battle.battle_time,
            Battle.result,
            Battle.player_crowns,
            Battle.opponent_crowns,
            Battle.player_elixir_leaked,
            Battle.crown_differential,
        )
        .where(
            Battle.battle_type == "PvP",
            Battle.result.in_(["win", "loss"]),
        )
        .order_by(Battle.battle_time.desc())
        .limit(n)
    ).all()

    return [
        {
            "battle_id": r[0],
            "battle_time": r[1],
            "result": r[2],
            "player_crowns": r[3],
            "opponent_crowns": r[4],
            "leak": r[5] or 0.0,
            "crown_diff": r[6] or 0,
        }
        for r in rows
    ]


def _count_consecutive_losses(games: list[dict]) -> int:
    """Count consecutive losses from the most recent game backward."""
    count = 0
    for g in games:  # already ordered most-recent-first
        if g["result"] == "loss":
            count += 1
        else:
            break
    return count


def _count_embedding_tilt_matches(
    session: Session, battle_ids: list[str]
) -> int:
    """Count how many of the given battles have TCN embeddings near tilt centroids."""
    try:
        from tracker.ml.storage import GameEmbedding, from_blob
    except ImportError:
        return 0

    rows = session.execute(
        select(GameEmbedding.battle_id, GameEmbedding.cluster_id)
        .where(
            GameEmbedding.battle_id.in_(battle_ids),
            GameEmbedding.model_version == "tcn-v1",
        )
    ).all()

    if not rows:
        return 0

    all_tilt = CORE_TILT_CLUSTERS | EXTENDED_TILT_CLUSTERS
    return sum(1 for _, cid in rows if cid in all_tilt)


def detect_tilt(session: Session) -> TiltStatus:
    """Analyze recent games for tilt patterns.

    Args:
        session: Database session.

    Returns:
        TiltStatus with detection results.
    """
    games = _get_recent_battles(session, LOOKBACK_GAMES)

    if not games:
        return TiltStatus(
            level="none",
            consecutive_losses=0,
            recent_record="0W-0L",
            avg_leak_recent=0.0,
            max_leak_recent=0.0,
            tilt_game_count=0,
            embedding_matches=0,
            message="No recent games to analyze.",
        )

    # Basic stats
    consecutive_losses = _count_consecutive_losses(games)
    wins = sum(1 for g in games if g["result"] == "win")
    losses = len(games) - wins
    recent_record = f"{wins}W-{losses}L"

    leaks = [g["leak"] for g in games]
    avg_leak = np.mean(leaks)
    max_leak = max(leaks)

    # Count heuristic tilt-pattern games:
    # loss + (high leak OR got 3-crowned)
    tilt_games = sum(
        1 for g in games
        if g["result"] == "loss" and (
            g["leak"] >= LEAK_ELEVATED
            or g["opponent_crowns"] == 3
        )
    )

    # Embedding-based detection
    battle_ids = [g["battle_id"] for g in games]
    embedding_matches = _count_embedding_tilt_matches(session, battle_ids)

    # Determine tilt level
    level = "none"
    message = ""

    # Severe: 5+ consecutive losses, or 3+ with high leak
    if consecutive_losses >= CONSECUTIVE_LOSS_SEVERE:
        level = "severe"
        message = (
            f"{consecutive_losses} consecutive losses. "
            f"Avg elixir leaked: {avg_leak:.1f}. Stop playing."
        )
    elif consecutive_losses >= CONSECUTIVE_LOSS_WARN and avg_leak >= LEAK_SEVERE:
        level = "severe"
        message = (
            f"{consecutive_losses} losses in a row, leaking {avg_leak:.1f} avg elixir. "
            f"You're hemorrhaging. Walk away."
        )

    # Tilting: 3+ consecutive losses, or heavy tilt pattern in lookback
    elif consecutive_losses >= CONSECUTIVE_LOSS_WARN:
        level = "tilting"
        message = (
            f"{consecutive_losses} consecutive losses ({recent_record} in last {len(games)}). "
            f"Take a break."
        )
    elif tilt_games >= 4:
        level = "tilting"
        message = (
            f"{tilt_games}/{len(games)} recent games match tilt patterns. "
            f"Avg leak: {avg_leak:.1f}. Consider stopping."
        )

    # Warning: early signs
    elif consecutive_losses >= 2 and avg_leak >= LEAK_ELEVATED:
        level = "warning"
        message = (
            f"{consecutive_losses} losses, elevated leak ({avg_leak:.1f} avg). "
            f"Watch the next game closely."
        )
    elif tilt_games >= 3:
        level = "warning"
        message = (
            f"{tilt_games}/{len(games)} recent games show tilt signals. "
            f"Leak pattern emerging."
        )
    elif embedding_matches >= 3:
        level = "warning"
        message = (
            f"{embedding_matches}/{len(games)} recent games match TCN tilt clusters. "
            f"Pattern detected."
        )

    if level == "none":
        message = f"No tilt detected. {recent_record} in last {len(games)} games."

    return TiltStatus(
        level=level,
        consecutive_losses=consecutive_losses,
        recent_record=recent_record,
        avg_leak_recent=round(avg_leak, 1),
        max_leak_recent=round(max_leak, 1),
        tilt_game_count=tilt_games,
        embedding_matches=embedding_matches,
        message=message,
    )


def print_tilt_warning(status: TiltStatus) -> None:
    """Print tilt status to terminal with appropriate severity formatting."""
    if status.level == "none":
        return

    icons = {
        "warning": "\u26a0\ufe0f ",
        "tilting": "\U0001f534 ",
        "severe":  "\U0001f6d1 ",
    }
    labels = {
        "warning": "TILT WARNING",
        "tilting": "TILTING",
        "severe":  "TILT — STOP PLAYING",
    }

    icon = icons.get(status.level, "")
    label = labels.get(status.level, status.level.upper())

    print(f"\n{icon}{label}")
    print(f"  {status.message}")
    print(f"  Recent: {status.recent_record} | "
          f"Streak: {status.consecutive_losses}L | "
          f"Leak: {status.avg_leak_recent} avg / {status.max_leak_recent} max")
    if status.embedding_matches > 0:
        print(f"  TCN tilt cluster matches: {status.embedding_matches}")
