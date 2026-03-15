"""ORM models for win probability data (ADR-004)."""

from typing import Optional

from sqlalchemy import ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from tracker.models import Base


class WinProbability(Base):
    """Per-tick win probability for a game."""

    __tablename__ = "win_probability"
    __table_args__ = (
        UniqueConstraint("battle_id", "game_tick", "model_version"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    battle_id: Mapped[str] = mapped_column(
        ForeignKey("battles.battle_id"), nullable=False
    )
    game_tick: Mapped[int] = mapped_column(nullable=False)
    win_prob: Mapped[float] = mapped_column(nullable=False)
    wpa: Mapped[Optional[float]]
    criticality: Mapped[Optional[float]]
    event_index: Mapped[Optional[int]]
    model_version: Mapped[str] = mapped_column(String, nullable=False)


class GameWPSummary(Base):
    """Per-game win probability summary statistics."""

    __tablename__ = "game_wp_summary"

    battle_id: Mapped[str] = mapped_column(
        ForeignKey("battles.battle_id"), primary_key=True
    )
    pre_game_wp: Mapped[Optional[float]]
    final_wp: Mapped[Optional[float]]
    max_wp: Mapped[Optional[float]]
    min_wp: Mapped[Optional[float]]
    volatility: Mapped[Optional[float]]
    top_positive_wpa_card: Mapped[Optional[str]]
    top_negative_wpa_card: Mapped[Optional[str]]
    critical_tick: Mapped[Optional[int]]
    critical_card: Mapped[Optional[str]]
    model_version: Mapped[str] = mapped_column(String, nullable=False)
