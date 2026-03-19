"""ORM models for counterfactual simulation data (ADR-006)."""

from datetime import datetime
from typing import Optional

from sqlalchemy import ForeignKey, JSON, String, func
from sqlalchemy.orm import Mapped, mapped_column

from tracker.models import Base


class CounterfactualResult(Base):
    """Stored result of a counterfactual deck swap simulation."""

    __tablename__ = "counterfactual_results"

    id: Mapped[int] = mapped_column(primary_key=True)
    battle_id: Mapped[str] = mapped_column(
        ForeignKey("battles.battle_id"), nullable=False
    )
    old_card: Mapped[str] = mapped_column(String, nullable=False)
    new_card: Mapped[str] = mapped_column(String, nullable=False)
    original_wp: Mapped[float] = mapped_column(nullable=False)
    counterfactual_wp_mean: Mapped[float] = mapped_column(nullable=False)
    counterfactual_wp_std: Mapped[float] = mapped_column(nullable=False)
    delta_wp: Mapped[float] = mapped_column(nullable=False)
    n_samples: Mapped[int] = mapped_column(nullable=False)
    model_version: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[Optional[datetime]] = mapped_column(default=func.now())
    raw_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)


class DeckGradientResult(Base):
    """Aggregated deck gradient: expected WR delta for a card swap."""

    __tablename__ = "deck_gradient_results"

    id: Mapped[int] = mapped_column(primary_key=True)
    old_card: Mapped[str] = mapped_column(String, nullable=False)
    new_card: Mapped[str] = mapped_column(String, nullable=False)
    mean_delta_wp: Mapped[float] = mapped_column(nullable=False)
    ci_low: Mapped[float] = mapped_column(nullable=False)
    ci_high: Mapped[float] = mapped_column(nullable=False)
    n_games: Mapped[int] = mapped_column(nullable=False)
    model_version: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[Optional[datetime]] = mapped_column(default=func.now())
