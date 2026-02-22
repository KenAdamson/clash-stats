"""SQLAlchemy ORM models for the Clash Royale battle tracker."""

from datetime import datetime
from typing import Optional

from sqlalchemy import ForeignKey, Index, String, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


class PlayerSnapshot(Base):
    """Player profile snapshot captured at each fetch interval."""

    __tablename__ = "player_snapshots"

    id: Mapped[int] = mapped_column(primary_key=True)
    timestamp: Mapped[Optional[datetime]] = mapped_column(default=func.now())
    player_tag: Mapped[str] = mapped_column(String, nullable=False)
    name: Mapped[Optional[str]]
    exp_level: Mapped[Optional[int]]
    trophies: Mapped[Optional[int]]
    best_trophies: Mapped[Optional[int]]
    wins: Mapped[Optional[int]]
    losses: Mapped[Optional[int]]
    battle_count: Mapped[Optional[int]]
    three_crown_wins: Mapped[Optional[int]]
    challenge_cards_won: Mapped[Optional[int]]
    challenge_max_wins: Mapped[Optional[int]]
    tournament_battle_count: Mapped[Optional[int]]
    tournament_cards_won: Mapped[Optional[int]]
    war_day_wins: Mapped[Optional[int]]
    total_donations: Mapped[Optional[int]]
    clan_tag: Mapped[Optional[str]]
    clan_name: Mapped[Optional[str]]
    arena_name: Mapped[Optional[str]]
    raw_json: Mapped[Optional[str]]


class Battle(Base):
    """Individual battle record."""

    __tablename__ = "battles"
    __table_args__ = (
        Index("idx_battles_player_tag", "player_tag"),
        Index("idx_battles_battle_time", "battle_time"),
        Index("idx_battles_player_deck_hash", "player_deck_hash"),
        Index("idx_battles_result", "result"),
        Index("idx_battles_battle_type", "battle_type"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    battle_id: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    timestamp: Mapped[Optional[datetime]] = mapped_column(default=func.now())
    battle_time: Mapped[Optional[str]]
    battle_type: Mapped[Optional[str]]
    arena_name: Mapped[Optional[str]]
    game_mode_name: Mapped[Optional[str]]
    is_ladder_tournament: Mapped[Optional[int]]

    # Player (you)
    player_tag: Mapped[str] = mapped_column(String, nullable=False)
    player_name: Mapped[Optional[str]]
    player_starting_trophies: Mapped[Optional[int]]
    player_trophy_change: Mapped[Optional[int]]
    player_crowns: Mapped[Optional[int]]
    player_king_tower_hp: Mapped[Optional[int]]
    player_princess_tower_hp: Mapped[Optional[str]]
    player_deck: Mapped[Optional[str]]
    player_deck_hash: Mapped[Optional[str]]

    # Opponent
    opponent_tag: Mapped[Optional[str]]
    opponent_name: Mapped[Optional[str]]
    opponent_starting_trophies: Mapped[Optional[int]]
    opponent_trophy_change: Mapped[Optional[int]]
    opponent_crowns: Mapped[Optional[int]]
    opponent_king_tower_hp: Mapped[Optional[int]]
    opponent_princess_tower_hp: Mapped[Optional[str]]
    opponent_deck: Mapped[Optional[str]]
    opponent_deck_hash: Mapped[Optional[str]]

    # Derived
    result: Mapped[Optional[str]]
    crown_differential: Mapped[Optional[int]]
    raw_json: Mapped[Optional[str]]

    # Added in migration v1
    player_elixir_leaked: Mapped[Optional[float]]
    opponent_elixir_leaked: Mapped[Optional[float]]
    battle_duration: Mapped[Optional[int]]

    # Added in migration v2
    replay_fetched: Mapped[int] = mapped_column(default=0)

    # Added in migration v3 (ADR-007)
    corpus: Mapped[str] = mapped_column(String, default="personal")

    # Relationship
    deck_cards: Mapped[list["DeckCard"]] = relationship(
        back_populates="battle", cascade="all, delete-orphan"
    )


class DeckCard(Base):
    """Individual card appearance in a battle deck."""

    __tablename__ = "deck_cards"
    __table_args__ = (
        Index("idx_deck_cards_card_name", "card_name"),
        Index("idx_deck_cards_battle_id", "battle_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    battle_id: Mapped[str] = mapped_column(
        ForeignKey("battles.battle_id"), nullable=False
    )
    card_name: Mapped[str] = mapped_column(String, nullable=False)
    card_level: Mapped[Optional[int]]
    card_max_level: Mapped[Optional[int]]
    card_elixir: Mapped[Optional[int]]
    is_player_deck: Mapped[Optional[int]]
    evolution_level: Mapped[int] = mapped_column(default=0)
    star_level: Mapped[int] = mapped_column(default=0)

    # Relationship
    battle: Mapped["Battle"] = relationship(back_populates="deck_cards")


class PlayerCorpus(Base):
    """Tracked players for corpus data collection (ADR-007)."""

    __tablename__ = "player_corpus"

    player_tag: Mapped[str] = mapped_column(String, primary_key=True)
    player_name: Mapped[Optional[str]]
    source: Mapped[str] = mapped_column(String, nullable=False)  # top_ladder, matchup_search, manual
    trophy_range_low: Mapped[Optional[int]]
    trophy_range_high: Mapped[Optional[int]]
    games_scraped: Mapped[int] = mapped_column(default=0)
    replays_scraped: Mapped[int] = mapped_column(default=0)
    last_scraped: Mapped[Optional[datetime]]
    active: Mapped[int] = mapped_column(default=1)


class ReplayEvent(Base):
    """Individual card placement event from a battle replay."""

    __tablename__ = "replay_events"
    __table_args__ = (
        Index("idx_replay_events_battle_id", "battle_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    battle_id: Mapped[str] = mapped_column(
        ForeignKey("battles.battle_id"), nullable=False
    )
    side: Mapped[str] = mapped_column(String, nullable=False)  # "team" or "opponent"
    card_name: Mapped[str] = mapped_column(String, nullable=False)
    game_tick: Mapped[int]
    arena_x: Mapped[int]
    arena_y: Mapped[int]
    play_number: Mapped[int] = mapped_column(default=1)
    ability_used: Mapped[int] = mapped_column(default=0)

    battle: Mapped["Battle"] = relationship()


class ReplaySummary(Base):
    """Per-side elixir and card-type stats from a battle replay."""

    __tablename__ = "replay_summaries"
    __table_args__ = (
        Index("idx_replay_summaries_battle_id", "battle_id"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    battle_id: Mapped[str] = mapped_column(
        ForeignKey("battles.battle_id"), nullable=False
    )
    side: Mapped[str] = mapped_column(String, nullable=False)
    total_plays: Mapped[Optional[int]]
    total_elixir: Mapped[Optional[int]]
    troop_plays: Mapped[Optional[int]]
    troop_elixir: Mapped[Optional[int]]
    spell_plays: Mapped[Optional[int]]
    spell_elixir: Mapped[Optional[int]]
    building_plays: Mapped[Optional[int]]
    building_elixir: Mapped[Optional[int]]
    ability_plays: Mapped[Optional[int]]
    ability_elixir: Mapped[Optional[int]]
    elixir_leaked: Mapped[Optional[float]]

    battle: Mapped["Battle"] = relationship()
