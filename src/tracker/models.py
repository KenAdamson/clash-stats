"""SQLAlchemy ORM models for the Clash Royale battle tracker."""

from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Index, JSON, String, func
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


# Unified in-game card-level cap. The CR API reports card_level on a rarity-
# normalized scale; the displayed (in-game) level adds a per-rarity offset
# encoded in card_max_level: displayed = card_level + (CAP - card_max_level).
DISPLAYED_LEVEL_CAP = 16


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
    raw_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)


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
    battle_time: Mapped[Optional[datetime]]
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
    raw_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # Added in migration v1
    player_elixir_leaked: Mapped[Optional[float]]
    opponent_elixir_leaked: Mapped[Optional[float]]
    battle_duration: Mapped[Optional[int]]

    # Added in migration v2
    replay_fetched: Mapped[int] = mapped_column(default=0)

    # Added in migration v3 (ADR-007)
    corpus: Mapped[str] = mapped_column(String, default="personal")

    # Added in migration v4 (ADR-009)
    video_path: Mapped[Optional[str]] = mapped_column(String, default=None)

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
    # evolution_level encodes the card's fielded form, NOT a stat tier:
    #   0 = base, 1 = evolved, 2 = hero, 3 = hero+evolved (vanishingly rare).
    # (Discovered 2026-06-13 from a known hero-wizard game: hero status is
    # smuggled into this field rather than exposed separately by the API.)
    evolution_level: Mapped[int] = mapped_column(default=0)
    star_level: Mapped[int] = mapped_column(default=0)  # cosmetic Star Points, 0-3
    card_variant: Mapped[str] = mapped_column(String, default="base")  # base, evo, hero

    # Relationship
    battle: Mapped["Battle"] = relationship(back_populates="deck_cards")

    @hybrid_property
    def is_evo(self) -> bool:
        """True if this card was fielded in its evolved form (incl. hero+evo)."""
        return self.evolution_level in (1, 3)

    @is_evo.inplace.expression
    @classmethod
    def _is_evo_expr(cls):
        return cls.evolution_level.in_((1, 3))

    @hybrid_property
    def is_hero(self) -> bool:
        """True if this card was fielded as a hero (incl. hero+evo)."""
        return self.evolution_level in (2, 3)

    @is_hero.inplace.expression
    @classmethod
    def _is_hero_expr(cls):
        return cls.evolution_level.in_((2, 3))

    @hybrid_property
    def displayed_level(self) -> Optional[int]:
        """In-game card level the player actually sees.

        ``card_level`` stores the CR API's RARITY-NORMALIZED level, not the
        displayed number. The offset to the unified in-game scale (cap 16) is
        encoded in ``card_max_level`` — common +0 (max 16), rare +2 (max 14),
        epic +5 (max 11), legendary +8 (max 8), champion +10 (max 6):
        ``displayed = card_level + (16 - card_max_level)``. Verified against
        ground truth (rare Fireball api-8 → 10, epic Bowler api-7 → 12).
        Returns None if either input is unknown.
        """
        if self.card_level is None or self.card_max_level is None:
            return None
        return self.card_level + (DISPLAYED_LEVEL_CAP - self.card_max_level)

    @displayed_level.inplace.expression
    @classmethod
    def _displayed_level_expr(cls):
        return cls.card_level + (DISPLAYED_LEVEL_CAP - cls.card_max_level)


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


class CorpusHourlyStat(Base):
    """Pre-aggregated corpus battle count by hour of day (UTC)."""

    __tablename__ = "corpus_hourly_stats"

    hour: Mapped[int] = mapped_column(primary_key=True)
    battle_count: Mapped[int] = mapped_column(default=0)


class ClanDim(Base):
    """Derived clan dimension, split into a cheap IDENTITY half and an
    API-enriched MEASURES half.

    IDENTITY (clan_tag, clan_name, first/last_seen, n_battles_seen,
    on_our_accounts) is harvested for FREE from battle ``raw_json`` at no API
    cost — :func:`tracker.dimensions.harvest_clan_dim` upserts a row for every
    clan ever seen as an opponent (~490K). MEASURES (member/trophy aggregates)
    require a ``/clans/%23TAG`` call and are filled lazily by
    :func:`tracker.dimensions.resolve_clan_dim`, which drains the backlog in
    priority-ordered batches (our own accounts' clans first). ``resolved_at`` is
    NULL until a clan's measures have been fetched. All derived, no source of
    truth. Migration 006 created it; 007 added the identity/resolution columns.
    """

    __tablename__ = "clan_dim"
    __table_args__ = (
        Index("idx_clan_dim_resolved_at", "resolved_at"),
        Index("idx_clan_dim_on_our_accounts", "on_our_accounts"),
    )

    clan_tag: Mapped[str] = mapped_column(String, primary_key=True)
    clan_name: Mapped[Optional[str]]

    # --- IDENTITY (harvested free from battles, no API) ---
    first_seen: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    last_seen: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    n_battles_seen: Mapped[Optional[int]]      # battles we've seen this clan in
    on_our_accounts: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False
    )  # ever an opponent's clan on personal/alt — resolver priority

    # --- MEASURES (API-enriched, NULL until resolved) ---
    member_count: Mapped[Optional[int]]
    max_trophies: Mapped[Optional[int]]
    avg_trophies: Mapped[Optional[int]]
    median_trophies: Mapped[Optional[int]]
    n_9k: Mapped[Optional[int]]   # members at >= 9000 trophies
    n_11k: Mapped[Optional[int]]  # members at >= 11000 trophies
    n_12k: Mapped[Optional[int]]  # members at >= 12000 trophies
    top_member_name: Mapped[Optional[str]]
    top_member_tag: Mapped[Optional[str]]
    top_member_trophies: Mapped[Optional[int]]

    # --- RESOLUTION tracking ---
    resolved_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True)
    )  # NULL = measures never fetched
    resolve_attempts: Mapped[int] = mapped_column(default=0)
    refreshed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), default=func.now()
    )  # last time this row (identity or measures) was touched


class PlayerDim(Base):
    """Derived opponent/player dimension — one row per opponent seen in battles.

    Aggregated from the ``battles`` table (opponent_tag/name/raw_json) by
    :func:`tracker.dimensions.refresh_player_dim`, enriched with ``clan_tag``
    pulled from each opponent's battle ``raw_json``. Like :class:`ClanDim`, it
    is fully derived data — repopulatable at any time, never a source of truth.
    """

    __tablename__ = "player_dim"
    __table_args__ = (
        Index("idx_player_dim_clan_tag", "clan_tag"),
        Index("idx_player_dim_last_seen", "last_seen"),
        Index("idx_player_dim_alt_suspect", "is_alt_suspect"),
    )

    player_tag: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[Optional[str]]
    latest_trophies: Mapped[Optional[int]]
    exp_level: Mapped[Optional[int]]
    # Nullable soft-FK to clan_dim.clan_tag. Not a hard FK: a player's clan may
    # not be present in clan_dim (clan_dim only covers clans we've refreshed),
    # and we never want a missing clan to block a player upsert.
    clan_tag: Mapped[Optional[str]]
    first_seen: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    last_seen: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    games: Mapped[int] = mapped_column(default=0)
    wins: Mapped[int] = mapped_column(default=0)
    losses: Mapped[int] = mapped_column(default=0)
    last_deck_hash: Mapped[Optional[str]]
    is_alt_suspect: Mapped[bool] = mapped_column(Boolean, default=False)

    # --- Smurf pillar 2/3: levels-implied-trophy gap (funded-smurf detector) ---
    # deck_top_level = the player's max displayed card level (from their latest
    # deck). implied_trophy_gap = where a deck of that top-level NORMALLY lives
    # (level_trophy_ref.median_trophy) minus latest_trophies. A large POSITIVE
    # gap means their card investment belongs far above their placement — the
    # pay-to-win / funded-smurf fingerprint (orthogonal to clan-shelter and
    # skill). ~0 for a level-appropriate account.
    deck_top_level: Mapped[Optional[int]]
    implied_trophy_gap: Mapped[Optional[int]] = mapped_column(index=True)

    # --- Smurf pillar 3: behavioral (pilot-fingerprint) skill match ---
    # Computed by tracker.ml.pilot_fingerprint.compute_behavioral_match from the
    # deck-invariant fingerprint (see PilotFingerprint). behavioral_neighbor_trophy
    # = median trophies of this account's k nearest pilots in fingerprint space;
    # behavioral_gap = that minus latest_trophies. A large POSITIVE gap = "plays
    # like a much-higher-trophy pilot" = the SKILL-smurf fingerprint (orthogonal
    # to clan-shelter and the funded/level gap — the user's own alt is this case).
    behavioral_neighbor_trophy: Mapped[Optional[int]]
    behavioral_gap: Mapped[Optional[int]] = mapped_column(index=True)
    refreshed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), default=func.now()
    )


class PilotFingerprint(Base):
    """Deck-INVARIANT behavioral fingerprint of a pilot (smurf pillar 3).

    Six timing/economy/discipline features extracted from a player's replay
    placements (``replay_events``), normalized so they capture the PILOT not the
    deck: tempo is divided by per-card elixir (banking discipline), spatial
    features (lane/aggression) are excluded because they encode deck ROLE.
    Validated 2026-06-22: the user's main and alt — zero shared cards, ~9000
    trophy gap — land as #2/405 nearest neighbors here (self-consistency AUC
    0.83). The match score is computed with plain z-Euclidean; do NOT whiten
    (Mahalanobis decorrelates away the correlated timing signal that IS the
    fingerprint). Fully derived/refreshable; one row per pilot with enough
    replay'd games, filled incrementally by
    :func:`tracker.ml.pilot_fingerprint.refresh_pilot_fingerprints`.
    """

    __tablename__ = "pilot_fingerprint"

    player_tag: Mapped[str] = mapped_column(String, primary_key=True)
    elixir_pace: Mapped[Optional[float]] = mapped_column(Float)
    throughput: Mapped[Optional[float]] = mapped_column(Float)
    reaction: Mapped[Optional[float]] = mapped_column(Float)
    pace_consistency: Mapped[Optional[float]] = mapped_column(Float)
    def_reaction: Mapped[Optional[float]] = mapped_column(Float)
    fast_react_frac: Mapped[Optional[float]] = mapped_column(Float)
    n_games: Mapped[int] = mapped_column(default=0)
    latest_trophies: Mapped[Optional[int]]
    refreshed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), default=func.now()
    )


class LevelTrophyRef(Base):
    """Empirical reference: where a deck of a given top displayed-level normally
    sits on the trophy ladder. Derived from the whole battles corpus by
    :func:`tracker.dimensions.refresh_level_trophy_ref` (median + p10 of
    opponent_starting_trophies grouped by deck-top displayed level). Used to
    compute ``PlayerDim.implied_trophy_gap``. Fully derived/refreshable.
    """

    __tablename__ = "level_trophy_ref"

    deck_top_level: Mapped[int] = mapped_column(primary_key=True)
    median_trophy: Mapped[Optional[int]]
    p10_trophy: Mapped[Optional[int]]
    n_samples: Mapped[Optional[int]]
    refreshed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), default=func.now()
    )


class PlayerKing(Base):
    """Persistent cache of player king (experience) level + best trophies.

    King level is NOT in the battle log — only ``/players/{tag}`` exposes it —
    so it's resolved by API in priority batches (:func:`tracker.dimensions.
    resolve_player_king`) and cached here. Kept in its own table (not on
    player_dim) because player_dim is rebuilt every refresh, while king levels
    are stable and expensive to re-fetch. King level disambiguates the smurf
    species: a LOW king level with cards far over the bracket = PAID whale
    (impossible F2P); a HIGH king level at low trophies = MATURE account
    tanking; best_trophies >> current also flags deliberate tanking.
    """

    __tablename__ = "player_king"

    player_tag: Mapped[str] = mapped_column(String, primary_key=True)
    king_level: Mapped[Optional[int]]
    best_trophies: Mapped[Optional[int]]
    resolved_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    resolve_attempts: Mapped[int] = mapped_column(default=0)
    refreshed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), default=func.now()
    )


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
