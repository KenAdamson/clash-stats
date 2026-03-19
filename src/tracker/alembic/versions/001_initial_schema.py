"""Consolidated initial schema for PostgreSQL.

Captures the full schema from 11 incremental migrations that evolved the
SQLite → MariaDB → PostgreSQL lineage. pgloader handles the actual data
migration; this migration exists so Alembic has a baseline to stamp.

Revision ID: 001
Revises: (none)
Create Date: 2026-03-18
"""

from alembic import op
import sqlalchemy as sa

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # -- player_snapshots --
    op.create_table(
        "player_snapshots",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("timestamp", sa.DateTime, server_default=sa.func.now()),
        sa.Column("player_tag", sa.String, nullable=False),
        sa.Column("name", sa.String),
        sa.Column("exp_level", sa.Integer),
        sa.Column("trophies", sa.Integer),
        sa.Column("best_trophies", sa.Integer),
        sa.Column("wins", sa.Integer),
        sa.Column("losses", sa.Integer),
        sa.Column("battle_count", sa.Integer),
        sa.Column("three_crown_wins", sa.Integer),
        sa.Column("challenge_cards_won", sa.Integer),
        sa.Column("challenge_max_wins", sa.Integer),
        sa.Column("tournament_battle_count", sa.Integer),
        sa.Column("tournament_cards_won", sa.Integer),
        sa.Column("war_day_wins", sa.Integer),
        sa.Column("total_donations", sa.Integer),
        sa.Column("clan_tag", sa.String),
        sa.Column("clan_name", sa.String),
        sa.Column("arena_name", sa.String),
        sa.Column("raw_json", sa.JSON, nullable=True),
    )

    # -- battles --
    op.create_table(
        "battles",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("battle_id", sa.String, unique=True, nullable=False),
        sa.Column("timestamp", sa.DateTime, server_default=sa.func.now()),
        sa.Column("battle_time", sa.DateTime),
        sa.Column("battle_type", sa.String),
        sa.Column("arena_name", sa.String),
        sa.Column("game_mode_name", sa.String),
        sa.Column("is_ladder_tournament", sa.Integer),
        sa.Column("player_tag", sa.String, nullable=False),
        sa.Column("player_name", sa.String),
        sa.Column("player_starting_trophies", sa.Integer),
        sa.Column("player_trophy_change", sa.Integer),
        sa.Column("player_crowns", sa.Integer),
        sa.Column("player_king_tower_hp", sa.Integer),
        sa.Column("player_princess_tower_hp", sa.String),
        sa.Column("player_deck", sa.String),
        sa.Column("player_deck_hash", sa.String),
        sa.Column("opponent_tag", sa.String),
        sa.Column("opponent_name", sa.String),
        sa.Column("opponent_starting_trophies", sa.Integer),
        sa.Column("opponent_trophy_change", sa.Integer),
        sa.Column("opponent_crowns", sa.Integer),
        sa.Column("opponent_king_tower_hp", sa.Integer),
        sa.Column("opponent_princess_tower_hp", sa.String),
        sa.Column("opponent_deck", sa.String),
        sa.Column("opponent_deck_hash", sa.String),
        sa.Column("result", sa.String),
        sa.Column("crown_differential", sa.Integer),
        sa.Column("raw_json", sa.JSON, nullable=True),
        sa.Column("player_elixir_leaked", sa.Float),
        sa.Column("opponent_elixir_leaked", sa.Float),
        sa.Column("battle_duration", sa.Integer),
        sa.Column("replay_fetched", sa.Integer, server_default="0"),
        sa.Column("corpus", sa.String, server_default="personal"),
        sa.Column("video_path", sa.String),
    )
    op.create_index("idx_battles_player_tag", "battles", ["player_tag"])
    op.create_index("idx_battles_battle_time", "battles", ["battle_time"])
    op.create_index("idx_battles_player_deck_hash", "battles", ["player_deck_hash"])
    op.create_index("idx_battles_result", "battles", ["result"])
    op.create_index("idx_battles_battle_type", "battles", ["battle_type"])
    op.create_index(
        "idx_battles_replay_lookup", "battles",
        ["player_tag", "replay_fetched", "battle_type"],
    )

    # -- deck_cards --
    op.create_table(
        "deck_cards",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column(
            "battle_id", sa.String,
            sa.ForeignKey("battles.battle_id"), nullable=False,
        ),
        sa.Column("card_name", sa.String, nullable=False),
        sa.Column("card_level", sa.Integer),
        sa.Column("card_max_level", sa.Integer),
        sa.Column("card_elixir", sa.Integer),
        sa.Column("is_player_deck", sa.Integer),
        sa.Column("evolution_level", sa.Integer, server_default="0"),
        sa.Column("star_level", sa.Integer, server_default="0"),
        sa.Column("card_variant", sa.String, server_default="'base'"),
    )
    op.create_index("idx_deck_cards_card_name", "deck_cards", ["card_name"])
    op.create_index("idx_deck_cards_battle_id", "deck_cards", ["battle_id"])

    # -- player_corpus --
    op.create_table(
        "player_corpus",
        sa.Column("player_tag", sa.String, primary_key=True),
        sa.Column("player_name", sa.String),
        sa.Column("source", sa.String, nullable=False),
        sa.Column("trophy_range_low", sa.Integer),
        sa.Column("trophy_range_high", sa.Integer),
        sa.Column("games_scraped", sa.Integer, server_default="0"),
        sa.Column("replays_scraped", sa.Integer, server_default="0"),
        sa.Column("last_scraped", sa.DateTime),
        sa.Column("active", sa.Integer, server_default="1"),
    )

    # -- replay_events --
    op.create_table(
        "replay_events",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column(
            "battle_id", sa.String,
            sa.ForeignKey("battles.battle_id"), nullable=False,
        ),
        sa.Column("side", sa.String, nullable=False),
        sa.Column("card_name", sa.String, nullable=False),
        sa.Column("game_tick", sa.Integer),
        sa.Column("arena_x", sa.Integer),
        sa.Column("arena_y", sa.Integer),
        sa.Column("play_number", sa.Integer, server_default="1"),
        sa.Column("ability_used", sa.Integer, server_default="0"),
    )
    op.create_index("idx_replay_events_battle_id", "replay_events", ["battle_id"])

    # -- replay_summaries --
    op.create_table(
        "replay_summaries",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column(
            "battle_id", sa.String,
            sa.ForeignKey("battles.battle_id"), nullable=False,
        ),
        sa.Column("side", sa.String, nullable=False),
        sa.Column("total_plays", sa.Integer),
        sa.Column("total_elixir", sa.Integer),
        sa.Column("troop_plays", sa.Integer),
        sa.Column("troop_elixir", sa.Integer),
        sa.Column("spell_plays", sa.Integer),
        sa.Column("spell_elixir", sa.Integer),
        sa.Column("building_plays", sa.Integer),
        sa.Column("building_elixir", sa.Integer),
        sa.Column("ability_plays", sa.Integer),
        sa.Column("ability_elixir", sa.Integer),
        sa.Column("elixir_leaked", sa.Float),
    )
    op.create_index("idx_replay_summaries_battle_id", "replay_summaries", ["battle_id"])

    # -- game_features (ADR-003) --
    op.create_table(
        "game_features",
        sa.Column(
            "battle_id", sa.String,
            sa.ForeignKey("battles.battle_id"), primary_key=True,
        ),
        sa.Column("feature_vector", sa.LargeBinary),
        sa.Column("feature_version", sa.String, server_default="'v1'"),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )

    # -- game_embeddings (ADR-003) --
    op.create_table(
        "game_embeddings",
        sa.Column(
            "battle_id", sa.String,
            sa.ForeignKey("battles.battle_id"), primary_key=True,
        ),
        sa.Column("embedding_15d", sa.LargeBinary),
        sa.Column("embedding_3d", sa.LargeBinary),
        sa.Column("cluster_id", sa.Integer),
        sa.Column("model_version", sa.String, server_default="'umap-v1'"),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )

    # -- win_probability (ADR-004) --
    op.create_table(
        "win_probability",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column(
            "battle_id", sa.String,
            sa.ForeignKey("battles.battle_id"), nullable=False,
        ),
        sa.Column("game_tick", sa.Integer, nullable=False),
        sa.Column("win_prob", sa.Float, nullable=False),
        sa.Column("wpa", sa.Float),
        sa.Column("criticality", sa.Float),
        sa.Column("event_index", sa.Integer),
        sa.Column("model_version", sa.String, nullable=False),
        sa.UniqueConstraint("battle_id", "game_tick", "model_version"),
    )
    op.create_index("idx_wp_battle", "win_probability", ["battle_id"])
    op.create_index("idx_wp_criticality", "win_probability", ["criticality"])

    # -- game_wp_summary (ADR-004) --
    op.create_table(
        "game_wp_summary",
        sa.Column(
            "battle_id", sa.String,
            sa.ForeignKey("battles.battle_id"), primary_key=True,
        ),
        sa.Column("pre_game_wp", sa.Float),
        sa.Column("final_wp", sa.Float),
        sa.Column("max_wp", sa.Float),
        sa.Column("min_wp", sa.Float),
        sa.Column("volatility", sa.Float),
        sa.Column("top_positive_wpa_card", sa.String),
        sa.Column("top_negative_wpa_card", sa.String),
        sa.Column("critical_tick", sa.Integer),
        sa.Column("critical_card", sa.String),
        sa.Column("model_version", sa.String, nullable=False),
    )


def downgrade() -> None:
    op.drop_table("game_wp_summary")
    op.drop_table("win_probability")
    op.drop_table("game_embeddings")
    op.drop_table("game_features")
    op.drop_table("replay_summaries")
    op.drop_table("replay_events")
    op.drop_table("player_corpus")
    op.drop_table("deck_cards")
    op.drop_table("battles")
    op.drop_table("player_snapshots")
