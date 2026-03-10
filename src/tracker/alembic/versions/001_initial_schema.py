"""Initial schema — all tables with evo tracking, elixir leak, and duration.

Revision ID: 001
Revises: None
Create Date: 2026-02-21
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "player_snapshots",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("timestamp", sa.DateTime, server_default=sa.func.now()),
        sa.Column("player_tag", sa.String(32), nullable=False),
        sa.Column("name", sa.String(64)),
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
        sa.Column("clan_tag", sa.String(32)),
        sa.Column("clan_name", sa.String(64)),
        sa.Column("arena_name", sa.String(64)),
        sa.Column("raw_json", sa.Text),
    )

    op.create_table(
        "battles",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("battle_id", sa.String(64), unique=True, nullable=False),
        sa.Column("timestamp", sa.DateTime, server_default=sa.func.now()),
        sa.Column("battle_time", sa.String(32)),
        sa.Column("battle_type", sa.String(32)),
        sa.Column("arena_name", sa.String(64)),
        sa.Column("game_mode_name", sa.String(64)),
        sa.Column("is_ladder_tournament", sa.Integer),
        sa.Column("player_tag", sa.String(32), nullable=False),
        sa.Column("player_name", sa.String(64)),
        sa.Column("player_starting_trophies", sa.Integer),
        sa.Column("player_trophy_change", sa.Integer),
        sa.Column("player_crowns", sa.Integer),
        sa.Column("player_king_tower_hp", sa.Integer),
        sa.Column("player_princess_tower_hp", sa.String(64)),
        sa.Column("player_deck", sa.Text),
        sa.Column("player_deck_hash", sa.String(64)),
        sa.Column("opponent_tag", sa.String(32)),
        sa.Column("opponent_name", sa.String(64)),
        sa.Column("opponent_starting_trophies", sa.Integer),
        sa.Column("opponent_trophy_change", sa.Integer),
        sa.Column("opponent_crowns", sa.Integer),
        sa.Column("opponent_king_tower_hp", sa.Integer),
        sa.Column("opponent_princess_tower_hp", sa.String(64)),
        sa.Column("opponent_deck", sa.Text),
        sa.Column("opponent_deck_hash", sa.String(64)),
        sa.Column("result", sa.String(8)),
        sa.Column("crown_differential", sa.Integer),
        sa.Column("raw_json", sa.Text),
        sa.Column("player_elixir_leaked", sa.Float),
        sa.Column("opponent_elixir_leaked", sa.Float),
        sa.Column("battle_duration", sa.Integer),
    )

    op.create_index("idx_battles_player_tag", "battles", ["player_tag"])
    op.create_index("idx_battles_battle_time", "battles", ["battle_time"])
    op.create_index("idx_battles_player_deck_hash", "battles", ["player_deck_hash"])
    op.create_index("idx_battles_result", "battles", ["result"])
    op.create_index("idx_battles_battle_type", "battles", ["battle_type"])

    op.create_table(
        "deck_cards",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("battle_id", sa.String(64), sa.ForeignKey("battles.battle_id"), nullable=False),
        sa.Column("card_name", sa.String(64), nullable=False),
        sa.Column("card_level", sa.Integer),
        sa.Column("card_max_level", sa.Integer),
        sa.Column("card_elixir", sa.Integer),
        sa.Column("is_player_deck", sa.Integer),
        sa.Column("evolution_level", sa.Integer, server_default="0"),
        sa.Column("star_level", sa.Integer, server_default="0"),
    )

    op.create_index("idx_deck_cards_card_name", "deck_cards", ["card_name"])
    op.create_index("idx_deck_cards_battle_id", "deck_cards", ["battle_id"])


def downgrade() -> None:
    op.drop_table("deck_cards")
    op.drop_table("battles")
    op.drop_table("player_snapshots")
