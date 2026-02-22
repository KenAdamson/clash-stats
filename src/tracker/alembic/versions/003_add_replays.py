"""Add replay_events, replay_summaries tables and replay_fetched column.

Replay data is scraped from RoyaleAPI and stored per-battle with individual
card placement events and per-side elixir summaries.

Revision ID: 003
Revises: 002
Create Date: 2026-02-22
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _column_exists(table: str, column: str) -> bool:
    """Check if a column already exists on a table (idempotent migrations)."""
    conn = op.get_bind()
    result = conn.execute(sa.text(f"PRAGMA table_info({table})"))
    columns = {row[1] for row in result}
    return column in columns


def _table_exists(table: str) -> bool:
    """Check if a table already exists."""
    conn = op.get_bind()
    result = conn.execute(sa.text(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=:name"
    ), {"name": table})
    return result.fetchone() is not None


def upgrade() -> None:
    # Add replay_fetched flag to battles
    if not _column_exists("battles", "replay_fetched"):
        op.add_column("battles", sa.Column(
            "replay_fetched", sa.Integer, server_default="0",
        ))

    # Create replay_events table
    if not _table_exists("replay_events"):
        op.create_table(
            "replay_events",
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column("battle_id", sa.String, sa.ForeignKey("battles.battle_id"), nullable=False),
            sa.Column("side", sa.String, nullable=False),
            sa.Column("card_name", sa.String, nullable=False),
            sa.Column("game_tick", sa.Integer),
            sa.Column("arena_x", sa.Integer),
            sa.Column("arena_y", sa.Integer),
            sa.Column("play_number", sa.Integer, server_default="1"),
            sa.Column("ability_used", sa.Integer, server_default="0"),
        )
        op.create_index("idx_replay_events_battle_id", "replay_events", ["battle_id"])

    # Create replay_summaries table
    if not _table_exists("replay_summaries"):
        op.create_table(
            "replay_summaries",
            sa.Column("id", sa.Integer, primary_key=True),
            sa.Column("battle_id", sa.String, sa.ForeignKey("battles.battle_id"), nullable=False),
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


def downgrade() -> None:
    op.drop_table("replay_summaries")
    op.drop_table("replay_events")
    op.drop_column("battles", "replay_fetched")
