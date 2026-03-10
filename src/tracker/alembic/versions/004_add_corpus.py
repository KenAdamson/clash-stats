"""Add corpus column to battles and player_corpus table.

Supports multi-player data collection for ML training corpus
as described in ADR-007.

Revision ID: 004
Revises: 003
Create Date: 2026-02-22
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "004"
down_revision: Union[str, None] = "003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _column_exists(table: str, column: str) -> bool:
    """Check if a column already exists on a table (idempotent migrations)."""
    conn = op.get_bind()
    insp = sa.inspect(conn)
    try:
        columns = {c["name"] for c in insp.get_columns(table)}
    except Exception:
        return False
    return column in columns


def _table_exists(table: str) -> bool:
    """Check if a table already exists."""
    conn = op.get_bind()
    insp = sa.inspect(conn)
    return table in insp.get_table_names()


def upgrade() -> None:
    # Add corpus provenance column to battles
    if not _column_exists("battles", "corpus"):
        op.add_column("battles", sa.Column(
            "corpus", sa.String(32), server_default="personal",
        ))
        op.create_index("idx_battles_corpus", "battles", ["corpus"])

    # Create player_corpus table for tracking scraped players
    if not _table_exists("player_corpus"):
        op.create_table(
            "player_corpus",
            sa.Column("player_tag", sa.String(32), primary_key=True),
            sa.Column("player_name", sa.String(64)),
            sa.Column("source", sa.String(32), nullable=False),
            sa.Column("trophy_range_low", sa.Integer),
            sa.Column("trophy_range_high", sa.Integer),
            sa.Column("games_scraped", sa.Integer, server_default="0"),
            sa.Column("replays_scraped", sa.Integer, server_default="0"),
            sa.Column("last_scraped", sa.DateTime),
            sa.Column("active", sa.Integer, server_default="1"),
        )


def downgrade() -> None:
    op.drop_table("player_corpus")
    op.drop_index("idx_battles_corpus", "battles")
    op.drop_column("battles", "corpus")
