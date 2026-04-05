"""Add corpus_hourly_stats aggregate table.

Pre-aggregates corpus battle counts by hour-of-day so the dashboard
reads 24 rows instead of scanning millions.

Revision ID: 005
Revises: 004
Create Date: 2026-04-05
"""

from alembic import op
import sqlalchemy as sa

revision = "005"
down_revision = "004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    if conn.dialect.name == "sqlite":
        return

    op.create_table(
        "corpus_hourly_stats",
        sa.Column("hour", sa.Integer, primary_key=True),
        sa.Column("battle_count", sa.Integer, nullable=False, server_default="0"),
    )

    # Backfill from existing corpus battles
    conn.execute(sa.text("""
        INSERT INTO corpus_hourly_stats (hour, battle_count)
        SELECT EXTRACT(hour FROM battle_time)::int AS hour, COUNT(*) AS battle_count
        FROM battles
        WHERE battle_time IS NOT NULL AND corpus != 'personal'
        GROUP BY EXTRACT(hour FROM battle_time)::int
        ON CONFLICT (hour) DO UPDATE SET battle_count = EXCLUDED.battle_count
    """))


def downgrade() -> None:
    op.drop_table("corpus_hourly_stats")
