"""Add player_king cache (king-level enrichment for the smurf 2x2).

King level isn't in the battle log; it's resolved from /players in priority
batches and cached here so it survives player_dim rebuilds. Disambiguates the
smurf species: low king + over-leveled cards = PAID whale; high king + low
trophies = MATURE account tanking.

Revision ID: 009
Revises: 008
Create Date: 2026-06-19
"""

from alembic import op
import sqlalchemy as sa

revision = "009"
down_revision = "008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    if conn.dialect.name == "sqlite":
        return  # tests build from metadata.create_all()
    op.create_table(
        "player_king",
        sa.Column("player_tag", sa.String, primary_key=True),
        sa.Column("king_level", sa.Integer),
        sa.Column("best_trophies", sa.Integer),
        sa.Column("resolved_at", sa.DateTime(timezone=True)),
        sa.Column("resolve_attempts", sa.Integer, nullable=False, server_default="0"),
        sa.Column("refreshed_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )


def downgrade() -> None:
    conn = op.get_bind()
    if conn.dialect.name == "sqlite":
        return
    op.drop_table("player_king")
