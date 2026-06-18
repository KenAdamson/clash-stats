"""Add level_trophy_ref + player_dim smurf-gap columns (funded-smurf pillar).

Third pillar of the smurf score: the levels-implied-trophy gap. A deck's top
displayed card level implies where it normally sits on the ladder (level_trophy_
ref, derived empirically from the corpus); an opponent sitting far BELOW that
implied placement is a funded/pay-to-win smurf (cards belong higher than their
trophies). Orthogonal to clan-shelter (pillar 1) and behavioral/skill.

Adds player_dim.deck_top_level + player_dim.implied_trophy_gap, and a small
level_trophy_ref reference table. All derived/refreshable — cheap ADD COLUMN +
CREATE.

Revision ID: 008
Revises: 007
Create Date: 2026-06-18
"""

from alembic import op
import sqlalchemy as sa

revision = "008"
down_revision = "007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    if conn.dialect.name == "sqlite":
        return  # tests build from metadata.create_all()
    op.add_column("player_dim", sa.Column("deck_top_level", sa.Integer))
    op.add_column("player_dim", sa.Column("implied_trophy_gap", sa.Integer))
    op.create_index("idx_player_dim_implied_gap", "player_dim", ["implied_trophy_gap"])
    op.create_table(
        "level_trophy_ref",
        sa.Column("deck_top_level", sa.Integer, primary_key=True),
        sa.Column("median_trophy", sa.Integer),
        sa.Column("p10_trophy", sa.Integer),
        sa.Column("n_samples", sa.Integer),
        sa.Column("refreshed_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )


def downgrade() -> None:
    conn = op.get_bind()
    if conn.dialect.name == "sqlite":
        return
    op.drop_table("level_trophy_ref")
    op.drop_index("idx_player_dim_implied_gap", "player_dim")
    op.drop_column("player_dim", "implied_trophy_gap")
    op.drop_column("player_dim", "deck_top_level")
