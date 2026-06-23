"""Add pilot_fingerprint table + player_dim behavioral-match columns (smurf pillar 3).

The deck-invariant behavioral fingerprint (timing/economy/discipline features
from replay placements) and the behavioral-gap score derived from it: median
trophies of an account's nearest pilots minus its own trophies. A large positive
gap = plays like a much-higher-trophy pilot = the skill-smurf signal, orthogonal
to clan-shelter (pillar 1) and the funded/level gap (pillar 2).

Revision ID: 010
Revises: 009
Create Date: 2026-06-22
"""

from alembic import op
import sqlalchemy as sa

revision = "010"
down_revision = "009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    if conn.dialect.name == "sqlite":
        return  # tests build from metadata.create_all()
    op.create_table(
        "pilot_fingerprint",
        sa.Column("player_tag", sa.String, primary_key=True),
        sa.Column("elixir_pace", sa.Float),
        sa.Column("throughput", sa.Float),
        sa.Column("reaction", sa.Float),
        sa.Column("pace_consistency", sa.Float),
        sa.Column("def_reaction", sa.Float),
        sa.Column("fast_react_frac", sa.Float),
        sa.Column("n_games", sa.Integer, nullable=False, server_default="0"),
        sa.Column("latest_trophies", sa.Integer),
        sa.Column("refreshed_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.add_column("player_dim", sa.Column("behavioral_neighbor_trophy", sa.Integer))
    op.add_column("player_dim", sa.Column("behavioral_gap", sa.Integer))
    op.create_index("idx_player_dim_behavioral_gap", "player_dim", ["behavioral_gap"])


def downgrade() -> None:
    conn = op.get_bind()
    if conn.dialect.name == "sqlite":
        return
    op.drop_index("idx_player_dim_behavioral_gap", table_name="player_dim")
    op.drop_column("player_dim", "behavioral_gap")
    op.drop_column("player_dim", "behavioral_neighbor_trophy")
    op.drop_table("pilot_fingerprint")
