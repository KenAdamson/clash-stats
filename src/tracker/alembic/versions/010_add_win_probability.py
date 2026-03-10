"""Add win probability tables.

Revision ID: 010
Revises: 009
Create Date: 2026-03-10
"""

from alembic import op
import sqlalchemy as sa

revision = "010"
down_revision = "009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "win_probability",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("battle_id", sa.String, sa.ForeignKey("battles.battle_id"), nullable=False),
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

    op.create_table(
        "game_wp_summary",
        sa.Column("battle_id", sa.String, sa.ForeignKey("battles.battle_id"), primary_key=True),
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
    op.drop_index("idx_wp_criticality")
    op.drop_index("idx_wp_battle")
    op.drop_table("win_probability")
