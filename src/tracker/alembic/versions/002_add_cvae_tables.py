"""Add CVAE counterfactual tables (ADR-006).

Revision ID: 002
Revises: 001
Create Date: 2026-03-19
"""

from alembic import op
from sqlalchemy import inspect
import sqlalchemy as sa

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Guard against tables already created via Base.metadata.create_all
    # (happens in test fixtures that mix create_all with Alembic)
    conn = op.get_bind()
    existing = set(inspect(conn).get_table_names())

    if "counterfactual_results" not in existing:
        op.create_table(
            "counterfactual_results",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("battle_id", sa.String, sa.ForeignKey("battles.battle_id"), nullable=False),
        sa.Column("old_card", sa.String, nullable=False),
        sa.Column("new_card", sa.String, nullable=False),
        sa.Column("original_wp", sa.Float, nullable=False),
        sa.Column("counterfactual_wp_mean", sa.Float, nullable=False),
        sa.Column("counterfactual_wp_std", sa.Float, nullable=False),
        sa.Column("delta_wp", sa.Float, nullable=False),
        sa.Column("n_samples", sa.Integer, nullable=False),
        sa.Column("model_version", sa.String, nullable=False),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("raw_json", sa.JSON, nullable=True),
    )
        op.create_index(
            "idx_cf_results_battle_id", "counterfactual_results", ["battle_id"],
        )

    if "deck_gradient_results" not in existing:
        op.create_table(
            "deck_gradient_results",
            sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column("old_card", sa.String, nullable=False),
            sa.Column("new_card", sa.String, nullable=False),
            sa.Column("mean_delta_wp", sa.Float, nullable=False),
            sa.Column("ci_low", sa.Float, nullable=False),
            sa.Column("ci_high", sa.Float, nullable=False),
            sa.Column("n_games", sa.Integer, nullable=False),
            sa.Column("model_version", sa.String, nullable=False),
            sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        )
        op.create_index(
            "idx_dg_results_old_card", "deck_gradient_results", ["old_card"],
        )


def downgrade() -> None:
    op.drop_table("deck_gradient_results")
    op.drop_table("counterfactual_results")
