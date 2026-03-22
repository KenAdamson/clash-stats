"""Add model registry table (ML Ops).

Revision ID: 003
Revises: 002
Create Date: 2026-03-21
"""

from alembic import op
from sqlalchemy import inspect
import sqlalchemy as sa

revision = "003"
down_revision = "002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    existing = set(inspect(conn).get_table_names())

    if "model_versions" not in existing:
        op.create_table(
            "model_versions",
            sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
            sa.Column("model_type", sa.String, nullable=False),
            sa.Column("version", sa.Integer, nullable=False),
            sa.Column("status", sa.String, nullable=False),
            sa.Column("filename", sa.String, nullable=False),
            sa.Column("epochs", sa.Integer),
            sa.Column("best_epoch", sa.Integer),
            sa.Column("training_games", sa.Integer),
            sa.Column("training_cutoff", sa.String),
            sa.Column("wall_time_seconds", sa.Integer),
            sa.Column("device", sa.String),
            sa.Column("val_loss", sa.Float),
            sa.Column("val_accuracy", sa.Float),
            sa.Column("metrics_json", sa.JSON, nullable=True),
            sa.Column("prev_version_id", sa.Integer, sa.ForeignKey("model_versions.id")),
            sa.Column("improvement_delta", sa.Float),
            sa.Column("trained_at", sa.DateTime, server_default=sa.func.now()),
            sa.Column("promoted_at", sa.DateTime),
            sa.Column("archived_at", sa.DateTime),
        )
        op.create_index("idx_model_versions_type_status", "model_versions", ["model_type", "status"])
        op.create_index("idx_model_versions_type_version", "model_versions", ["model_type", "version"])


def downgrade() -> None:
    op.drop_table("model_versions")
