"""Add game_features and game_embeddings tables for ML Phase 0.

Revision ID: 006
Revises: 005
Create Date: 2026-02-24
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "006"
down_revision: Union[str, None] = "005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "game_features",
        sa.Column("battle_id", sa.String(64), sa.ForeignKey("battles.battle_id"), primary_key=True),
        sa.Column("feature_vector", sa.LargeBinary(), nullable=False),
        sa.Column("feature_version", sa.String(32), server_default="v1"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )

    op.create_table(
        "game_embeddings",
        sa.Column("battle_id", sa.String(64), sa.ForeignKey("battles.battle_id"), primary_key=True),
        sa.Column("embedding_15d", sa.LargeBinary(), nullable=False),
        sa.Column("embedding_2d", sa.LargeBinary(), nullable=False),
        sa.Column("cluster_id", sa.Integer(), nullable=True),
        sa.Column("model_version", sa.String(32), server_default="umap-v1"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("game_embeddings")
    op.drop_table("game_features")
