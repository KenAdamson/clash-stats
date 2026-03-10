"""Rename embedding_2d to embedding_3d for 3-component UMAP visualization.

Revision ID: 007
Revises: 006
Create Date: 2026-02-26
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "007"
down_revision: Union[str, None] = "006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("game_embeddings") as batch_op:
        batch_op.alter_column(
            "embedding_2d",
            new_column_name="embedding_3d",
            existing_type=sa.LargeBinary(),
        )


def downgrade() -> None:
    with op.batch_alter_table("game_embeddings") as batch_op:
        batch_op.alter_column(
            "embedding_3d",
            new_column_name="embedding_2d",
            existing_type=sa.LargeBinary(),
        )
