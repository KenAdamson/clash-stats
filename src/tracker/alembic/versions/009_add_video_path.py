"""Add video_path column to battles table (ADR-009).

Stores the path to the video capture file for visual game state recognition.
Videos live on /mnt/media/clash-videos/ and are linked to battles for
replay-guided label generation.

Revision ID: 009
Revises: 008
Create Date: 2026-03-06
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "009"
down_revision: Union[str, None] = "008"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("battles") as batch_op:
        batch_op.add_column(sa.Column("video_path", sa.String(512), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("battles") as batch_op:
        batch_op.drop_column("video_path")
