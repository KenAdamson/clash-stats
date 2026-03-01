"""Add composite index for replay scrape lookups.

The replay scraper queries (player_tag, replay_fetched, battle_type) on every
player visit. A composite index turns this from a table scan into a direct
B-tree lookup — critical with 160K+ battles.

Revision ID: 008
Revises: 007
Create Date: 2026-02-28
"""
from typing import Sequence, Union

from alembic import op

revision: str = "008"
down_revision: Union[str, None] = "007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index(
        "idx_battles_replay_lookup",
        "battles",
        ["player_tag", "replay_fetched", "battle_type"],
    )


def downgrade() -> None:
    op.drop_index("idx_battles_replay_lookup", table_name="battles")
