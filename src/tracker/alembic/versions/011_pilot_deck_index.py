"""Add partial index on battles(player_tag, player_deck_hash) WHERE replay_fetched.

Accelerates pilot-and-deck enumeration for the pilot-verification training
pipeline (deck-disjoint positive-pair sampling). The deck-diversity aggregate
went from a 280s+ timeout to ~0.9s (index-only scan). Already created live with
CREATE INDEX CONCURRENTLY; this records it so a fresh DB rebuilds it. Idempotent.

Revision ID: 011
Revises: 010
Create Date: 2026-06-23
"""

from alembic import op

revision = "011"
down_revision = "010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    if conn.dialect.name == "sqlite":
        return  # perf-only index; tests don't need it
    op.execute(
        "CREATE INDEX IF NOT EXISTS idx_battles_pilot_deck_replay "
        "ON battles (player_tag, player_deck_hash) WHERE replay_fetched = 1"
    )


def downgrade() -> None:
    conn = op.get_bind()
    if conn.dialect.name == "sqlite":
        return
    op.execute("DROP INDEX IF EXISTS idx_battles_pilot_deck_replay")
