"""Split clan_dim into harvested identity + lazily-resolved measures.

clan_dim originally (006) held only API-fetched measure columns and was
truncate-and-rebuilt. That doesn't scale: the full corpus has ~490K distinct
opponent clans, which would be ~490K CR API calls. This migration adds the
IDENTITY columns (harvested for free from battle raw_json — every clan ever
seen) plus RESOLUTION tracking, so measures can be filled lazily in
priority-ordered batches by a background resolver instead of a single API burst.

All ADD COLUMN (cheap, no table rewrite); clan_dim is derived data.

Revision ID: 007
Revises: 006
Create Date: 2026-06-14
"""

from alembic import op
import sqlalchemy as sa

revision = "007"
down_revision = "006"
branch_labels = None
depends_on = None

_ADDED = (
    ("first_seen", sa.Column("first_seen", sa.DateTime(timezone=True))),
    ("last_seen", sa.Column("last_seen", sa.DateTime(timezone=True))),
    ("n_battles_seen", sa.Column("n_battles_seen", sa.Integer)),
    ("on_our_accounts", sa.Column(
        "on_our_accounts", sa.Boolean, nullable=False, server_default=sa.false())),
    ("resolved_at", sa.Column("resolved_at", sa.DateTime(timezone=True))),
    ("resolve_attempts", sa.Column(
        "resolve_attempts", sa.Integer, nullable=False, server_default="0")),
)


def upgrade() -> None:
    conn = op.get_bind()
    if conn.dialect.name == "sqlite":
        # Tests build the table from metadata.create_all() with the full schema.
        return
    for _, col in _ADDED:
        op.add_column("clan_dim", col)
    op.create_index("idx_clan_dim_resolved_at", "clan_dim", ["resolved_at"])
    op.create_index("idx_clan_dim_on_our_accounts", "clan_dim", ["on_our_accounts"])


def downgrade() -> None:
    conn = op.get_bind()
    if conn.dialect.name == "sqlite":
        return
    op.drop_index("idx_clan_dim_on_our_accounts", "clan_dim")
    op.drop_index("idx_clan_dim_resolved_at", "clan_dim")
    for name, _ in _ADDED:
        op.drop_column("clan_dim", name)
