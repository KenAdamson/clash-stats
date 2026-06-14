"""Add derived dimension tables: clan_dim and player_dim.

``clan_dim`` already exists in the live DB — it was created by an ad-hoc script,
NOT by Alembic, so it is un-tracked. This migration brings it under Alembic
management by DROPping it IF EXISTS and re-CREATEing it to match the ``ClanDim``
ORM model. That is safe because clan_dim is *fully derived* data,
repopulatable from the CR clan API via ``--refresh-dims`` — there is nothing to
preserve.

``player_dim`` is new — a per-opponent dimension aggregated from ``battles``.

Both tables are derived; neither holds a source of truth.

Revision ID: 006
Revises: 005
Create Date: 2026-06-14
"""

from alembic import op
import sqlalchemy as sa

revision = "006"
down_revision = "005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    if conn.dialect.name == "sqlite":
        # Tests use SQLite; metadata.create_all() builds these tables instead.
        return

    # clan_dim exists un-tracked (ad-hoc script). Drop and recreate so it
    # matches the ORM model exactly and is Alembic-managed going forward.
    # Derived data — DROP is safe; --refresh-dims repopulates from the CR API.
    op.execute("DROP TABLE IF EXISTS clan_dim")

    op.create_table(
        "clan_dim",
        sa.Column("clan_tag", sa.String, primary_key=True),
        sa.Column("clan_name", sa.String),
        sa.Column("member_count", sa.Integer),
        sa.Column("max_trophies", sa.Integer),
        sa.Column("avg_trophies", sa.Integer),
        sa.Column("median_trophies", sa.Integer),
        sa.Column("n_9k", sa.Integer),
        sa.Column("n_11k", sa.Integer),
        sa.Column("n_12k", sa.Integer),
        sa.Column("top_member_name", sa.String),
        sa.Column("top_member_tag", sa.String),
        sa.Column("top_member_trophies", sa.Integer),
        sa.Column("refreshed_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # player_dim — new derived opponent/player dimension.
    op.create_table(
        "player_dim",
        sa.Column("player_tag", sa.String, primary_key=True),
        sa.Column("name", sa.String),
        sa.Column("latest_trophies", sa.Integer),
        sa.Column("exp_level", sa.Integer),
        sa.Column("clan_tag", sa.String),
        sa.Column("first_seen", sa.DateTime(timezone=True)),
        sa.Column("last_seen", sa.DateTime(timezone=True)),
        sa.Column("games", sa.Integer, nullable=False, server_default="0"),
        sa.Column("wins", sa.Integer, nullable=False, server_default="0"),
        sa.Column("losses", sa.Integer, nullable=False, server_default="0"),
        sa.Column("last_deck_hash", sa.String),
        sa.Column("is_alt_suspect", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("refreshed_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("idx_player_dim_clan_tag", "player_dim", ["clan_tag"])
    op.create_index("idx_player_dim_last_seen", "player_dim", ["last_seen"])
    op.create_index("idx_player_dim_alt_suspect", "player_dim", ["is_alt_suspect"])


def downgrade() -> None:
    op.drop_table("player_dim")
    op.drop_table("clan_dim")
