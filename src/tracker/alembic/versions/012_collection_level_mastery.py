"""Collection Level + per-card mastery from the badges array.

King Level was replaced by Collection Level in-game. The API kept `expLevel`
as a LEGACY field: it only carries a real value for accounts that existed
before the rework, and post-rework accounts read 1 (with expPoints and
totalExpPoints at 0). The documented `collectionLevel` field is a dead
placeholder — it reads 0 for every account tested. The live value is in the
`badges` array as a badge named "CollectionLevel" (level = the collection
level, progress/target = progress within it).

Left as-is deliberately: `player_king.king_level` keeps its legacy meaning
and is NOT backfilled from collection level — the two are different scales,
and silently mixing them would corrupt the existing smurf-gap reference
(008) that was calibrated against king levels.

Also adds per-card mastery: the badges array carries Mastery<Card> entries
whose level counts USAGE, not upgrade investment — a behavioural signal the
corpus has never captured.

Revision ID: 012
Revises: 011
"""

import sqlalchemy as sa
from alembic import op

revision = "012"
down_revision = "011"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Collection Level alongside (not replacing) the legacy king level.
    op.add_column("player_king", sa.Column("collection_level", sa.Integer))
    op.add_column("player_king", sa.Column("collection_progress", sa.Integer))
    op.add_column("player_dim", sa.Column("collection_level", sa.Integer))

    # Per-card mastery: one row per (player, card). Mastery is a usage
    # measure (games played with the card), distinct from card_level.
    op.create_table(
        "player_card_mastery",
        sa.Column("player_tag", sa.String, primary_key=True),
        sa.Column("card_name", sa.String, primary_key=True),
        sa.Column("mastery_level", sa.Integer),
        sa.Column("mastery_progress", sa.Integer),
        sa.Column("refreshed_at", sa.DateTime(timezone=True)),
    )
    op.create_index("idx_card_mastery_card", "player_card_mastery", ["card_name"])

    # Per-mode progression from the undocumented `progress` dict (mode ->
    # {arena, trophies, bestTrophies}). Undocumented means it may change or
    # vanish without notice, so this is stored loosely rather than modelled.
    op.create_table(
        "player_mode_progress",
        sa.Column("player_tag", sa.String, primary_key=True),
        sa.Column("mode_key", sa.String, primary_key=True),
        sa.Column("arena_id", sa.Integer),
        sa.Column("arena_name", sa.String),
        sa.Column("trophies", sa.Integer),
        sa.Column("best_trophies", sa.Integer),
        sa.Column("refreshed_at", sa.DateTime(timezone=True)),
    )


def downgrade() -> None:
    op.drop_table("player_mode_progress")
    op.drop_index("idx_card_mastery_card", table_name="player_card_mastery")
    op.drop_table("player_card_mastery")
    op.drop_column("player_dim", "collection_level")
    op.drop_column("player_king", "collection_progress")
    op.drop_column("player_king", "collection_level")
