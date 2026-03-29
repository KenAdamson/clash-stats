"""Add native vector columns alongside BLOB columns (VectorChord migration).

Revision ID: 004
Revises: 003
Create Date: 2026-03-29
"""

from alembic import op
from sqlalchemy import inspect

revision = "004"
down_revision = "003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    conn = op.get_bind()
    existing_cols = {
        c["name"] for c in inspect(conn).get_columns("game_embeddings")
    }
    feature_cols = {
        c["name"] for c in inspect(conn).get_columns("game_features")
    }

    # Ensure extensions exist (idempotent)
    op.execute("CREATE EXTENSION IF NOT EXISTS vchord CASCADE")

    # game_embeddings: properly-typed vector columns
    if "embedding_tcn_128d" not in existing_cols:
        op.execute(
            "ALTER TABLE game_embeddings ADD COLUMN embedding_tcn_128d vector(128)"
        )
    if "embedding_vec_3d" not in existing_cols:
        op.execute(
            "ALTER TABLE game_embeddings ADD COLUMN embedding_vec_3d vector(3)"
        )

    # game_features: native vector column (table is currently empty but
    # will be populated when features are next extracted)
    if "feature_vec" not in feature_cols:
        op.execute(
            "ALTER TABLE game_features ADD COLUMN feature_vec vector(50)"
        )


def downgrade() -> None:
    op.execute("ALTER TABLE game_features DROP COLUMN IF EXISTS feature_vec")
    op.execute(
        "ALTER TABLE game_embeddings DROP COLUMN IF EXISTS embedding_vec_3d"
    )
    op.execute(
        "ALTER TABLE game_embeddings DROP COLUMN IF EXISTS embedding_tcn_128d"
    )
