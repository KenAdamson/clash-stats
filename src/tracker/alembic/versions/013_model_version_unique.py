"""Deduplicate model_versions and enforce one row per (model_type, version).

The registry had no uniqueness guarantee, so re-registering an already-trained
checkpoint (which the eval harness does) silently inserted a second row rather
than updating the first. wp v12 and v13 each ended up with two rows carrying
slightly different rounded metrics.

That is not cosmetic. `model_registry.promote()` resolves the target with
`scalar_one_or_none()`, which raises MultipleResultsFound on a duplicated
version — so a duplicated row makes its model impossible to promote at all.

Dedupe keeps the LOWEST id: that is the row written by the training run itself,
carrying full-precision val_loss/val_accuracy, rather than the later
re-registration that rounded them to 4 decimal places.

Revision ID: 013
Revises: 012
"""

from alembic import op
import sqlalchemy as sa

revision = "013"
down_revision = "012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Collapse duplicate (model_type, version) rows, then add the constraint."""
    conn = op.get_bind()

    # Re-point any FK reference to a surviving row before deleting its twin,
    # or prev_version_id could be left dangling.
    conn.execute(sa.text("""
        UPDATE model_versions m SET prev_version_id = keep.id
        FROM (
            SELECT model_type, version, MIN(id) AS id
            FROM model_versions GROUP BY model_type, version
        ) keep
        JOIN model_versions dup
          ON dup.model_type = keep.model_type
         AND dup.version = keep.version
         AND dup.id <> keep.id
        WHERE m.prev_version_id = dup.id
    """))

    result = conn.execute(sa.text("""
        DELETE FROM model_versions d
        USING (
            SELECT model_type, version, MIN(id) AS keep_id
            FROM model_versions GROUP BY model_type, version
        ) k
        WHERE d.model_type = k.model_type
          AND d.version = k.version
          AND d.id <> k.keep_id
    """))
    print(f"  013: removed {result.rowcount} duplicate model_versions row(s)")

    op.create_unique_constraint(
        "uq_model_versions_type_version", "model_versions", ["model_type", "version"]
    )


def downgrade() -> None:
    """Drop the constraint. Deleted duplicates are not restored."""
    op.drop_constraint(
        "uq_model_versions_type_version", "model_versions", type_="unique"
    )
