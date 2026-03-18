"""Convert battle_time from VARCHAR to DATETIME.

Parses the compact ISO 8601 strings ("20260228T200232.000Z") into native
DATETIME values for proper indexing and date function support.

Revision ID: 011
Revises: 010
Create Date: 2026-03-18
"""

from alembic import op
import sqlalchemy as sa

revision = "011"
down_revision = "010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name

    if dialect == "sqlite":
        # SQLite: battle_time strings already sort correctly as TEXT.
        # SQLite has no real DATETIME type — column affinity stays the same.
        # Just leave it as-is; the ORM handles conversion on read/write.
        pass

    elif dialect == "mysql":
        # MariaDB/MySQL: convert VARCHAR → DATETIME via STR_TO_DATE
        op.add_column("battles", sa.Column("battle_time_dt", sa.DateTime, nullable=True))

        op.execute("""
            UPDATE battles
            SET battle_time_dt = STR_TO_DATE(battle_time, '%Y%m%dT%H%i%s.000Z')
            WHERE battle_time IS NOT NULL
        """)

        # Drop old index and old column
        op.drop_index("idx_battles_battle_time", table_name="battles")
        op.drop_column("battles", "battle_time")

        # Rename new column — use raw SQL because Alembic's alter_column
        # doesn't reliably rename on MariaDB
        op.execute("ALTER TABLE battles CHANGE COLUMN battle_time_dt battle_time DATETIME NULL")

        op.create_index("idx_battles_battle_time", "battles", ["battle_time"])


def downgrade() -> None:
    bind = op.get_bind()
    dialect = bind.dialect.name

    if dialect == "sqlite":
        pass

    elif dialect == "mysql":
        op.add_column("battles", sa.Column("battle_time_str", sa.String(32), nullable=True))

        op.execute("""
            UPDATE battles
            SET battle_time_str = DATE_FORMAT(battle_time, '%Y%m%dT%H%i%s.000Z')
            WHERE battle_time IS NOT NULL
        """)

        op.drop_index("idx_battles_battle_time", table_name="battles")
        op.drop_column("battles", "battle_time")
        op.execute("ALTER TABLE battles CHANGE COLUMN battle_time_str battle_time VARCHAR(32) NULL")
        op.create_index("idx_battles_battle_time", "battles", ["battle_time"])
