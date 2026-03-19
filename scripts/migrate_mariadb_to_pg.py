#!/usr/bin/env python3
"""Migrate MariaDB → PostgreSQL using SQLAlchemy streaming.

Creates schema via Alembic models, then streams data table-by-table
in batches using server-side cursors. Handles raw_json TEXT→JSONB
conversion inline.

Usage:
    python scripts/migrate_mariadb_to_pg.py
"""

import json
import sys
import time
from datetime import datetime

from psycopg2.extras import Json
from sqlalchemy import create_engine, inspect, text, MetaData

MARIA_URL = "mysql+pymysql://clash_stats:clash_stats_pw_2026@192.168.7.58:3306/clash_stats"
PG_URL = "postgresql://clash_stats:clash_stats_pw_2026@192.168.7.58:5432/clash_stats"

BATCH_SIZE = 5000

# Tables in dependency order (parents before children)
TABLE_ORDER = [
    "player_snapshots",
    "battles",
    "deck_cards",
    "player_corpus",
    "replay_events",
    "replay_summaries",
    "game_features",
    "game_embeddings",
    "win_probability",
    "game_wp_summary",
]

# Columns that need TEXT→JSONB conversion (pass as dict, not string)
JSONB_COLUMNS = {
    "battles": "raw_json",
    "player_snapshots": "raw_json",
}


def get_row_count(engine, table: str) -> int:
    with engine.connect() as conn:
        return conn.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()


def migrate_table(maria_eng, pg_eng, table: str, columns: list[str]):
    """Stream one table from MariaDB → PostgreSQL in batches."""
    total = get_row_count(maria_eng, table)
    if total == 0:
        print(f"  {table}: empty, skipping")
        return

    jsonb_col = JSONB_COLUMNS.get(table)
    col_list = ", ".join(columns)
    placeholders = ", ".join(f":{c}" for c in columns)
    insert_sql = f"INSERT INTO {table} ({col_list}) VALUES ({placeholders})"

    print(f"  {table}: {total:,} rows", end="", flush=True)
    t0 = time.time()
    migrated = 0

    with maria_eng.connect().execution_options(stream_results=True) as maria_conn:
        result = maria_conn.execute(text(f"SELECT {col_list} FROM {table}"))

        batch = []
        for row in result:
            row_dict = row._asdict()

            # Convert raw_json TEXT → Json wrapper for psycopg2 JSONB
            if jsonb_col and row_dict.get(jsonb_col) is not None:
                try:
                    row_dict[jsonb_col] = Json(json.loads(row_dict[jsonb_col]))
                except (json.JSONDecodeError, TypeError):
                    row_dict[jsonb_col] = Json(row_dict[jsonb_col])

            batch.append(row_dict)

            if len(batch) >= BATCH_SIZE:
                with pg_eng.begin() as pg_conn:
                    pg_conn.execute(text(insert_sql), batch)
                migrated += len(batch)
                elapsed = time.time() - t0
                rate = migrated / elapsed if elapsed > 0 else 0
                print(f"\r  {table}: {migrated:,}/{total:,} ({rate:,.0f} rows/s)", end="", flush=True)
                batch = []

        # Flush remaining
        if batch:
            with pg_eng.begin() as pg_conn:
                pg_conn.execute(text(insert_sql), batch)
            migrated += len(batch)

    elapsed = time.time() - t0
    rate = migrated / elapsed if elapsed > 0 else 0
    print(f"\r  {table}: {migrated:,} rows in {elapsed:.1f}s ({rate:,.0f} rows/s)")


def reset_sequences(pg_eng, table: str, columns: list[str]):
    """Reset auto-increment sequences after COPY."""
    if "id" in columns:
        seq_name = f"{table}_id_seq"
        with pg_eng.begin() as conn:
            try:
                conn.execute(text(
                    f"SELECT setval('{seq_name}', COALESCE((SELECT MAX(id) FROM {table}), 1))"
                ))
            except Exception:
                pass  # Table might not have a sequence


def main():
    print(f"=== MariaDB → PostgreSQL Migration ===")
    print(f"  Source: {MARIA_URL.split('@')[1]}")
    print(f"  Target: {PG_URL.split('@')[1]}")
    print()

    maria_eng = create_engine(MARIA_URL, echo=False)
    pg_eng = create_engine(PG_URL, echo=False, pool_size=5)

    # Step 1: Create schema in PostgreSQL from our models
    print("Step 1: Creating schema...")
    from tracker.models import Base
    # Import ML models so they register on Base.metadata
    import tracker.ml.storage  # noqa: F401 — GameFeature, GameEmbedding
    import tracker.ml.wp_storage  # noqa: F401 — WinProbability, GameWPSummary
    Base.metadata.create_all(pg_eng)
    print("  Schema created (all tables)")

    # Step 2: Verify source tables
    maria_insp = inspect(maria_eng)
    pg_insp = inspect(pg_eng)

    print("\nStep 2: Migrating data...")
    t_total = time.time()

    for table in TABLE_ORDER:
        if table not in maria_insp.get_table_names():
            print(f"  {table}: not in source, skipping")
            continue
        if table not in pg_insp.get_table_names():
            print(f"  {table}: not in target schema, skipping")
            continue

        # Get column names from MariaDB (intersect with PG to handle schema diffs)
        maria_cols = {c["name"] for c in maria_insp.get_columns(table)}
        pg_cols = {c["name"] for c in pg_insp.get_columns(table)}
        columns = sorted(maria_cols & pg_cols)

        if not columns:
            print(f"  {table}: no common columns, skipping")
            continue

        migrate_table(maria_eng, pg_eng, table, columns)
        reset_sequences(pg_eng, table, columns)

    elapsed = time.time() - t_total
    print(f"\nMigration complete in {elapsed:.1f}s")

    # Step 3: Verify row counts
    print("\nStep 3: Verification...")
    all_good = True
    for table in TABLE_ORDER:
        if table not in maria_insp.get_table_names():
            continue
        m_count = get_row_count(maria_eng, table)
        try:
            p_count = get_row_count(pg_eng, table)
        except Exception:
            p_count = 0
        match = "OK" if m_count == p_count else "MISMATCH"
        if m_count != p_count:
            all_good = False
        print(f"  {table}: MariaDB={m_count:,}  PostgreSQL={p_count:,}  [{match}]")

    if all_good:
        print("\nAll row counts match.")
    else:
        print("\nWARNING: Row count mismatches detected!")
        sys.exit(1)

    # Step 4: Test JSONB
    print("\nStep 4: JSONB verification...")
    with pg_eng.connect() as conn:
        result = conn.execute(text(
            "SELECT raw_json->>'battleTime' AS bt FROM battles WHERE raw_json IS NOT NULL LIMIT 3"
        )).all()
        for row in result:
            print(f"  raw_json->>'battleTime' = {row[0]}")

    print("\nDone. Run scripts/pg_post_migration.sql next for manual indexes + Alembic stamp.")


if __name__ == "__main__":
    main()
