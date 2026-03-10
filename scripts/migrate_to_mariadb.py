#!/usr/bin/env python3
"""Migrate data from SQLite to MariaDB.

Copies every row from the SQLite database to MariaDB using SQLAlchemy
Core bulk-inserts.  Safe to re-run — it uses INSERT IGNORE so already-migrated
rows are skipped.

Usage:
    python3 scripts/migrate_to_mariadb.py [--sqlite data/clash_royale_history.db]

Environment:
    DATABASE_URL   MariaDB target URL (default: mysql+pymysql://clash_stats:clash_stats_pw_2026@192.168.7.59/clash_stats)
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sqlalchemy import create_engine, event, inspect, text

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("migrate")

MARIADB_URL = os.environ.get(
    "DATABASE_URL",
    "mysql+pymysql://clash_stats:clash_stats_pw_2026@192.168.7.59/clash_stats",
)

CHUNK = 2000  # rows per INSERT batch

# Tables in dependency order (FK parents first)
TABLE_ORDER = [
    "player_snapshots",
    "battles",
    "deck_cards",
    "replay_events",
    "replay_summaries",
    "player_corpus",
    "game_features",
    "game_embeddings",
    "win_probability",
    "game_wp_summary",
    "alembic_version",
]


def _set_sqlite_pragmas(dbapi_conn, _record):
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA journal_mode=WAL")
    cur.execute("PRAGMA busy_timeout=30000")
    cur.close()


def main():
    parser = argparse.ArgumentParser(description="Migrate SQLite → MariaDB")
    parser.add_argument(
        "--sqlite",
        default="data/clash_royale_history.db",
        help="Path to source SQLite file",
    )
    parser.add_argument(
        "--tables",
        nargs="+",
        metavar="TABLE",
        help="Only migrate these tables (default: all)",
    )
    args = parser.parse_args()

    sqlite_url = f"sqlite:///{args.sqlite}"
    log.info("Source : %s", sqlite_url)
    log.info("Target : %s", MARIADB_URL)

    src_engine = create_engine(sqlite_url, echo=False)
    event.listen(src_engine, "connect", _set_sqlite_pragmas)

    # Run Alembic migrations on target so schema is current
    log.info("Running migrations on target …")
    from tracker.database import run_migrations
    run_migrations(MARIADB_URL)

    dst_engine = create_engine(
        MARIADB_URL, echo=False, pool_recycle=3600, pool_pre_ping=True
    )

    src_insp = inspect(src_engine)
    src_tables = set(src_insp.get_table_names())

    tables_to_migrate = args.tables if args.tables else TABLE_ORDER

    total_rows = 0
    with src_engine.connect() as src_conn, dst_engine.connect() as dst_conn:
        for table in tables_to_migrate:
            if table not in src_tables:
                log.info("  skip %-30s (not in source)", table)
                continue
            if table == "alembic_version":
                # Let Alembic manage this; don't copy
                continue

            count = src_conn.execute(
                text(f"SELECT COUNT(*) FROM `{table}`")
            ).scalar()
            log.info("  %-30s %8d rows", table, count)
            if count == 0:
                continue

            # Read all columns
            columns = [c["name"] for c in src_insp.get_columns(table)]
            col_list = ", ".join(f"`{c}`" for c in columns)
            placeholders = ", ".join(f":{c}" for c in columns)
            insert_sql = text(
                f"INSERT IGNORE INTO `{table}` ({col_list}) VALUES ({placeholders})"
            )

            offset = 0
            inserted = 0
            while offset < count:
                rows = src_conn.execute(
                    text(f"SELECT {col_list} FROM `{table}` LIMIT {CHUNK} OFFSET {offset}")
                ).fetchall()
                if not rows:
                    break
                batch = [dict(zip(columns, row)) for row in rows]
                result = dst_conn.execute(insert_sql, batch)
                dst_conn.commit()
                inserted += result.rowcount
                offset += len(rows)
                if offset % 50000 == 0 or offset >= count:
                    log.info("    … %d / %d", min(offset, count), count)

            total_rows += inserted
            log.info("    → inserted %d new rows", inserted)

    log.info("Migration complete. %d total rows inserted.", total_rows)


if __name__ == "__main__":
    main()
