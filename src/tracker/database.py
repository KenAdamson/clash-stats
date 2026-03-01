"""Database engine, session management, and Alembic migration runner."""

import os
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from tracker.models import Base

ALEMBIC_DIR = Path(__file__).parent / "alembic"


def _set_sqlite_pragmas(dbapi_conn, connection_record):
    """Enable WAL mode and set busy timeout for concurrent access."""
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.close()


def get_engine(db_path: str) -> Engine:
    """Create a SQLAlchemy engine for the given SQLite database.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        SQLAlchemy Engine instance.
    """
    engine = create_engine(f"sqlite:///{db_path}", echo=False)
    event.listen(engine, "connect", _set_sqlite_pragmas)
    return engine


def get_session(engine: Engine) -> Session:
    """Create a new session bound to the given engine.

    Args:
        engine: SQLAlchemy Engine instance.

    Returns:
        SQLAlchemy Session instance.
    """
    factory = sessionmaker(bind=engine)
    return factory()


def _get_alembic_config(db_path: str) -> Config:
    """Build an Alembic Config pointing at our migrations directory.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        Alembic Config instance.
    """
    cfg = Config()
    cfg.set_main_option("script_location", str(ALEMBIC_DIR))
    cfg.set_main_option("sqlalchemy.url", f"sqlite:///{db_path}")
    return cfg


def _database_exists(engine: Engine) -> bool:
    """Check if the battles table already exists (i.e. pre-existing database)."""
    insp = inspect(engine)
    return "battles" in insp.get_table_names()


def run_migrations(db_path: str) -> None:
    """Run Alembic migrations to bring the database up to date.

    For existing databases (created before Alembic was added), this stamps
    the initial revision so Alembic knows the schema is current, then runs
    any subsequent migrations.

    Args:
        db_path: Path to the SQLite database file.
    """
    engine = get_engine(db_path)
    cfg = _get_alembic_config(db_path)

    # Check if this is a pre-Alembic database
    insp = inspect(engine)
    tables = insp.get_table_names()
    has_alembic = "alembic_version" in tables
    has_battles = "battles" in tables

    if has_battles and not has_alembic:
        # Existing database without Alembic — stamp as already at initial revision
        command.stamp(cfg, "001")
        # Drop the old hand-rolled schema_version table if present
        if "schema_version" in tables:
            with engine.begin() as conn:
                conn.execute(text("DROP TABLE schema_version"))

    # Run any pending migrations
    command.upgrade(cfg, "head")
    engine.dispose()


def init_db(db_path: str) -> Engine:
    """Initialize the database: run migrations and return an engine.

    This is the main entry point for setting up the database connection.

    Args:
        db_path: Path to the SQLite database file.

    Returns:
        SQLAlchemy Engine instance ready for use.
    """
    run_migrations(db_path)
    return get_engine(db_path)
