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


def _db_url(db_ref: str) -> str:
    """Convert a file path or existing URL to a full SQLAlchemy URL.

    Args:
        db_ref: Either a SQLite file path or a full database URL
                (e.g. ``postgresql://user:pw@host/db``).

    Returns:
        Full SQLAlchemy connection URL string.
    """
    if "://" in db_ref:
        # Normalize postgres:// to postgresql:// for SQLAlchemy compatibility
        if db_ref.startswith("postgres://"):
            db_ref = "postgresql://" + db_ref[len("postgres://"):]
        return db_ref
    return f"sqlite:///{db_ref}"


def _is_sqlite(url: str) -> bool:
    """Return True if the URL is a SQLite connection."""
    return url.startswith("sqlite")


def _set_sqlite_pragmas(dbapi_conn, connection_record):
    """Enable WAL mode and set busy timeout for concurrent access."""
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA busy_timeout=30000")
    cursor.close()


def get_engine(db_ref: str) -> Engine:
    """Create a SQLAlchemy engine for the given database.

    Args:
        db_ref: Either a SQLite file path or a full database URL
                (e.g. ``postgresql://user:pw@host/db``).

    Returns:
        SQLAlchemy Engine instance.
    """
    url = _db_url(db_ref)
    if _is_sqlite(url):
        engine = create_engine(url, echo=False)
        event.listen(engine, "connect", _set_sqlite_pragmas)
    else:
        # PostgreSQL — connection pooling with reconnect on stale connections
        engine = create_engine(
            url,
            echo=False,
            pool_size=20,
            max_overflow=20,
            pool_recycle=3600,
            pool_pre_ping=True,
        )
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


def _get_alembic_config(db_ref: str) -> Config:
    """Build an Alembic Config pointing at our migrations directory.

    Args:
        db_ref: Either a SQLite file path or a full database URL.

    Returns:
        Alembic Config instance.
    """
    cfg = Config()
    cfg.set_main_option("script_location", str(ALEMBIC_DIR))
    cfg.set_main_option("sqlalchemy.url", _db_url(db_ref))
    return cfg


def _database_exists(engine: Engine) -> bool:
    """Check if the battles table already exists (i.e. pre-existing database)."""
    insp = inspect(engine)
    return "battles" in insp.get_table_names()


def run_migrations(db_ref: str) -> None:
    """Run Alembic migrations to bring the database up to date.

    For existing SQLite databases created before Alembic was added, this stamps
    the initial revision so Alembic knows the schema is current, then runs
    any subsequent migrations.  Fresh PostgreSQL databases always run all
    migrations from revision 001 to head.

    Args:
        db_ref: Either a SQLite file path or a full database URL.
    """
    engine = get_engine(db_ref)
    cfg = _get_alembic_config(db_ref)

    url = _db_url(db_ref)
    if _is_sqlite(url):
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


def init_db(db_ref: str) -> Engine:
    """Initialize the database: run migrations and return an engine.

    This is the main entry point for setting up the database connection.

    Args:
        db_ref: Either a SQLite file path or a full database URL
                (e.g. ``postgresql://user:pw@host/db``).  Also honours
                the ``DATABASE_URL`` environment variable — if set it takes
                precedence over ``db_ref``.

    Returns:
        SQLAlchemy Engine instance ready for use.
    """
    effective = os.environ.get("DATABASE_URL", db_ref)
    run_migrations(effective)
    return get_engine(effective)
