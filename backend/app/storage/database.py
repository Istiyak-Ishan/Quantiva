"""Hardened local SQLite engine and session lifecycle for Quantiva."""
from __future__ import annotations

import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import Engine, URL, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from app.config import ensure_data_directories, settings
from app.storage.models import Base


SQLITE_PRAGMAS: tuple[str, ...] = (
    "PRAGMA journal_mode=WAL;",
    "PRAGMA synchronous=NORMAL;",
    "PRAGMA busy_timeout=10000;",
    "PRAGMA foreign_keys=ON;",
    "PRAGMA temp_store=MEMORY;",
    "PRAGMA cache_size=-64000;",
)


@event.listens_for(Engine, "connect")
def configure_sqlite_connection(dbapi_connection, _connection_record) -> None:
    """Apply concurrency and integrity PRAGMAs to every SQLite connection."""
    if not isinstance(dbapi_connection, sqlite3.Connection):
        return
    cursor = dbapi_connection.cursor()
    try:
        for statement in SQLITE_PRAGMAS:
            cursor.execute(statement)
    finally:
        cursor.close()


def sqlite_database_path(data_dir: Path | str | None = None) -> Path:
    root = Path(data_dir or settings.QUANTIVA_DATA_DIR)
    return ensure_data_directories(root).database_file


def create_sqlite_engine(
    data_dir: Path | str | None = None,
    *,
    echo: bool = False,
) -> Engine:
    """Create a file-backed SQLite engine rooted in *data_dir*."""
    database_path = sqlite_database_path(data_dir)
    url = URL.create(drivername="sqlite", database=str(database_path))
    return create_engine(
        url,
        echo=echo,
        future=True,
        pool_pre_ping=True,
        connect_args={
            "check_same_thread": False,
            "timeout": 10.0,
        },
    )


def create_session_factory(bind: Engine) -> sessionmaker[Session]:
    return sessionmaker(
        bind=bind,
        class_=Session,
        expire_on_commit=False,
        autoflush=False,
        future=True,
    )


def initialize_database(bind: Engine | None = None) -> None:
    """Create all local tables idempotently."""
    Base.metadata.create_all(bind or engine)


# Concise compatibility alias for startup hooks and migration utilities.
init_database = initialize_database


@contextmanager
def session_scope(
    factory: sessionmaker[Session] | None = None,
) -> Generator[Session, None, None]:
    """Transactional session context that rolls back on failure."""
    session = (factory or SessionLocal)()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


engine = create_sqlite_engine()
SessionLocal = create_session_factory(engine)


__all__ = [
    "Base",
    "SQLITE_PRAGMAS",
    "SessionLocal",
    "configure_sqlite_connection",
    "create_session_factory",
    "create_sqlite_engine",
    "engine",
    "initialize_database",
    "init_database",
    "session_scope",
    "sqlite_database_path",
]
