"""Database engine + session factory with Postgres/pgvector primary, SQLite fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import get_settings


class Base(DeclarativeBase):
    pass


def _connect_args(url: str) -> dict:
    if url.startswith("sqlite"):
        return {"check_same_thread": False}
    return {}


def get_engine(url: str | None = None):
    settings = get_settings()
    db_url = url or settings.resolved_database_url
    engine = create_engine(
        db_url,
        connect_args=_connect_args(db_url),
        pool_pre_ping=True,
        future=True,
    )
    # SQLite pragmas for WAL mode (matches legacy database.py)
    if db_url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _set_sqlite_pragma(dbapi_conn, _):  # type: ignore[no-untyped-def]
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("PRAGMA synchronous=NORMAL;")
            cursor.execute("PRAGMA busy_timeout=5000;")
            cursor.close()

    return engine


def _should_enable_pgvector(engine) -> bool:
    return engine.url.drivername.startswith("postgresql")


def init_db(engine=None) -> None:
    """Create all tables; enable pgvector if Postgres."""
    from app.storage import models  # noqa: F401  # ensure models imported

    eng = engine or get_engine()
    # Try to enable pgvector extension on Postgres
    if _should_enable_pgvector(eng):
        try:
            with eng.begin() as conn:
                conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        except Exception:
            # pgvector not available in test/CI Postgres without extension; continue
            pass
    Base.metadata.create_all(bind=eng)


# Global session factory (tests can override)
_settings = get_settings()
_engine = get_engine()
SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False, future=True)


def get_session() -> Generator[Session, None, None]:
    """FastAPI dependency — yields a Session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_session_factory(url: str | None = None) -> sessionmaker:
    """Return a sessionmaker bound to given URL (for scripts/tests)."""
    eng = get_engine(url)
    return sessionmaker(bind=eng, autoflush=False, autocommit=False, future=True)
