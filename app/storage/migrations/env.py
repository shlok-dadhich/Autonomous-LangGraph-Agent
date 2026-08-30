"""Alembic env — supports both Postgres and SQLite via DATABASE_URL."""

from __future__ import annotations

import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import pool

# Ensure project root on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

# Import models so metadata is populated
import app.storage.models  # noqa: E402,F401
from app.core.config import get_settings  # noqa: E402
from app.storage.db import Base  # noqa: E402

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def get_url() -> str:
    settings = get_settings()
    return settings.resolved_database_url


def run_migrations_offline() -> None:
    url = get_url()
    context.configure(url=url, target_metadata=target_metadata, literal_binds=True, dialect_opts={"paramstyle": "named"})
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    from sqlalchemy import create_engine

    url = get_url()
    connectable = create_engine(url, poolclass=pool.NullPool, future=True)

    with connectable.connect() as connection:
        # Enable pgvector for Postgres if available
        if url.startswith("postgresql"):
            try:
                connection.exec_driver_sql("CREATE EXTENSION IF NOT EXISTS vector")
            except Exception:
                pass
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
