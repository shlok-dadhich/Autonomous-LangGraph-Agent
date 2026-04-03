"""SQLite persistence utilities for history and checkpoint lifecycle."""

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import sqlite3
from typing import Generator, Iterable, List, Optional, Set

from loguru import logger


DATA_DIR = Path(os.getenv("NEWSLETTER_DATA_DIR", "data"))


def ensure_data_dir() -> Path:
    """Ensure data directory is available at startup and log its status."""
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"Persistent data directory ready: {DATA_DIR}")
    except Exception as exc:
        logger.error(f"Failed to initialize data directory {DATA_DIR}: {exc}")
        raise
    return DATA_DIR


def create_sqlite_connection(db_path: Path | str) -> sqlite3.Connection:
    """Create a SQLite connection configured to reduce lock contention."""
    conn = sqlite3.connect(str(db_path), timeout=30, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    return conn


@contextmanager
def sqlite_connection(db_path: Path | str) -> Generator[sqlite3.Connection, None, None]:
    """Context manager for safe SQLite open/commit/close operations."""
    conn = create_sqlite_connection(db_path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


class DatabaseManager:
    """Manage read/write operations for newsletter history database."""

    def __init__(self, db_path: Optional[str] = None) -> None:
        data_dir = ensure_data_dir()
        self.db_path = Path(db_path) if db_path else data_dir / "history.db"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _initialize(self) -> None:
        """Create history table if it does not exist."""
        with sqlite_connection(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sent_articles (
                    url TEXT PRIMARY KEY,
                    sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def get_sent_ids(self) -> Set[str]:
        """Return all previously sent article URLs."""
        with sqlite_connection(self.db_path) as conn:
            rows = conn.execute("SELECT url FROM sent_articles").fetchall()
        return {row[0] for row in rows}

    def add_sent_ids(self, ids: Iterable[str]) -> None:
        """Persist new sent article URLs, ignoring duplicates safely."""
        id_rows = [(url,) for url in ids if url]
        if not id_rows:
            return

        with sqlite_connection(self.db_path) as conn:
            conn.executemany("INSERT OR IGNORE INTO sent_articles(url) VALUES (?)", id_rows)

    def cleanup_checkpoints(self, days_to_keep: int = 30) -> int:
        """Delete old checkpoint rows from checkpoints.db only."""
        data_dir = ensure_data_dir()
        checkpoint_db_path = data_dir / "checkpoints.db"

        if not checkpoint_db_path.exists():
            logger.warning(f"Checkpoint DB not found, skipping cleanup: {checkpoint_db_path}")
            return 0

        cutoff = (datetime.now(timezone.utc) - timedelta(days=days_to_keep)).isoformat()

        with sqlite_connection(checkpoint_db_path) as conn:
            try:
                cursor = conn.execute(
                    "DELETE FROM checkpoints WHERE timestamp < ?",
                    (cutoff,),
                )
            except sqlite3.OperationalError:
                cursor = conn.execute(
                    "DELETE FROM checkpoints WHERE ts < ?",
                    (cutoff,),
                )

        deleted = cursor.rowcount if cursor.rowcount is not None else 0
        logger.info(
            "Checkpoint cleanup completed",
            days_to_keep=days_to_keep,
            deleted_rows=deleted,
        )
        return deleted


def purge_old_checkpoints(
    thread_id: Optional[str] = None,
    days_to_keep: int = 30,
    checkpoint_db_path: Optional[str] = None,
) -> int:
    """
    Purge old checkpoints using raw SQL without decoding checkpoint blobs.

    Deletes rows older than cutoff. If thread_id is provided, limits
    deletion to that thread only.
    """
    data_dir = ensure_data_dir()
    db_path = Path(checkpoint_db_path) if checkpoint_db_path else data_dir / "checkpoints.db"
    if not db_path.exists():
        logger.warning(f"Checkpoint DB not found, skipping purge: {db_path}")
        return 0

    cutoff = (datetime.now(timezone.utc) - timedelta(days=days_to_keep)).isoformat()

    with sqlite_connection(db_path) as conn:
        try:
            if thread_id:
                cursor = conn.execute(
                    "DELETE FROM checkpoints WHERE thread_id = ? AND timestamp < ?",
                    (thread_id, cutoff),
                )
            else:
                cursor = conn.execute(
                    "DELETE FROM checkpoints WHERE timestamp < ?",
                    (cutoff,),
                )
        except sqlite3.OperationalError:
            # Compatibility fallback for schemas using ts instead of timestamp.
            if thread_id:
                cursor = conn.execute(
                    "DELETE FROM checkpoints WHERE thread_id = ? AND ts < ?",
                    (thread_id, cutoff),
                )
            else:
                cursor = conn.execute(
                    "DELETE FROM checkpoints WHERE ts < ?",
                    (cutoff,),
                )

    deleted = cursor.rowcount if cursor.rowcount is not None else 0
    logger.info(
        "Purged old checkpoints",
        thread_id=thread_id,
        days_to_keep=days_to_keep,
        deleted_rows=deleted,
    )
    return deleted


def run_monthly_checkpoint_housekeeping(days_to_keep: int = 30) -> int:
    """
    Housekeeping job: purge stale LangGraph checkpoints only.

    Important: This function never touches history.db; sent article history
    is retained forever.
    """
    deleted = DatabaseManager().cleanup_checkpoints(days_to_keep=days_to_keep)
    logger.info(
        "Monthly checkpoint housekeeping complete",
        days_to_keep=days_to_keep,
        deleted_rows=deleted,
    )
    return deleted


def cleanup_checkpoints(days_to_keep: int = 30) -> int:
    """Backward-compatible module helper for checkpoint cleanup."""
    return DatabaseManager().cleanup_checkpoints(days_to_keep=days_to_keep)
