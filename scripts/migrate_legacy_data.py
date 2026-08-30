"""Migrate legacy data: profile.json, sent URLs, schedules → Postgres."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loguru import logger
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.domain.documents.identity import canonicalize_url, content_hash, title_hash
from app.domain.users import seed_user_from_profile
from app.storage.db import SessionLocal, get_engine, init_db
from app.storage.models import Document


def migrate_profile(db: Session, profile_path: str, email: str) -> None:
    path = Path(profile_path)
    if not path.exists():
        logger.warning(f"Profile not found: {profile_path}")
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    seed_user_from_profile(db, data, email=email)
    logger.info(f"Migrated profile {profile_path} -> user {email}")


def migrate_history(db: Session, db_path: Path | None = None) -> int:
    """Import legacy sent_articles URLs into documents (as already-seen markers)."""
    settings = get_settings()
    legacy_path = db_path or (settings.resolved_data_dir / "history.db")

    # If alembic already dropped the table, try to find backup or skip gracefully
    if not legacy_path.exists():
        logger.info(f"No legacy DB at {legacy_path}, skipping history migration")
        return 0

    # Open legacy DB directly via sqlite3 to inspect sent_articles before it was dropped
    # After Phase 1 migration, the table may already be gone — check first
    conn = sqlite3.connect(str(legacy_path))
    try:
        cur = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sent_articles'")
        if not cur.fetchone():
            logger.info("No sent_articles table (already migrated), skipping")
            return 0
        rows = conn.execute("SELECT url, sent_at FROM sent_articles").fetchall()
    finally:
        conn.close()

    imported = 0
    for url, sent_at in rows:
        if not url:
            continue
        canon = canonicalize_url(url)
        if not canon:
            continue
        # skip if already present
        exists = db.query(Document).filter_by(canonical_url=canon).first()
        if exists:
            continue
        doc = Document(
            canonical_url=canon,
            original_url=url,
            title=url,  # placeholder; original title not stored in legacy
            source_id=None,
            content_hash=content_hash(url),
            title_hash=title_hash(url),
            status="archived",
            meta={"legacy_sent_at": sent_at, "migrated": True},
        )
        db.add(doc)
        imported += 1
        if imported % 100 == 0:
            db.flush()
    logger.info(f"Migrated {imported} legacy sent URLs")
    return imported


def migrate_schedules(db: Session, schedules_path: str = "config/schedules.json") -> None:
    path = Path(schedules_path)
    if not path.exists():
        logger.info(f"No schedules at {schedules_path}")
        return
    data = json.loads(path.read_text(encoding="utf-8"))
    schedules = data.get("schedules", []) if isinstance(data, dict) else []
    logger.info(f"Found {len(schedules)} legacy schedules (preserved as config; DB schedules TODO Phase 4)")


def main():
    init_db()
    db = SessionLocal()
    try:
        # 1. profiles
        migrate_profile(db, "config/profile.json", email="demo@example.com")
        # try news profile as second user if exists
        if Path("config/profile_news.json").exists():
            migrate_profile(db, "config/profile_news.json", email="news@example.com")

        # 2. history
        migrate_history(db)

        # 3. schedules
        migrate_schedules(db)

        db.commit()
        print("Migration complete")
    except Exception as e:
        db.rollback()
        logger.error(f"Migration failed: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()
