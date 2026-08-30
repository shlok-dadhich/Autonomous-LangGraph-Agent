"""Seed demo user from config/profile.json."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.storage.db import SessionLocal, init_db
from app.domain.users import get_or_create_demo_user


def main():
    init_db()
    db = SessionLocal()
    try:
        user = get_or_create_demo_user(db)
        db.commit()
        print(f"Seeded user {user.email} id={user.id}")
    finally:
        db.close()


if __name__ == "__main__":
    main()
