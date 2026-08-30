"""FastAPI dependencies — DB session."""

from __future__ import annotations

from typing import Generator

from sqlalchemy.orm import Session

from app.storage.db import SessionLocal


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
