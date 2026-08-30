"""Search — hybrid (keyword + semantic placeholder)."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.storage.models import Document

router = APIRouter()

@router.get("")
def search(q: str, limit: int = 10, db: Session = Depends(get_db)):
    ql = q.lower()
    rows = db.query(Document).all()
    matched = [r for r in rows if ql in (r.title or "").lower() or ql in (r.summary or "").lower()]
    matched = matched[:limit]
    return [{"id": str(r.id), "title": r.title, "url": r.canonical_url, "source_tier": r.source_tier} for r in matched]
