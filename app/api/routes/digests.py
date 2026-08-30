"""Digests routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.storage.models import Digest

router = APIRouter()

@router.get("")
def list_digests(db: Session = Depends(get_db), limit: int = 10):
    rows = db.query(Digest).order_by(Digest.created_at.desc()).limit(limit).all()
    return [{"id": str(r.id), "status": r.status, "stories": len(r.story_ids or []), "created_at": r.created_at.isoformat() if r.created_at else None} for r in rows]

@router.get("/{digest_id}")
def get_digest(digest_id: str, db: Session = Depends(get_db)):
    import uuid
    row = db.query(Digest).filter_by(id=uuid.UUID(digest_id)).first()
    if not row:
        from fastapi import HTTPException
        raise HTTPException(404, "Digest not found")
    return {"id": str(row.id), "status": row.status, "story_ids": row.story_ids, "html": row.rendered_html}
