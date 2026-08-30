"""Stories routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.storage.models import StoryCluster

router = APIRouter()

@router.get("")
def list_stories(q: str | None = None, limit: int = 20, db: Session = Depends(get_db)):
    query = db.query(StoryCluster).order_by(StoryCluster.created_at.desc()).limit(limit)
    rows = query.all()
    if q:
        ql = q.lower()
        rows = [r for r in rows if ql in (r.title or "").lower()]
    return [{"id": str(r.id), "title": r.title, "confidence": r.cluster_confidence, "doc_count": len(r.document_ids or [])} for r in rows]

@router.get("/{story_id}")
def get_story(story_id: str, db: Session = Depends(get_db)):
    import uuid
    row = db.query(StoryCluster).filter_by(id=uuid.UUID(story_id)).first()
    if not row:
        from fastapi import HTTPException
        raise HTTPException(404, "Story not found")
    return {"id": str(row.id), "title": row.title, "summary": row.summary, "confidence": row.cluster_confidence, "document_ids": row.document_ids}
