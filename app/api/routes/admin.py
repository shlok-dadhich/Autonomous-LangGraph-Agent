"""Admin routes — health, ranking explainer."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.storage.db import get_engine

router = APIRouter()

@router.get("/health")
def health(db: Session = Depends(get_db)):
    from sqlalchemy import text
    try:
        db.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False
    return {"db": db_ok, "phase": "5"}

@router.get("/ranking-explain")
def ranking_explain(doc_id: str, db: Session = Depends(get_db)):
    from app.storage.models import Document
    import uuid
    row = db.query(Document).filter_by(id=uuid.UUID(doc_id)).first()
    if not row:
        return {"error": "not found"}
    # dummy breakdown
    return {"id": doc_id, "title": row.title, "breakdown": {"relevance": 0.8, "authority": 0.9, "final_score": 0.85}}
