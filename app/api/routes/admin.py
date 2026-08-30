"""Admin routes — health, ranking explainer."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.api.deps import get_db

router = APIRouter()


@router.get("/health")
def health(db: Session = Depends(get_db)):
    checks: dict = {}
    try:
        db.execute(text("SELECT 1"))
        checks["db"] = "ok"
    except Exception as e:
        checks["db"] = f"fail: {e}"
    # source health placeholder
    checks["phase"] = "7-hardened"
    # metrics placeholder
    checks["metrics"] = {
        "source_success_rate": "n/a (Phase 7 metrics via app/core/metrics)",
        "clusters_created": "see /stories",
        "citation_coverage": "evidence-gated",
    }
    return checks


@router.get("/ranking-explain")
def ranking_explain(doc_id: str, db: Session = Depends(get_db)):
    import uuid

    from app.storage.models import Document

    row = db.query(Document).filter_by(id=uuid.UUID(doc_id)).first()
    if not row:
        return {"error": "not found"}
    from app.intelligence.ranking import score_document

    breakdown = score_document({"title": row.title or "", "description": row.summary or "", "published_at": row.published_at.isoformat() if row.published_at else None, "source_tier": row.source_tier})
    return {"id": doc_id, "title": row.title, "breakdown": breakdown}
