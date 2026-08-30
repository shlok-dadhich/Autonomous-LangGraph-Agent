"""Ask Your Intelligence — grounded QA."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.storage.models import Document

router = APIRouter()

class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    citations: list[dict]

@router.post("", response_model=AskResponse)
def ask(req: AskRequest, db: Session = Depends(get_db)):
    # grounded retrieval: simple keyword over documents
    ql = req.question.lower()
    docs = db.query(Document).all()
    hits = [d for d in docs if any(tok in (d.title or "").lower() for tok in ql.split())][:5]
    if not hits:
        return AskResponse(answer="No stored evidence matches your question yet. Try broader terms or ingest more sources.", citations=[])
    citations = [{"id": str(h.id), "title": h.title, "url": h.canonical_url} for h in hits]
    # naive answer synthesis
    answer = f"Based on {len(hits)} stored document(s): " + "; ".join(h.title for h in hits[:3]) + ". (Retrieved from your corpus; not from model memory.)"
    return AskResponse(answer=answer, citations=citations)
