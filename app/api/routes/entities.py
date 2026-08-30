"""Entities routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.storage.models import Entity

router = APIRouter()

@router.get("")
def list_entities(q: str | None = None, limit: int = 20, db: Session = Depends(get_db)):
    rows = db.query(Entity).limit(limit).all()
    if q:
        rows = [r for r in rows if q.lower() in r.canonical_name.lower()]
    return [{"id": str(r.id), "name": r.canonical_name, "kind": r.kind} for r in rows]
