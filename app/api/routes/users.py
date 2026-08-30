"""Users routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.storage.models import User

router = APIRouter()

@router.get("")
def list_users(db: Session = Depends(get_db)):
    rows = db.query(User).all()
    return [{"id": str(r.id), "email": r.email, "display_name": r.display_name} for r in rows]
