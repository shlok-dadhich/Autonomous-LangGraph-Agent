"""Feedback routes."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.services.feedback_service import record_interaction

router = APIRouter()

class FeedbackIn(BaseModel):
    user_id: str
    target_type: str
    target_id: str
    action: str
    context: dict | None = None

@router.post("")
def post_feedback(body: FeedbackIn, db: Session = Depends(get_db)):
    inter = record_interaction(db, body.user_id, body.target_type, body.target_id, body.action, body.context)
    db.commit()
    return {"id": str(inter.id), "action": inter.action}

@router.get("/save")
def save_via_get(story: str, user_id: str = "demo@example.com", db: Session = Depends(get_db)):
    # GET handler for email links: ?story=... saves
    try:
        inter = record_interaction(db, user_id, "story", story, "SAVE", {})
        db.commit()
        return {"saved": True, "story": story}
    except Exception as e:
        return {"saved": False, "error": str(e)}
