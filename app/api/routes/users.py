"""Users routes — including delivery preferences (user-selectable time)."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.api.deps import get_db
from app.storage.models import User, UserPreference

router = APIRouter()


@router.get("")
def list_users(db: Session = Depends(get_db)):
    rows = db.query(User).all()
    return [{"id": str(r.id), "email": r.email, "display_name": r.display_name} for r in rows]


class PreferencesIn(BaseModel):
    delivery_time: str | None = None  # HH:MM, e.g. "22:00"
    timezone: str | None = None  # e.g. "Asia/Kolkata"
    explicit_topics: list[str] | None = None


@router.get("/{user_id}/preferences")
def get_preferences(user_id: str, db: Session = Depends(get_db)):
    pref = db.query(UserPreference).filter_by(user_id=uuid.UUID(user_id)).first()
    if not pref:
        raise HTTPException(404, "Preferences not found")
    return {
        "user_id": user_id,
        "delivery_time": pref.delivery_time,
        "timezone": pref.timezone,
        "explicit_topics": pref.explicit_topics,
        "preferred_depth": pref.preferred_depth,
    }


@router.patch("/{user_id}/preferences")
def update_preferences(user_id: str, body: PreferencesIn, db: Session = Depends(get_db)):
    pref = db.query(UserPreference).filter_by(user_id=uuid.UUID(user_id)).first()
    if not pref:
        raise HTTPException(404, "Preferences not found")
    if body.delivery_time is not None:
        # validate HH:MM
        import re

        if not re.match(r"^\d{2}:\d{2}$", body.delivery_time):
            raise HTTPException(400, "delivery_time must be HH:MM")
        pref.delivery_time = body.delivery_time
    if body.timezone is not None:
        pref.timezone = body.timezone
    if body.explicit_topics is not None:
        pref.explicit_topics = body.explicit_topics
    db.commit()
    return {"ok": True, "delivery_time": pref.delivery_time, "timezone": pref.timezone}


@router.post("/{user_id}/preferences/adaptive")
def enable_adaptive(user_id: str, enable: bool = True, db: Session = Depends(get_db)):
    """Enable/disable adaptive send window (learns from open/click)."""
    pref = db.query(UserPreference).filter_by(user_id=uuid.UUID(user_id)).first()
    if not pref:
        raise HTTPException(404, "Preferences not found")
    # store as meta flag
    pref.meta = {**(pref.meta or {}), "adaptive_delivery": enable}
    db.commit()
    return {"adaptive": enable}
