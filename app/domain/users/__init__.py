"""User domain — seeding from profile.json bootstrap."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.storage.models import User, UserPreference


def load_profile(path: str | Path = "config/profile.json") -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Profile not found: {path}")
    return json.loads(p.read_text(encoding="utf-8"))


def seed_user_from_profile(db: Session, profile: dict[str, Any], email: str = "demo@example.com") -> User:
    """Idempotent seed: create or update User + UserPreference from profile dict."""
    topics = profile.get("topics", [])
    keywords = profile.get("keywords", [])
    # find or create user
    user = db.query(User).filter_by(email=email).first()
    if not user:
        user = User(email=email, display_name="Demo User", timezone="UTC")
        db.add(user)
        db.flush()

    pref = db.query(UserPreference).filter_by(user_id=user.id).first()
    if not pref:
        pref = UserPreference(user_id=user.id)
        db.add(pref)

    # map profile -> preferences
    pref.explicit_topics = topics
    pref.excluded_topics = profile.get("excluded_topics", [])
    pref.preferred_sources = list(profile.get("sources", {}).keys())
    pref.topic_affinity = dict.fromkeys(topics, 1.0)
    # store original profile for audit
    pref.meta = {"seed_profile": profile}
    db.flush()
    return user


def get_or_create_demo_user(db: Session, profile_path: str = "config/profile.json") -> User:
    profile = load_profile(profile_path)
    return seed_user_from_profile(db, profile)
