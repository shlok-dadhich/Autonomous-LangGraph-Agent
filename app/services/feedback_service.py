"""Feedback service — records UserInteraction and aggregates."""

from __future__ import annotations

import uuid

from sqlalchemy.orm import Session

from app.storage.models import UserInteraction

VALID_ACTIONS = {"OPEN", "CLICK", "SAVE", "LIKE", "DISLIKE", "HIDE", "MUTE_TOPIC", "FOLLOW", "UNFOLLOW", "SKIP", "SHARE", "SEARCH", "READ_DURATION"}


def record_interaction(
    db: Session,
    user_id: str | uuid.UUID,
    target_type: str,
    target_id: str,
    action: str,
    context: dict | None = None,
) -> UserInteraction:
    act = action.upper()
    if act not in VALID_ACTIONS:
        raise ValueError(f"Invalid action: {action}")
    inter = UserInteraction(
        user_id=user_id if isinstance(user_id, uuid.UUID) else uuid.UUID(str(user_id)),
        target_type=target_type,
        target_id=target_id,
        action=act,
        context=context or {},
    )
    db.add(inter)
    db.flush()
    return inter


def get_interactions(db: Session, user_id: str | uuid.UUID, limit: int = 100) -> list[UserInteraction]:
    uid = user_id if isinstance(user_id, uuid.UUID) else uuid.UUID(str(user_id))
    return db.query(UserInteraction).filter_by(user_id=uid).order_by(UserInteraction.created_at.desc()).limit(limit).all()
