"""Digest service — idempotent delivery handling."""

from __future__ import annotations

import uuid
from sqlalchemy.orm import Session

from app.storage.models import Delivery, Digest


def create_digest(db: Session, user_id, story_ids: list[str]) -> Digest:
    digest = Digest(user_id=user_id, story_ids=story_ids, status="draft")
    db.add(digest)
    db.flush()
    return digest


def mark_digest(db: Session, digest_id, status: str) -> None:
    d = db.query(Digest).filter_by(id=digest_id).first()
    if d:
        d.status = status
        db.flush()


def ensure_delivery(db: Session, digest_id, provider: str = "smtp") -> Delivery:
    """Idempotent: return existing pending/sending delivery for digest, else create."""
    existing = db.query(Delivery).filter_by(digest_id=digest_id, status="pending").first()
    if existing:
        return existing
    # also check sending to prevent duplicate send after restart
    sending = db.query(Delivery).filter_by(digest_id=digest_id, status="sending").first()
    if sending:
        return sending
    delivery = Delivery(digest_id=digest_id, provider=provider, status="pending")
    db.add(delivery)
    db.flush()
    return delivery
