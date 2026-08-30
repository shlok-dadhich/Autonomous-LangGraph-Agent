"""Storage models — Personal Intelligence Platform.

All entities have id (UUID), created_at, updated_at, metadata (JSON), status.
Designed to work on both Postgres+pgvector (prod) and SQLite (dev/CI).
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.storage.db import Base
from app.storage.models.base import TimestampMixin, UUIDMixin, utcnow


# ---------------------------------------------------------------------------
# User
# ---------------------------------------------------------------------------
class User(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "users"

    email: Mapped[str] = mapped_column(String(320), unique=True, index=True)
    display_name: Mapped[str | None] = mapped_column(String(200))
    timezone: Mapped[str] = mapped_column(String(64), default="UTC")
    is_active: Mapped[bool] = mapped_column(default=True)
    meta: Mapped[dict] = mapped_column("metadata", JSON, default=dict)

    preferences: Mapped[UserPreference] = relationship(back_populates="user", cascade="all, delete-orphan", uselist=False)
    interactions: Mapped[list[UserInteraction]] = relationship(back_populates="user", cascade="all, delete-orphan")


class UserPreference(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "user_preferences"

    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), unique=True)
    explicit_topics: Mapped[list] = mapped_column(JSON, default=list)
    excluded_topics: Mapped[list] = mapped_column(JSON, default=list)
    preferred_sources: Mapped[list] = mapped_column(JSON, default=list)
    disliked_sources: Mapped[list] = mapped_column(JSON, default=list)
    preferred_depth: Mapped[str | None] = mapped_column(String(32))  # executive/technical/research etc
    preferred_frequency: Mapped[str | None] = mapped_column(String(32))  # daily/weekly
    preferred_formats: Mapped[list] = mapped_column(JSON, default=list)
    delivery_time: Mapped[str | None] = mapped_column(String(16))  # HH:MM
    timezone: Mapped[str | None] = mapped_column(String(64))
    # affinity maps
    topic_affinity: Mapped[dict] = mapped_column(JSON, default=dict)
    entity_affinity: Mapped[dict] = mapped_column(JSON, default=dict)
    source_affinity: Mapped[dict] = mapped_column(JSON, default=dict)
    # embeddings stored as JSON list (pgvector Vector type when postgres)
    semantic_interest_vector: Mapped[list | None] = mapped_column(JSON, nullable=True)
    reading_time_patterns: Mapped[dict] = mapped_column(JSON, default=dict)
    recent_interest_shift: Mapped[dict] = mapped_column(JSON, default=dict)
    meta: Mapped[dict] = mapped_column("metadata", JSON, default=dict)

    user: Mapped[User] = relationship(back_populates="preferences")


class UserInteraction(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "user_interactions"
    __table_args__ = (Index("ix_interactions_user_target", "user_id", "target_type", "target_id"),)

    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    target_type: Mapped[str] = mapped_column(String(32))  # story/document/entity/topic
    target_id: Mapped[str] = mapped_column(String(128))
    action: Mapped[str] = mapped_column(String(32))  # OPEN/CLICK/SAVE/LIKE/DISLIKE/HIDE/MUTE_TOPIC/FOLLOW/UNFOLLOW/SKIP/SHARE/READ_DURATION/SEARCH
    context: Mapped[dict] = mapped_column(JSON, default=dict)

    user: Mapped[User] = relationship(back_populates="interactions")


# ---------------------------------------------------------------------------
# Source
# ---------------------------------------------------------------------------
class Source(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "sources"

    name: Mapped[str] = mapped_column(String(128), unique=True)
    source_type: Mapped[str] = mapped_column(String(32), default="UNKNOWN")  # PRIMARY/SECONDARY/COMMUNITY/UNKNOWN
    adapter: Mapped[str | None] = mapped_column(String(128))
    enabled: Mapped[bool] = mapped_column(default=True)
    rate_limit: Mapped[int | None] = mapped_column(Integer)
    reliability_score: Mapped[float | None] = mapped_column(Float)
    last_success: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_failure: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    failure_count: Mapped[int] = mapped_column(default=0)
    tier: Mapped[str | None] = mapped_column(String(8))  # A/B/C/D
    meta: Mapped[dict] = mapped_column("metadata", JSON, default=dict)


# ---------------------------------------------------------------------------
# Document
# ---------------------------------------------------------------------------
class Document(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("canonical_url", name="uq_document_canonical_url"),
        Index("ix_document_content_hash", "content_hash"),
        Index("ix_document_source", "source_id"),
    )

    canonical_url: Mapped[str] = mapped_column(Text)
    original_url: Mapped[str] = mapped_column(Text)
    title: Mapped[str] = mapped_column(Text)
    source_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("sources.id"))
    publisher: Mapped[str | None] = mapped_column(String(256))
    author: Mapped[str | None] = mapped_column(String(256))
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    fetched_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    language: Mapped[str | None] = mapped_column(String(16), default="en")
    summary: Mapped[str | None] = mapped_column(Text)
    text: Mapped[str | None] = mapped_column(Text)
    content_hash: Mapped[str | None] = mapped_column(String(64), index=True)
    title_hash: Mapped[str | None] = mapped_column(String(64))
    external_id: Mapped[str | None] = mapped_column(String(256))
    topics: Mapped[list] = mapped_column(JSON, default=list)
    entities: Mapped[list] = mapped_column(JSON, default=list)
    source_tier: Mapped[str | None] = mapped_column(String(8))
    status: Mapped[str] = mapped_column(String(32), default="active")
    meta: Mapped[dict] = mapped_column("metadata", JSON, default=dict)


# ---------------------------------------------------------------------------
# StoryCluster
# ---------------------------------------------------------------------------
class StoryCluster(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "story_clusters"

    title: Mapped[str] = mapped_column(Text)
    summary: Mapped[str | None] = mapped_column(Text)
    why_it_matters: Mapped[str | None] = mapped_column(Text)
    confidence: Mapped[float | None] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(32), default="active")
    cluster_reason: Mapped[str | None] = mapped_column(Text)
    cluster_confidence: Mapped[float | None] = mapped_column(Float)
    document_ids: Mapped[list] = mapped_column(JSON, default=list)
    event_ids: Mapped[list] = mapped_column(JSON, default=list)
    entity_ids: Mapped[list] = mapped_column(JSON, default=list)
    meta: Mapped[dict] = mapped_column("metadata", JSON, default=dict)


# ---------------------------------------------------------------------------
# Event / Entity / Claim
# ---------------------------------------------------------------------------
class Event(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "events"

    event_type: Mapped[str] = mapped_column(String(64))  # 18 types
    event_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    entities: Mapped[dict] = mapped_column(JSON, default=dict)
    claims: Mapped[list] = mapped_column(JSON, default=list)
    confidence: Mapped[float | None] = mapped_column(Float)
    source_ids: Mapped[list] = mapped_column(JSON, default=list)
    story_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("story_clusters.id"))
    status: Mapped[str] = mapped_column(String(32), default="active")
    meta: Mapped[dict] = mapped_column("metadata", JSON, default=dict)


class Entity(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "entities"
    __table_args__ = (Index("ix_entity_kind_name", "kind", "canonical_name"),)

    kind: Mapped[str] = mapped_column(String(64))  # Company/Person/Model etc
    canonical_name: Mapped[str] = mapped_column(String(256))
    aliases: Mapped[list] = mapped_column(JSON, default=list)
    relations: Mapped[list] = mapped_column(JSON, default=list)
    meta: Mapped[dict] = mapped_column("metadata", JSON, default=dict)


class Claim(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "claims"

    text: Mapped[str] = mapped_column(Text)
    claim_type: Mapped[str | None] = mapped_column(String(64))
    confidence: Mapped[float | None] = mapped_column(Float)
    evidence_refs: Mapped[list] = mapped_column(JSON, default=list)  # [{document_id, span, url}]
    story_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("story_clusters.id"))
    meta: Mapped[dict] = mapped_column("metadata", JSON, default=dict)


# ---------------------------------------------------------------------------
# Digest / Delivery
# ---------------------------------------------------------------------------
class Digest(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "digests"

    user_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("users.id"))
    status: Mapped[str] = mapped_column(String(32), default="draft")  # draft/review/approved/scheduled/sent/failed/archived
    subject_variants: Mapped[dict] = mapped_column(JSON, default=dict)
    story_ids: Mapped[list] = mapped_column(JSON, default=list)
    rendered_html: Mapped[str | None] = mapped_column(Text)
    quality_score: Mapped[float | None] = mapped_column(Float)
    meta: Mapped[dict] = mapped_column("metadata", JSON, default=dict)


class Delivery(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "deliveries"

    digest_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("digests.id"))
    provider: Mapped[str | None] = mapped_column(String(64))
    message_id: Mapped[str | None] = mapped_column(String(256))
    status: Mapped[str] = mapped_column(String(32), default="pending")  # pending/sending/sent/failed/retrying
    sent_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    meta: Mapped[dict] = mapped_column("metadata", JSON, default=dict)


class DeliveryEvent(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "delivery_events"

    delivery_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("deliveries.id", ondelete="CASCADE"))
    event_type: Mapped[str] = mapped_column(String(32))  # delivered/opened/clicked/bounced/complained/suppressed/failed
    payload: Mapped[dict] = mapped_column(JSON, default=dict)


class SavedItem(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "saved_items"
    __table_args__ = (UniqueConstraint("user_id", "story_id", name="uq_saved_user_story"),)

    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    story_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("story_clusters.id", ondelete="CASCADE"))
    saved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class TrendObservation(Base, UUIDMixin, TimestampMixin):
    __tablename__ = "trend_observations"
    __table_args__ = (Index("ix_trend_target_ts", "target_id", "observed_at"),)

    target_id: Mapped[str] = mapped_column(String(128))  # topic or entity id
    target_type: Mapped[str] = mapped_column(String(32))  # topic/entity
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    mentions: Mapped[int] = mapped_column(Integer, default=0)
    unique_sources: Mapped[int] = mapped_column(Integer, default=0)
    velocity: Mapped[float | None] = mapped_column(Float)
    state: Mapped[str | None] = mapped_column(String(32))  # RISING/ACCELERATING etc
    meta: Mapped[dict] = mapped_column("metadata", JSON, default=dict)


__all__ = [
    "User",
    "UserPreference",
    "UserInteraction",
    "Source",
    "Document",
    "StoryCluster",
    "Event",
    "Entity",
    "Claim",
    "Digest",
    "Delivery",
    "DeliveryEvent",
    "SavedItem",
    "TrendObservation",
]
