# Data Model — Personal Intelligence Platform

**Status:** Phase 1 (foundation). All entities have `id (UUID), created_at, updated_at, metadata (JSON), status` unless noted.
**Storage:** Postgres + pgvector primary, SQLite fallback for dev. Migrations via Alembic (`app/storage/migrations/versions/18ef7cc87060`).

## 1. Entity Overview

| Entity | Table | Purpose |
|--------|-------|---------|
| User | `users` | Multi-user support; demo user seeded from `profile.json` |
| UserPreference | `user_preferences` | Topics/affinities/vectors per user |
| UserInteraction | `user_interactions` | Feedback stream (OPEN/CLICK/SAVE/LIKE/DISLIKE/HIDE/MUTE_TOPIC/FOLLOW/UNFOLLOW/SKIP/SHARE/READ_DURATION/SEARCH) |
| Source | `sources` | Catalog of adapters, health, tier A-D |
| Document | `documents` | Normalized content with canonical identity |
| StoryCluster | `story_clusters` | Event-level dedup — one story = many documents |
| Event | `events` | 18 types (MODEL_RELEASE, PAPER_RELEASE, etc.) |
| Entity | `entities` | KG nodes (Company/Person/Model/Repo etc.) |
| Claim | `claims` | Evidence-grounded statements |
| Digest | `digests` | Newsletter draft with lifecycle |
| Delivery / DeliveryEvent | `deliveries`, `delivery_events` | Idempotent send + webhook events |
| TrendObservation | `trend_observations` | Time-series for trend engine |

## 2. Document Identity (Phase 1.7)

- `canonical_url` vs `original_url` preserved.
- `canonicalize_url()` strips `utm_*`, `gclid`, `fbclid`, lowercases host, strips `www.`, sorts query, trims trailing slash.
- `content_hash = sha256(normalized_text)` and `title_hash = sha256(lower normalized title)` for fast pre-embedding dedup.
- `document_id = uuid5(NAMESPACE_URL, canonical_url + "#" + content_hash[:16])` — stable, no DB lookup needed for dedup.

## 3. Key Schemas (SQLAlchemy)

### User + Preference
- `users.email unique`, `display_name`, `timezone`, `is_active`
- `user_preferences.user_id unique`, `explicit_topics JSON`, `excluded_topics`, `preferred_sources`, `topic_affinity JSON`, `semantic_interest_vector JSON` (Vector when pgvector), `reading_time_patterns`, `recent_interest_shift`

### Document
- `canonical_url unique`, `original_url`, `title`, `source_id FK`, `publisher`, `author`, `published_at`, `fetched_at`, `language`, `summary`, `text`, `content_hash indexed`, `title_hash`, `external_id`, `topics JSON`, `entities JSON`, `source_tier A-D`, `status`

### StoryCluster
- `title`, `summary`, `why_it_matters`, `confidence`, `cluster_reason`, `cluster_confidence`, `document_ids JSON`, `event_ids JSON`, `entity_ids JSON`

### Digest / Delivery
- `digests.status: draft|review|approved|scheduled|sent|failed|archived`, `subject_variants JSON`, `story_ids JSON`, `rendered_html`, `quality_score`
- `deliveries.status: pending|sending|sent|failed|retrying`, `message_id`, `provider`, `sent_at`
- `delivery_events.event_type: delivered|opened|clicked|bounced|complained|suppressed|failed`

## 4. Migrations

- `alembic.ini` points to `app/storage/migrations` and uses `resolved_database_url` (SQLite fallback).
- `env.py` enables `CREATE EXTENSION IF NOT EXISTS vector` on Postgres when available.
- Initial migration `18ef7cc87060` creates all tables and drops legacy `sent_articles` (if existed) — legacy URLs migrated via `scripts/migrate_legacy_data.py` before drop.

## 5. DB Helpers

- `app/storage/db.py`: `get_engine()`, `init_db()`, `SessionLocal`, `get_session()` (FastAPI dep), SQLite WAL pragmas.
- `app/storage/models/base.py`: `TimestampMixin`, `UUIDMixin`, `utcnow()`.

## 6. Seeding

- `scripts/seed_demo.py` and `app/domain/users.seed_user_from_profile()` — idempotent import from `config/profile.json`.
- `scripts/migrate_legacy_data.py` handles `profile.json` → User, `history.db sent_articles` → Document (archived), `schedules.json` → preserved (DB schedules in Phase 4).

## 7. Next (Phase 2)

Add `pgvector` Vector column for embeddings, claim evidence spans, and graph relations. Ranking will read `Document.content_hash` before embedding.
