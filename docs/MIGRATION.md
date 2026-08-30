# Migration — Legacy → Platform

**From:** `main` (SQLite `history.db` + `profile.json` + `schedules.json`)
**To:** `main2` (Postgres+pgvector primary, SQLite fallback, `User`/`Document`/`Digest`)

## Steps

1. `alembic upgrade head` — creates 15 tables; drops legacy `sent_articles` only after backup check.
2. `python scripts/migrate_legacy_data.py` — idempotent:
   - `config/profile.json` → `demo@example.com` User + UserPreference
   - `config/profile_news.json` → `news@example.com`
   - `data/history.db:sent_articles` → `documents` (archived, canonicalized) if table still exists
   - `config/schedules.json` → logged (DB schedules in Phase 7 via `Digests.schedule` — TODO)
3. `python scripts/seed_demo.py` — minimal demo user.

## Backwards Compatibility

- `DATABASE_URL` unset → SQLite `data/history.db` (dev/CI).
- `profile.json` remains bootstrap seed; DB is source of truth after Phase 1.
- GitHub Actions workflow should switch to `python -m app.cli run --profile config/profile.json` (DB-backed).
- `src/` remains operasional during strangulation; `app/` is canonical.

## Rollback

- `alembic downgrade -1` restores `sent_articles`.
- `main` branch retains pre-migration state; `main2` is integration branch.
