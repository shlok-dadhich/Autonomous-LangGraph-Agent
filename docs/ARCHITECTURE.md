# Architecture — Personal Intelligence Platform

**Status:** Phase 0 skeleton — evolves each phase. See `plan.md` for full roadmap.
**Last updated:** 2026-08-30 (Phase 0)

---

## 1. Vision

From `Automated newsletter generator (run-centric)` to
`Personal Intelligence Platform (user-centric, event-centric, feedback-driven, knowledge-centric)`.

The newsletter becomes **one delivery surface**. The intelligence engine is reusable for
email / web dashboard / Telegram / Slack / alerts / weekly reports / topic pages / entity pages / saved knowledge / Ask-Your-Intelligence / API.

**Data flow:**
```
Sources → Documents → Claims → Entities → Events → StoryClusters → Evidence → CompositeRanking → PersonalizedBrief → KnowledgeMemory → Feedback → Learning
```

---

## 2. System Diagram (Target)

```
                         ┌────────────────────┐
                         │      Web App       │
                         │ Dashboard / Reader │
                         └─────────┬──────────┘
                                   │
                         ┌─────────▼──────────┐
                         │     API Layer      │
                         │  FastAPI / REST    │
                         └─────────┬──────────┘
                                   │
                  ┌────────────────▼────────────────┐
                  │      Intelligence Platform      │
                  │  Profile / Memory / Feedback   │
                  │  Ranking / Clustering / Trends │
                  │  Knowledge Graph / Evidence    │
                  └────────────────┬───────────────┘
                                   │
                     ┌─────────────▼─────────────┐
                     │        LangGraph          │
                     │  Orchestration + HITL     │
                     └─────────────┬─────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
 ┌──────▼──────┐            ┌──────▼──────┐            ┌──────▼──────┐
 │ Acquisition │            │ Intelligence│            │   Delivery  │
 │   Layer     │            │    Layer    │            │    Layer    │
 │ RSS/Arxiv/  │            │ Clustering  │            │ Email (SMTP │
 │ GitHub/HF/  │            │ Ranking     │            │  + Resend)  │
 │ OpenAlex/   │            │ Entities    │            │ Web/Telegram│
 │ Tavily/HN/  │            │ Events      │            │ Slack/Push  │
 │ Reddit      │            │ Evidence/   │            │ Alerts      │
 └──────┬──────┘            │ Trends      │            └──────┬──────┘
        │                   └──────┬──────┘                   │
        └─────────────┬────────────┘                          │
                      │                                       │
              ┌───────▼────────┐                     ┌────────▼──────┐
              │   Postgres     │                     │  Delivery     │
              │  + pgvector    │                     │  Events Store │
              └───────┬────────┘                     └───────────────┘
                      │
              ┌───────▼────────┐
              │ Object Storage │  (optional, raw docs)
              └────────────────┘
  + Redis only for cache / rate-limit / queue / locks when measured need exists
```

---

## 3. Current (Phase 0) vs Target

| Area | Current (src/) | Target (app/) | Phase |
|------|---------------|---------------|-------|
| Entrypoint | `main.py` | `app/main.py` + `app/cli.py` | 1 |
| Config | `os.getenv` + `profile.json` as truth | `app/core/config.py` (pydantic-settings) + DB-backed prefs | 0.5 → 1 |
| Graph state | `TypedDict` with `Annotated[list,operator.add]` | Pydantic state with stable IDs (`document_id/story_id/event_id`) | 1 |
| DB | SQLite `history.db` + `checkpoints.db` via `SqliteSaver` | Postgres+pgvector primary, SQLite fallback for dev, Alembic migrations | 1 |
| Ranking | `RelevanceRanker` cosine vs threshold `0.45/0.30` + `fallback_search_node` | Composite `final_score` + two-stage reranker + quality gate (`0-N` stories) | 3 |
| Writer | Groq-coupled `NewsletterWriter` summarizing title+desc | `LLMProvider` gateway + evidence-grounded structured outputs | 1 → 2 |
| Email | `EmailService` SMTP only | `EmailProvider` (SMTP + Resend) with idempotent lifecycle | 4 |
| Sources | 6 clients in `src/tools/` | `SourceConnector` protocol + `config/sources.yaml` per-source adapter | 1 → 5 |
| Scheduler | `WorkerScheduler` (BlockingScheduler) | Per-user DB-backed schedules + adaptive delivery | 4 |
| Templates | `src/templates/email_body.html` | `app/templates/email/` with variable sections + citations | 5 |
| Observability | loguru file only | Structured logs + metrics per stage + health dashboard | 7 |

---

## 4. New File Structure (Clean)

See `plan.md §5` for canonical layout. Phase 0 creates the skeleton:

```
app/
├── __init__.py
├── core/
│   ├── config.py       ← DONE (Phase 0.5)
│   └── logging.py      ← DONE (Phase 0.5)
├── api/                ← Phase 5
├── graph/              ← Phase 1-2
├── connectors/         ← Phase 1
├── intelligence/       ← Phase 2-3
├── providers/          ← Phase 1
├── storage/            ← Phase 1 (models + migrations)
├── services/           ← Phase 4
├── templates/          ← Phase 5
└── workers/            ← Phase 4

config/
├── profile.json        ← bootstrap only after Phase 1
├── sources.yaml        ← Phase 1
└── ranking.yaml        ← Phase 3

docs/
├── UPGRADE_AUDIT.md    ← DONE
└── ARCHITECTURE.md     ← THIS FILE
```

`src/` remains operational during strangulation; `app/` becomes canonical.

---

## 5. Key Design Decisions (Phase 0)

1. **Evolve, don't rewrite.** LangGraph orchestration and source clients are kept behind adapters.
2. **Postgres+pgvector is primary, SQLite for dev.** Dual support via `DATABASE_URL` (unset → SQLite).
3. **Provider abstraction first.** No business logic imports `groq`/`tavily` directly after Phase 1.
4. **Typed settings.** `pydantic-settings` with `NEWSLETTER_*` aliases for backwards compat.
5. **Tooling baseline.** `pyproject.toml` pins deps; `ruff` + `mypy --ignore-missing-imports` + `pytest`; `pre-commit` hooks for trailing-whitespace/yaml/ruff.
6. **Baseline tests lock current helpers** (`_has_verified_url`, `_normalize_allowed_domains`, `_is_url_allowed`, `RelevanceRanker._article_text`, UTM tracking, WAL mode) so refactors are safe.

---

## 6. LangGraph Topology (Today + Next)

**Today (Phase 0):**
```
START → {arxiv, web, hf, rss} → merge → dedupe → filter --<3--> fallback → writer → delivery → END
```

**Next (Phases 1-3):**
```
START → load_user_context → plan_research → parallel_source_acquisition → normalize → identity → cluster → detect_events → extract_entities → extract_claims → evaluate_sources → rank (stage1→stage2) → personalize → diversity → quality_gate → generate_digest → verify → [human_review?] → deliver → capture_delivery_state → END
```
Smaller subgraphs for acquisition / intelligence / delivery.

---

## 7. Storage Strategy

- **Alembic** from Phase 1; every model change = migration.
- Core entities: `User, UserPreference, UserInteraction, Source, Document, StoryCluster, Event, Entity, Claim, TrendObservation, Digest, Delivery`.
- All have `id (UUID), created_at, updated_at, metadata (JSONB), status`. Stable IDs, not URL-only.
- Identity: `canonical_url` vs `original_url`, `content_hash`, `title_hash`, `external_id`.

---

## 8. Verification (Phase 0 Exit)

- `pytest -q` → 29 passed (8 legacy + 21 baseline)
- `ruff check` configured (legacy violations acknowledged; `app/` + `tests/` will be enforced from Phase 1)
- `mypy --ignore-missing-imports` → no blocking errors in `app/core`
- Settings smoke: `from app.core.config import get_settings; get_settings()` loads env correctly
- Docs: `UPGRADE_AUDIT.md` + `ARCHITECTURE.md` present

Next: `feat/phase-1-foundation` (DB + providers + identity + migration).
