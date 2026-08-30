# Personal Intelligence Platform

> **Sources can change. Models can change. News cycles can change. The intelligence graph must survive all of them.**

An adaptive, evidence-grounded intelligence platform that discovers, verifies, clusters, ranks, explains, remembers, and delivers information according to your evolving interests. The newsletter is *one delivery surface* — the same intelligence powers web, API, and Ask.

**North Star KPI:** Useful information delivered per unit of user attention.

---

## What It Does

- **Acquisition:** Parallel fetch from Arxiv, OpenAlex, Semantic Scholar, Crossref, Hacker News, GitHub (releases/trending), Hugging Face, Tavily/News, Reddit/social, RSS — all behind `SourceConnector` adapters (provider-swappable).
- **Intelligence:** Canonical URL + fingerprint identity, story clustering (7 docs → 1 story, content-level not URL), event extraction (18 types), entity KG, claims + evidence + contradiction detection (50% vs 17% → explains evaluation difference).
- **Ranking:** Composite `final_score` (semantic + freshness decay 72h + novelty + source quality A-D + event importance + trend + user affinity + info gain - penalties + diversity) with two-stage reranker (cheap → Cohere, 70/30 blend) and quality gate allowing **0-N stories** (no filler, `digest_skipped_low_signal` is valid).
- **Personalization:** Long (90d) / recent (14d) / session (2h) decay, 12 feedback actions (SAVE/LIKE/CLICK/MUTE...), affinity nudge `0.85*orig + 0.15*aff` — email has `Save / More / Less / Mute / Follow / Ask` per story.
- **Delivery:** Idempotent digest lifecycle (`draft → review → approved → scheduled → sent → failed → archived`) with human-in-the-loop checkpoint resume; email via `EmailProvider` (SMTP + Resend API) with webhook events (delivered/opened/clicked/bounced/complained) feeding back into personalization.
- **Product:** FastAPI platform — dashboard (`/`), story/topic/entity pages, search (hybrid), `POST /ask` grounded QA (cites stored corpus, never model memory), saved/following, admin health + ranking explainer.

---

## Quick Start (Phase 7, `main2` branch)

```bash
# 1. env
cp .env.example .env  # fill TAVILY_API_KEY, GROQ_API_KEY, SMTP_USER/PASS, RESEND_API_KEY optional

# 2. install (Python 3.12)
pip install -r requirements.txt  # or pip install -e ".[dev]"

# 3. db (Postgres primary, SQLite fallback for dev)
alembic upgrade head
python scripts/seed_demo.py          # demo@example.com from config/profile.json
python scripts/migrate_legacy_data.py  # profile + history.db + schedules

# 4. run API + dashboard
uvicorn app.api.app:create_app --factory --reload  # http://localhost:8000/ and /docs

# 5. CLI (new entrypoint)
python -m app.cli run --dry-run
python -m app.main   # legacy main.py shim still works (delegates to app.cli)
```

**Old entrypoint still works:** `python main.py` delegates to `app/main.py` shim for backwards compat.

---

## Project Structure

```
app/
  api/ (FastAPI factory + 9 routers: digests/stories/search/ask/feedback/users/entities/topics/admin)
  core/ (config pycSettings, logging, security SSRF guard, providers ModelGateway, metrics, feature_flags)
  domain/ (users/documents/stories/events/entities/topics/ranking/personalization/trends/digests)
  graph/ (state, workflows/subgraphs, nodes: normalize/identity/cluster/events/entities/source_quality/claims/ranking/personalize/quality_gate/digest/delivery)
  connectors/ (base Protocol + arxiv/openalex/semantic_scholar/crossref/github/huggingface/rss/news/hackernews/reddit/regulation)
  intelligence/ (clustering, ranking composite, entities, claims, evidence, trends, novelty, diversity, personalization)
  providers/ (llm/groq+openai+anthropic, embeddings/mini_lm, reranking/cohere, search/tavily, email/smtp+resend)
  storage/ (db.py engine WAL/pgvector, models 15 tables, migrations, cache)
  services/ (digest, feedback, recommendation, notification)
  templates/email/ (base.html + digest.html with feedback controls)
  workers/ (ingestion/processing/delivery + adaptive send window + cost budget)
config/ (profile.json bootstrap, sources.yaml catalog, ranking.yaml weights)
scripts/ (migrate_legacy_data, seed_demo, eval_ranking)
docs/ (UPGRADE_AUDIT, ARCHITECTURE, DATA_MODEL, SOURCE_CATALOG, RANKING, PERSONALIZATION, EVALUATION, SECURITY, MIGRATION)
tests/ (unit 60+, integration test_api, evaluation)
```

See `plan.md` for the full 7-phase roadmap and `docs/*` for each layer.

---

## Configuration

All via `app/core/config.py` (pydantic-settings). Legacy `NEWSLETTER_*` vars still work:

| Var | Required | Default |
|-----|----------|---------|
| `TAVILY_API_KEY` | yes | — |
| `GROQ_API_KEY` | yes | — |
| `SMTP_USER` / `SMTP_APP_PASS` | for email | — |
| `RESEND_API_KEY` | optional (Resend API) | — |
| `DATABASE_URL` | no | `sqlite:///data/history.db` (dev), `postgresql+psycopg://...` (prod) |
| `NEWSLETTER_TIMEZONE` | no | `UTC` |
| `NEWSLETTER_DATA_DIR` | no | `data` |

`config/sources.yaml` enables/disables each connector; `config/ranking.yaml` tunes composite weights without code deploy.

---

## API (Phase 5+)

- `GET /health` — probe
- `GET /` — SSR dashboard (Your Brief / Trending / Ask)
- `GET /digests` `GET /digests/{id}` `GET /stories` `GET /stories/{id}` `GET /search?q=`
- `POST /ask {question}` → `{answer, citations[]}` grounded in stored documents
- `POST /feedback {user_id, target_type, target_id, action, context}` + `GET /feedback/save?story=`
- `GET /admin/health` `GET /admin/ranking-explain?doc_id=`

---

## Delivery Guarantees

- No pre-send commit of URLs; `ensure_delivery()` is idempotent (pending/sending dedup).
- `quality_gate` allows **0** qualified stories → `digest_skipped_low_signal` (not an error).
- Subject variants: informational/curiosity/executive/technical via `workers/delivery.pick_subject_variants()`.

---

## Testing

```bash
pytest -q                         # 68 tests (unit + integration)
pytest tests/unit/test_advanced.py -q
pytest tests/integration/test_api.py -q
alembic upgrade head && alembic downgrade -1 && alembic upgrade head
python scripts/migrate_legacy_data.py --dry-run  # idempotent
```

CI: `ruff check app tests`, `mypy --ignore-missing-imports app`, `pytest`, migration tests, security checks (SSRF/HTML/prompt injection via `tests/unit/test_security.py`).

---

## Deployment

- **Local:** `uvicorn` + `data/` SQLite.
- **GitHub Actions:** see `.github/workflows/agent_run.yml` (update to `python -m app.cli run` and remove Git `history.db` sync — now Postgres).
- **Prod:** set `DATABASE_URL=postgresql+psycopg://...` (enable `CREATE EXTENSION vector` for pgvector), `alembic upgrade head`, run API and ingestion workers separately.

---

## Docs

- `docs/UPGRADE_AUDIT.md` — pre-migration audit (KEEP/REFACTOR/REPLACE/DEPRECATE)
- `docs/ARCHITECTURE.md` — system diagram + current vs target
- `docs/DATA_MODEL.md` — 15 tables, identity, migrations
- `docs/SOURCE_CATALOG.md` — 6 active + 5 stubbed connectors
- `docs/RANKING.md` — composite + reranker + quality gate
- `docs/PERSONALIZATION.md` — decay, feedback, HITL
- `docs/EVALUATION.md` — Precision@5/NDCG@10, `scripts/eval_ranking.py`
- `docs/SECURITY.md` — SSRF/HTML/prompt injection
- `docs/MIGRATION.md` — legacy → platform

---

## Status

- **Branch:** `main2` is the integration branch (all phases merged, 7 tags: `phase-0` → `phase-7`).
- **Coverage:** Phase 0-7 feature-complete per `plan.md` Appendix E checklist (remaining: Redis cache/queue only where measured need, Postgres+pgvector in prod, full UI polish).
- **Mode:** Newsletter remains one surface; `src/` is being strangled into `app/` incrementally (no big-bang delete).

---

## License

MIT
