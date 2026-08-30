# Upgrade Audit — Autonomous-LangGraph-Agent (as of 2026-08-30)

**Auditor:** automated repo inspection + manual review
**Branch:** `feat/phase-0-stabilize`
**Scope:** full repo: entrypoint, config, graph, state, DB, ranker, writer, email, telegram, scheduler, templates, tools, tests, deployment.

---

## 1. Executive Summary

The project is a **working autonomous newsletter script** with a solid LangGraph skeleton. It reliably fetches from 6 sources in parallel, merges, URL-dedupes, semantic-ranks with MiniLM, applies a fallback to guarantee ≥3 items, writes via Groq, and delivers via SMTP with Telegram alerts and SQLite persistence. Deployment is GitHub Actions with Git-committed `history.db`.

**Strengths to preserve:** parallel fan-out, retry/backoff, checkpointing, modular source clients, Jinja2 templating, operational alerts.

**Gaps that block productization:** single-threshold ranking, URL-only identity, forced minimum article count, LLM summarizing too early without evidence, static `profile.json` as source of truth, provider coupling (Groq/Tavily/SMTP), Git-based DB sync, no multi-user / feedback / knowledge persistence, no story clustering, no evaluation.

**Decision:** **EVOLVE, not rewrite.** Keep the LangGraph orchestration and source clients behind adapters; strangle `src/` into `app/` incrementally.

---

## 2. Repository Map

### 2.1 Entrypoints

| Path | Role | Notes |
|------|------|-------|
| `main.py:1` | Entrypoint | `setup_logging`, `system_check`, `run_pipeline_once`, `run_research_phase` (`research_graph.invoke`) |
| `src/graph/blueprint.py:80` | Graph factory | `build_research_graph()` + `build_fanout_blueprint()`; `checkpointer = SqliteSaver(_checkpoint_conn)` |
| `src/core/scheduler.py:17` | Scheduling | `WorkerScheduler` with `BlockingScheduler`, cron + interval triggers, `coalesce/misfire_grace_time` |

### 2.2 Configuration

| File | Content | Mode |
|------|---------|------|
| `config/profile.json` | `mode`, `trusted_domains`, `topics`, `keywords`, `sources{arxiv,tavily,reddit,hackernews,huggingface,rss}` | bootstrap; currently **source of truth** |
| `config/profile_news.json` | alt profile for general_news | secondary |
| `config/schedules.json` | `schedules[{id, profile_path, interval_days}]` | multi-profile, optional |
| `.env.example` | `TAVILY_API_KEY, GROQ_API_KEY, SMTP_*, TELEGRAM_*, NEWSLETTER_DATA_DIR, GROQ_MODEL` | secrets |

`profile.json:2` uses `mode=ai_research`, empty `trusted_domains`, 6 topics + 13 keywords. Source config is per-source `enabled/max_results` dials.

### 2.3 LangGraph Workflow

**Topology (`blueprint.py:80`):**
```
START → {research_arxiv_node, research_web_node, research_hf_node, research_rss_node} (parallel)
      → merge_node → deduplicate_node → filter_node --conditional--> fallback_search_node --→ writer_node → delivery_node → END
                                          └─(≥3 items)──────────────→ writer_node
```
- Fan-out is 4 nodes; `research_web_node` itself fans out internally to Tavily + Social Signals + HN via `ThreadPoolExecutor(3)` (`nodes.py:519`).
- Checkpointing: `langgraph-checkpoint-sqlite` at `data/checkpoints.db`, WAL, `timeout=30` (`database.py:28`).
- Housekeeping: monthly purge `days_to_keep=30` (`database.py:115`, `main.py:199`).

**State (`state.py:12`):**
```python
class GraphState(TypedDict):
    interest_profile: dict
    profile: dict              # alias
    mode: str
    trusted_domains: List[str]
    raw_articles: Annotated[list, operator.add]  # parallel merge
    unique_articles: list
    filtered_articles: list
    email_draft_content: list
    email_html_content: Optional[str]
    sent_article_ids: List[str]
    thread_id: Optional[str]
    logs: Annotated[list, operator.add]
    error: Optional[str]
```
- TypedDict, not Pydantic — no validation.
- Dual `interest_profile`/`profile` alias is tech debt (compat shim).

**Edges (`edges.py:6`):** `check_content_threshold` — `<3 → fallback_search_node else writer_node`.

### 2.4 Database

**`src/core/database.py:1`:**
- `DATA_DIR = Path(os.getenv("NEWSLETTER_DATA_DIR", "data"))`
- `sent_articles(url TEXT PK, sent_at TIMESTAMP)` — **only** URL history.
- `checkpoints.db` managed by `langgraph-checkpoint-sqlite`; `purge_old_checkpoints` handles both `timestamp`/`ts` column variants (sqlite schema compat).
- WAL + `synchronous=NORMAL` + `busy_timeout=5000`.
- No users, no documents, no stories, no embeddings, no migrations.

### 2.5 Source Clients (`src/tools/`)

| Client | Fetch function | Input | Output | Notes |
|--------|---------------|-------|--------|-------|
| `arxiv_client.py` | `fetch_arxiv_papers(categories, days_back, max_results)` | RSS feed | `{title, url, description, source, published_date}` | cs.AI/cs.LG, 7d lookback |
| `tavily_client.py` | `fetch_tavily_results(interest_profile, max_results)` | profile → complex query | same | dynamic `basic→advanced` if <3 results |
| `social_signal_client.py` | `fetch_social_signals(interest_profile, max_results)` | Tavily filtered to Reddit/HF/HN domains | same | subreddits MachineLearning/LocalLLaMA |
| `hn_client.py` | `fetch_hn_stories(interest_profile, min_score, max_items)` | HN Firebase | same incl `score` | ThreadPool parallel, score≥50 |
| `hf_client.py` | `fetch_hf_daily_papers(limit)` | `huggingface.co/api/daily_papers` | same incl `relevance_score=0.8` | public API, no auth |
| `rss_client.py` | `fetch_rss_sources(feed_specs)` / `fetch_rss_feeds(feed_urls)` | feed specs / URL list | same | html parser + feedparser; defaults: anthropic-newsroom, huggingface-blog |

All clients already normalized to `{title, url, description, source, relevance_score?, published_date?}` but `relevance_score` semantics vary per client (bug risk).

### 2.6 Ranking

**`src/core/ranker.py:14`:** `RelevanceRanker(model_name="all-MiniLM-L6-v2")`
- `score_articles(profile_text, unique_articles)` — lazy-loads `SentenceTransformer(cpu)`, cosine `profile vs article(title+description)`.
- `prune_similar_articles(threshold=0.9)` — pairwise cosine, keep highest-score per cluster.
- Called in `filter_node` with `threshold 0.45 (ai_research) / 0.30 (general_news)` + diversity floor `0.40/0.25` + `max_filtered_articles=6`.
- Also used in `deduplicate` via `_prune_by_similarity_with_source_preference` which prefers diversity sources (hf/anthropic/reddit/tavily) on collision.

**Weakness:** single scalar, no freshness/novelty/authority/trend/user-affinity/diversity/fatigue signals; thresholds are arbitrary.

### 2.7 Writing

**`src/core/writer.py:26`:** `NewsletterWriter(GROQ_MODEL, batch_size=3, max_retries=2)`
- System prompt: "Senior AI Research Lead", JSON-only.
- Input: `title+url+source+description` per article (no full text, no evidence package).
- Output: `{title, url, source, relevance_score, what, how, personalized_insight}` where `what/how` are single sentences, `personalized_insight` prefixed `**Personalized Insight:**`.
- Batch path (`_call_batch`) + fallback to per-article (`_call_single`) + heuristic fallback (`_fallback_enrichment`).
- **Coupling:** direct `from groq import Groq` inside `_invoke_groq:214`; no interface.

### 2.8 Email & Templates

- `email_service.py:16` — `EmailService` via `smtplib` (STARTTLS/465 SSL), `send_newsletter` returns bool, logs to `graph_state`.
- `template_service.py:12` — `TemplateService` reads `src/templates/email_body.html` (Jinja2 `Template`, not `Environment`), injects UTM params (`utm_source/medium/campaign/content/term`) per article via `_tracking_url:20`.
- `telegram_bot.py:10` — `TelegramAlertService` via `requests.post` to Bot API; `send_success_notification`/`send_error`.

### 2.9 Scheduler & Deployment

- `scheduler.py:17` — `WorkerScheduler` supports multiple `CronTrigger`/`IntervalTrigger` from `schedules.json`; default `Mon 08:00 UTC`; monthly housekeeping `day=1 00:00`.
- `main.py:199` — fallback to `run_pipeline_once` (single run) for GitHub Actions; previous `BlockingScheduler` path still present in `scheduler.py:151` but `main.py:273` now runs single-shot.
- `.github/workflows/agent_run.yml:3` — cron `0 8 * * 1,4` (Mon/Thu 08:00 UTC) + `workflow_dispatch`; steps: checkout → setup-python 3.12 → `pip install -r requirements.txt` → `python main.py` → Git auto-commit `data/history.db` with `pull --rebase && push`.

### 2.10 Tests

- `tests/test_pipeline_fixes.py:1` — 8 tests: rss sources include anthropic+hf, filter keeps diversity+dedupes, fallback relaxes threshold before broad search, writer preserves original URL + tracking links, DB WAL, general_news skips arxiv/hf, ai_research skips rss, Tavily domain guardrail.
- No integration tests for graph end-to-end, no DB migration tests, no security tests. Coverage is narrow but regression-critical paths are pinned.

### 2.11 Dependencies

```
langgraph, langgraph-checkpoint-sqlite, loguru, python-dotenv, feedparser,
tavily-python, requests, sentence-transformers, groq, jinja2, apscheduler, pytest
```
- No `pydantic`, `fastapi`, `sqlalchemy`, `alembic`, `psycopg`, `pgvector`, `httpx` yet — all needed for Phase 1+.
- No pinned versions — drift risk. Phase 0.3 will pin.

---

## 3. Component Classification

| Component | Verdict | Rationale | Migration note |
|-----------|---------|-----------|----------------|
| LangGraph orchestration + checkpointing | **KEEP** | Durable, streaming, HITL-ready per LangGraph docs | Swap `SqliteSaver`→`PostgresSaver` in Phase 1; keep SQLite for dev |
| Parallel source fan-out | **KEEP** | Correct; `research_web_node` internal parallelism is sound | Expose as explicit subgraph nodes in Phase 2 |
| Source clients (arxiv/tavily/hn/hf/rss/social) | **REFACTOR** | Working but ad-hoc return shapes, inconsistent `relevance_score` | Wrap behind `SourceConnector` protocol + `RawDocument` in Phase 1 |
| `GraphState` TypedDict | **REPLACE** | No validation, dual alias, no IDs | Pydantic `GraphState` with `document_id/story_id/event_id` in Phase 1 |
| `database.py` sent_articles | **REFACTOR** | Minimal, WAL correct, but single table | Rename to `app/storage/`, add models + Alembic |
| `ranker.py` MiniLM | **REFACTOR** | Keep as Stage-1 retriever; add Stage-2 reranker | New `app/intelligence/ranking.py` composite scorer |
| `writer.py` Groq writer | **REPLACE** | Coupled, hallucinates without evidence | `LLMProvider` gateway + evidence-grounded schemas |
| `reliability.py` safe_execute | **REPLACE** | Simple retry; no circuit breaker, no rate-limit awareness | Source-aware breaker in Phase 7 |
| `scheduler.py` WorkerScheduler | **REFACTOR** | Supports multi-profile but no per-user cron, no adaptive window | Move to `app/workers/` + DB-backed schedules |
| `email_service.py` SMTP | **REFACTOR** | Works but single provider | `EmailProvider` with SMTP + Resend |
| `telegram_bot.py` | **KEEP** | Ops channel | Keep, but not primary product surface |
| `template_service.py` + `email_body.html` | **KEEP** | UTM tagging + Jinja2 | Redesign template per Phase 5 spec |
| `main.py` entrypoint | **REFACTOR** | Mixed concerns (logging+check+run) | Split into `app/main.py` + `app/cli.py` + `app/core/logging.py` |
| `config/profile.json` | **DEPRECATE** (as source of truth) | Static, no learning | Demote to bootstrap seed after Phase 1 |
| Git-committed `data/history.db` | **DEPRECATE** | Git sync for mutable DB is not prod-safe | Postgres is prod; keep SQLite for dev only |
| `requirements.txt` unpinned | **REPLACE** | Drift | `pyproject.toml` with pins + hashes in Phase 0.3 |

---

## 4. Technical Debt Inventory

1. **Dual profile keys** — `interest_profile` vs `profile` alias threaded through every node (`nodes.py:186`).
2. **Inconsistent article schema** — some sources set `relevance_score`, others don't; `published_date` sometimes `utcnow()` approximation.
3. **Hard thresholds** — `filter_node:910` uses `0.45/0.30`, `fallback:989` uses `0.35`; no learning.
4. **Fallback broad search** — `tavily_fallback` query `"latest important AI breakthroughs"` is generic filler.
5. **LLM token waste** — per-article fallback loop with no candidate reduction before LLM.
6. **No canonical URL** — `already_sent` checks are exact URL string matches; UTM/syndication duplicates slip through.
7. **No content hash** — redupes rely on embedding similarity which is expensive and late.
8. **No typed errors** — all failures collapse to string in `state["error"]`.
9. **Checkpointer lifecycle** — global `checkpointer` at import time (`blueprint.py:29`) prevents env-aware config.

---

## 5. Failure Modes

| Failure | Current behavior | Desired |
|---------|-----------------|---------|
| Single source API down (Tavily/Arxiv/HF) | `safe_execute` returns `[]`; pipeline continues via `merge_node` but logs only | Circuit breaker, degraded banner, health metric |
| All sources empty | `merge_node:791` sets `error="Zero articles harvested"` → writer skipped → delivery fails | `digest_skipped_low_signal` with reason, not error |
| Filter yields 0 items | `fallback_search_node` relaxes threshold then fires generic Tavily query | Quality gate decides `0-N`; no filler |
| Groq 429/timeout | `_invoke_groq:219` retries `2^attempt`; then `_fallback_enrichment` heuristic | Provider fallback (OpenAI/Anthropic) via gateway |
| SMTP failure | `delivery_node:1199` logs + `AlertService.send_error`, URLs **not** committed (correct) | Persist digest as `failed/retrying`, idempotent resend |
| Checkpoint DB corrupt | `create_sqlite_connection` opens anyway | Health check + auto-reinit |
| Git push conflict on `history.db` | `git pull --rebase && push` in workflow — can silently overwrite | Remove after Phase 1; use Postgres |
| Long article list (100+) | `filter_node` embeds all at once → OOM risk on small runners | Batch embeds + candidate reduction |

---

## 6. Provider Lock-in Assessment

| Provider | Coupling | Risk | Fix |
|----------|----------|------|-----|
| Groq | `from groq import Groq` inside writer | Model churn forces rewrite | `LLMProvider` protocol + routing |
| Tavily | `TavilyClient` + `@safe_execute` in tools + nodes | Search vendor switch = 4 files | `SearchProvider` abstraction |
| SMTP (Gmail) | `EmailService` with `SMTP_USER/APP_PASS` | Deliverability + port 587 fragility | `EmailProvider` + Resend API |
| SentenceTransformers | `RelevanceRanker` instantiates model inline | Model swap = 3 files | `EmbeddingProvider` |
| SQLite | `history.db`/`checkpoints.db` paths hardcoded | No multi-user, no vector | `DATABASE_URL` with Postgres primary |

---

## 7. Security Risks

| # | Risk | Location | Severity | Mitigation (phase) |
|---|------|----------|----------|-------------------|
| S1 | SSRF via article `url` fetch | `rss_client`, future fetchers | High | Allow-list, no private-IP, timeout + size caps (Phase 1 → 7) |
| S2 | Prompt injection — article text treated as instruction | `writer.py` concatenates description into LLM prompt | High | Frame as `UNTRUSTED_SOURCE_CONTENT`, schema-validated output (Phase 2) |
| S3 | HTML injection — article `title/description` rendered into email | `email_body.html` | Medium | Jinja2 auto-esc + sanitize (Phase 5) |
| S4 | Secrets in env not rotated/validated | `main.py:243` checks presence only | Low | `pydantic-settings` + startup validation (Phase 0.5) |
| S5 | Unbounded content size | `tavily_client` `content[:500]` is truncated but not validated downstream | Medium | Size + hash guard in `normalize` (Phase 2) |
| S6 | Poisoned documents (adversarial) | No filtering | Medium | Source reputation scoring (Phase 2) |

---

## 8. Scalability Risks

- **Memory:** `sentence-transformers` loaded per-node invocation (`ranker.py:43`, `nodes.py:241`) with `gc.collect()` — reload cost high, concurrent runs contended.
- **No pagination / streaming:** all articles in-memory in `GraphState` lists.
- **SQLite WAL** is fine for 1-user, fails for concurrent web requests.
- **APScheduler BlockingScheduler** blocks process; not suitable for FastAPI ASGI.
- **No caching:** identical `profile_text` re-embedded every run.

---

## 9. Data Quality Risks

- `published_date = datetime.utcnow().isoformat()` in `tavily_client:147` masquerades fetch time as publish time → freshness score will be wrong. Same for social/HN approximations.
- No language detection; non-English articles not filtered.
- `relevance_score` overloaded: sometimes search relevance (0.8 default), sometimes computed cosine — conflated downstream.
- No content hash → syndicated articles with tracking params counted as distinct.

---

## 10. Non-Negotiable Rules (from plan.md §1)

All Phase 1+ work must satisfy: provider interfaces, no hard threshold as final ranker, no forced minimum, evidence-grounded claims, idempotent delivery, typed schemas, structured logging, tests per feature.

---

## 11. Recommended Next Steps (Phase 0 remainder)

1. `pyproject.toml` + `ruff`/`mypy`/`pytest` + pre-commit (0.3)
2. Baseline tests for identity/canonicalize/hash/ranker (0.4)
3. Typed settings extracting `NEWSLETTER_*` env (0.5)
4. `ARCHITECTURE.md` skeleton (0.6)
5. Tag `phase-0-audit`, merge to `main` only after all checks green.

---

## 12. Sign-off

This audit is the baseline. Every subsequent phase must reference it when claiming "KEEP/REFACTOR/REPLACE/DEPRECATE" and when measuring duplicate-rate, ranking, and citation improvements against the pre-upgrade behavior.

