# Autonomous-LangGraph-Agent → Personal Intelligence Platform — Master Implementation Plan

**Version:** 1.0 — August 30, 2026
**Status:** Ready to implement (feature-by-feature, commit-by-commit)
**Principle:** Sources can change. Models can change. News cycles can change. The intelligence graph must survive all of them.
**North Star KPI:** Useful information delivered per unit of user attention.

---

## Table of Contents

1. [How to Use This Plan](#1-how-to-use-this-plan)
2. [Current Repo Audit (Snapshot)](#2-current-repo-audit-snapshot)
3. [Product Vision Shift](#3-product-vision-shift)
4. [Target Architecture](#4-target-architecture)
5. [New File Structure (Clean)](#5-new-file-structure-clean)
6. [Data Model / Storage Strategy](#6-data-model--storage-strategy)
7. [Provider Abstraction (Anti Lock-in)](#7-provider-abstraction-anti-lock-in)
8. [Intelligence Layer — Detailed Design](#8-intelligence-layer--detailed-design)
9. [Email & Delivery Redesign](#9-email--delivery-redesign)
10. [Web App / API / Ask-Your-Intelligence](#10-web-app--api--ask-your-intelligence)
11. [Phased Roadmap (P0 → P3) with Atomic Commits](#11-phased-roadmap-p0--p3-with-atomic-commits)
12. [Branching & Commit Discipline](#12-branching--commit-discipline)
13. [Migration & Backwards Compatibility](#13-migration--backwards-compatibility)
14. [Observability, Evaluation & Cost Control](#14-observability-evaluation--cost-control)
15. [Security, Privacy & Licensing](#15-security-privacy--licensing)
16. [Docs to Produce Per Phase](#16-docs-to-produce-per-phase)
17. [Definition of Done (Product Launch Gates)](#17-definition-of-done-product-launch-gates)
18. [What NOT to Do](#18-what-not-to-do)
19. [Appendix A: Source Catalog](#appendix-a-source-catalog)
20. [Appendix B: Ranking Formula](#appendix-b-ranking-formula)
21. [Appendix C: Digest Template Spec](#appendix-c-digest-template-spec)
22. [Appendix D: Evaluation Dataset Spec](#appendix-d-evaluation-dataset-spec)
23. [Appendix E: Deliverable Checklist](#appendix-e-deliverable-checklist)

---

## 1. How to Use This Plan

- This is a **sequential implementation plan**. Do not skip phases.
- Each phase ends with tests + docs + migration before moving on.
- Each commit is **small, atomic, reviewable** (see §11).
- The newsletter is *one delivery surface* — the intelligence engine underneath is the product.
- Development stays on **`main` with feature branches**; every phase has a tagged checkpoint.
- After this plan lands, create issues from §11.1 commit list and implement one commit at a time.

---

## 2. Current Repo Audit (Snapshot)

### 2.1 What exists (keep / refactor)

| Component | File | Verdict |
|-----------|------|---------|
| Entrypoint + logging + system checks | `main.py:1` | **REFACTOR** → move to `app/main.py` + `app/core/logging.py` |
| Profile loading (json bootstrap) | `main.py:52` | **KEEP as seed** → migrates into DB |
| LangGraph orchestration | `src/graph/blueprint.py:80` | **KEEP + EVOLVE** (durable, checkpointed) |
| State (TypedDict) | `src/graph/state.py:12` | **REPLACE** with typed Pydantic state + subgraphs |
| Nodes (fetch/merge/dedupe/filter/fallback/writer/delivery) | `src/graph/nodes.py:344` | **REFACTOR** — split acquisition vs intelligence vs delivery |
| Semantic ranker (MiniLM) | `src/core/ranker.py:14` | **KEEP as Stage-1 retriever**, add Stage-2 reranker + composite scorer |
| LLM writer (Groq-coupled) | `src/core/writer.py:26` | **REPLACE** with `LLMProvider` gateway + structured outputs |
| Template (Jinja2) | `src/templates/email_body.html` | **KEEP** + redesign per §9 |
| Email SMTP | `src/services/email_service.py` | **REFACTOR** to `EmailProvider` interface |
| Telegram alerts | `src/services/telegram_bot.py` | **KEEP** as ops channel, not product surface |
| SQLite history/checkpoints | `src/core/database.py:47` | **KEEP for dev**, add Postgres+pgvector as primary |
| Scheduler (APScheduler) | `src/core/scheduler.py` | **REFACTOR** → per-user cron + adaptive delivery |
| Source clients (arxiv/tavily/hn/hf/rss/social) | `src/tools/*` | **KEEP + generalize** behind `SourceConnector` protocol |
| Reliability decorator | `src/utils/reliability.py:20` | **REPLACE** with source-aware circuit breaker |

### 2.2 Gaps (why this is still a script, not a product)

- Ranking = single cosine threshold `0.45/0.30` — no freshness/novelty/authority/trend/user-affinity.
- Dedupe = URL history + pairwise similarity — no event/story cluster.
- Forced minimum of 3 articles via fallback broad search → filler problem.
- LLM summarizes directly from title/description → no evidence package, no citation.
- Profile = static `profile.json` → no learning, no feedback, no per-user DB.
- Groq-coupled writer → no provider swap.
- Git-committed SQLite → not multi-user, not durable at scale.
- No web app, no saved/following, no ask-your-news, no trend/entity memory.

### 2.3 Risks to address early

- Provider lock-in (Groq, Tavily), URL-only identity, silent hallucination, SSRF via untrusted URLs, prompt injection from article content, cost blow-up from per-article LLM calls on junk.

> Detailed audit doc to be generated in Phase 0 as `docs/UPGRADE_AUDIT.md` — the table above is the executive summary.

---

## 3. Product Vision Shift

**From:** `Automated newsletter generator (run-centric)`
**To:** `Personal Intelligence Platform (user-centric, event-centric, feedback-driven, knowledge-centric)`

**Data flow shift:**

```
OLD: Sources → raw_articles → dedupe(URL) → score(0.45) → summarize → email

NEW: Sources → Documents → Claims → Entities → Events → StoryClusters → Evidence → CompositeRanking → PersonalizedBrief → KnowledgeMemory → Feedback → Learning
        ↘ delivery surfaces: email / web / Telegram / Slack / alerts / API / weekly report
```

**Key product decisions (non-negotiable):**

1. Fixed thresholds removed → composite configurable ranker.
2. "Minimum 3 articles" removed → quality gate decides (0-N stories; zero-send is valid).
3. `profile.json` demoted to bootstrap seed → DB is source of truth.
4. URL-only identity removed → canonical IDs + fingerprints.
5. LLM summaries require evidence refs — no naked claims.
6. Delivery has lifecycle `draft → review → approved → scheduled → sent → failed → archived`.
7. Git DB sync deprecated beyond bootstrap; Postgres is prod.

---

## 4. Target Architecture

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
 │ RSS/Arxiv   │            │ Clustering  │            │ Email (SMTP │
 │ GitHub/HF   │            │ Ranking     │            │  + API prov)│
 │ OpenAlex/   │            │ Entities    │            │ Web/Telegram│
 │ Semantic/   │            │ Events      │            │ Slack/Push  │
 │ News/Tavily │            │ Trends      │            │ Alerts      │
 │ HN/Reddit   │            │ Evidence    │            │             │
 └──────┬──────┘            └──────┬──────┘            └──────┬──────┘
        │                          │                          │
        └─────────────┬────────────┘                          │
                      │                                       │
              ┌───────▼────────┐                     ┌────────▼──────┐
              │   Postgres     │                     │  Delivery     │
              │  + pgvector    │                     │  Events Store │
              └───────┬────────┘                     └───────────────┘
                      │
              ┌───────▼────────┐
              │ Object Storage │
              │ Raw docs (opt) │
              └────────────────┘
  + Redis (only for cache / rate-limit / queue / locks) — not mandatory day 1
```

**Stack choices (version-pinned in Phase 0):**

- Python 3.12, LangGraph + `langgraph-checkpoint-postgres`, FastAPI, SQLAlchemy + Alembic, Postgres 16 + pgvector, Pydantic v2, Jinja2, pytest + httpx, ruff + mypy, loguru/structlog + OpenTelemetry.

---

## 5. New File Structure (Clean)

You have a backup. The repo will be migrated **incrementally** into this structure. Do not delete `src/` in one commit — strangle it.

```
Autonomous-LangGraph-Agent/
├── app/
│   ├── __init__.py
│   ├── main.py                     # entrypoint (replaces root main.py)
│   ├── cli.py                      # typer CLI: run / dry-run / migrate / reprocess
│   │
│   ├── api/
│   │   ├── app.py                  # FastAPI factory
│   │   ├── deps.py                 # auth, db session, rate limit
│   │   └── routes/
│   │       ├── digests.py
│   │       ├── stories.py
│   │       ├── entities.py
│   │       ├── topics.py
│   │       ├── feedback.py
│   │       ├── search.py
│   │       ├── ask.py              # Ask Your Intelligence
│   │       ├── users.py
│   │       └── admin.py
│   │   └── schemas/               # Pydantic request/response
│   │
│   ├── core/
│   │   ├── config.py              # typed settings (pydantic-settings, no secrets in DB)
│   │   ├── logging.py
│   │   ├── security.py            # auth, SSRF guard, HTML sanitize, rate limit
│   │   ├── providers.py           # Model Gateway registry
│   │   └── feature_flags.py
│   │
│   ├── domain/                    # pure domain logic, no I/O
│   │   ├── users/
│   │   ├── documents/
│   │   ├── stories/
│   │   ├── events/
│   │   ├── entities/
│   │   ├── topics/
│   │   ├── ranking/
│   │   ├── personalization/
│   │   ├── trends/
│   │   └── digests/
│   │
│   ├── graph/
│   │   ├── state.py               # typed state (Pydantic)
│   │   ├── workflows/
│   │   │   ├── main.py            # top-level DAG
│   │   │   └── subgraphs/
│   │   │       ├── acquisition.py
│   │   │       ├── intelligence.py
│   │   │       └── delivery.py
│   │   ├── nodes/
│   │   │   ├── acquisition.py
│   │   │   ├── normalize.py
│   │   │   ├── identity.py
│   │   │   ├── cluster.py
│   │   │   ├── events.py
│   │   │   ├── entities.py
│   │   │   ├── claims.py
│   │   │   ├── ranking.py
│   │   │   ├── personalize.py
│   │   │   ├── quality_gate.py
│   │   │   ├── digest.py
│   │   │   └── delivery.py
│   │   └── routing.py
│   │
│   ├── connectors/                # SourceConnector protocol + adapters
│   │   ├── base.py                # SourceConnector(Protocol), SourceQuery, RawDocument
│   │   ├── arxiv/
│   │   ├── openalex/
│   │   ├── semantic_scholar/
│   │   ├── crossref/
│   │   ├── github/
│   │   ├── huggingface/
│   │   ├── rss/
│   │   ├── news/
│   │   ├── hackernews/
│   │   ├── reddit/
│   │   └── regulation/            # NIST / EU AI Office / CISA / CVE stubs
│   │
│   ├── intelligence/
│   │   ├── clustering.py
│   │   ├── ranking.py             # composite ranker + reranker
│   │   ├── entities.py
│   │   ├── claims.py
│   │   ├── evidence.py
│   │   ├── trends.py
│   │   ├── novelty.py
│   │   ├── diversity.py
│   │   └── personalization.py
│   │
│   ├── providers/                 # provider implementations
│   │   ├── llm/
│   │   │   ├── base.py
│   │   │   ├── groq.py
│   │   │   ├── openai.py
│   │   │   └── anthropic.py
│   │   ├── embeddings/
│   │   │   ├── base.py
│   │   │   └── mini_lm.py
│   │   ├── reranking/
│   │   │   ├── base.py
│   │   │   └── cohere.py
│   │   ├── search/
│   │   │   ├── base.py
│   │   │   └── tavily.py
│   │   └── email/
│   │       ├── base.py
│   │       ├── smtp.py
│   │       └── resend.py
│   │
│   ├── storage/
│   │   ├── models/                # SQLAlchemy models
│   │   ├── repositories/          # DB access
│   │   ├── migrations/            # Alembic (alembic/ or app/storage/migrations)
│   │   └── cache.py               # optional Redis helper
│   │
│   ├── services/
│   │   ├── digest_service.py
│   │   ├── feedback_service.py
│   │   ├── recommendation_service.py
│   │   └── notification_service.py
│   │
│   ├── templates/
│   │   └── email/
│   │       ├── base.html
│   │       ├── digest.html        # new structure per §9 / Appendix C
│   │       └── partials/
│   │
│   └── workers/
│       ├── ingestion.py
│       ├── processing.py
│       └── delivery.py
│
├── config/
│   ├── profile.json               # bootstrap only (deprecated as source of truth after Phase 1)
│   ├── profile_news.json
│   ├── sources.yaml               # source catalog (enabled, rate limits, trusted domains)
│   └── ranking.yaml               # weight config for composite ranker
│
├── scripts/
│   ├── migrate_legacy_data.py
│   ├── seed_demo.py
│   └── eval_ranking.py
│
├── docs/
│   ├── UPGRADE_AUDIT.md
│   ├── ARCHITECTURE.md
│   ├── DATA_MODEL.md
│   ├── SOURCE_CATALOG.md
│   ├── RANKING.md
│   ├── PERSONALIZATION.md
│   ├── EVALUATION.md
│   ├── SECURITY.md
│   └── MIGRATION.md
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── regression/
│   └── evaluation/
│
├── .github/workflows/agent_run.yml  # retained, updated for Postgres + new entrypoint
├── alembic.ini
├── pyproject.toml
├── requirements.txt                 # or pyproject deps — consolidated in Phase 0
├── .env.example
└── plan.md                         # this file
```

**Removed / deprecated (cleaned):**

- Root `main.py` → `app/main.py`
- `src/core/scheduler.py` Git-DB sync path → replaced by DB-backed scheduler.
- `HF_DAILY_PAPERS_INTEGRATION.md` → merged into `docs/SOURCE_CATALOG.md`.
- `about_project.md` → replaced by `docs/*` + README.
- `!data/history.db` git-tracking hack → removed; DB is Postgres (SQLite stays only for local dev/tests).

---

## 6. Data Model / Storage Strategy

### 6.1 Postgres is primary; SQLite stays for dev/tests

- Use **Alembic** migrations from day 1. Every model change = migration.
- Support `DATABASE_URL` with fallback to SQLite for local dev/CI.
- Add `pgvector` extension when Postgres is used (vector column for embeddings).

### 6.2 Core entities (all have `id: UUID`, `created_at`, `updated_at`, `metadata: JSONB`, `status`)

```
User
  id, email, display_name, timezone, created_at
  preferences (FK → UserPreference), delivery_preferences, privacy

UserPreference
  explicit_topics, excluded_topics, preferred_sources, disliked_sources,
  preferred_depth (enum), preferred_frequency, preferred_formats,
  semantic_interest_vector (vector), topic_affinity JSONB, entity_affinity JSONB,
  source_affinity JSONB, reading_time_patterns JSONB, recent_interest_shift JSONB

UserFeedback / UserInteraction
  user_id, target_type (story/document/entity), target_id,
  action (OPEN/CLICK/SAVE/LIKE/DISLIKE/HIDE/MUTE_TOPIC/FOLLOW/UNFOLLOW/SKIP/SHARE/READ_DURATION),
  created_at, context JSONB

Source
  id, name, source_type (PRIMARY/SECONDARY/COMMUNITY), adapter, enabled,
  rate_limit, reliability_score, last_success, last_failure, failure_count

Document
  id, canonical_url, original_url, title, source_id, publisher, author,
  published_at, fetched_at, language, text, summary, content_hash, title_hash,
  external_id, topics JSONB, entities JSONB, source_tier (A/B/C/D)

StoryCluster (Story)
  id, title, summary, why_it_matters, confidence, status,
  cluster_reason, cluster_confidence, document_ids[], event_ids[], entity_ids[]

Event
  id, event_type (MODEL_RELEASE | PAPER_RELEASE | … 18 types), event_date,
  entities JSONB, claims JSONB, confidence, source_ids[]

Entity / EntityMention
  id, kind (Company/Person/Model/Repo/Dataset/Benchmark/…),
  canonical_name, aliases[], relations JSONB

Claim / ClaimEvidence
  id, text, claim_type, confidence, evidence_refs[] -> {document_id, span, url}

Topic / TopicRelation
  trend states stored via TrendObservation

TrendObservation
  topic_or_entity_id, ts, mentions, unique_sources, velocity, state (RISING/ACCELERATING/PEAKING/STABLE/DECLINING/RE-EMERGING)

Digest / DigestItem
  id, user_id, status (draft/review/approved/scheduled/sent/failed/archived),
  subject_variants JSONB, story_ids[], rendered_html, quality_score

SavedItem
  user_id, story_id, saved_at

Delivery / DeliveryEvent
  digest_id, provider, message_id, status (pending/sending/sent/failed/retrying),
  events: delivered/opened/clicked/bounced/complained/suppressed/failed

ModelRun / PromptVersion / EvaluationRun
  for versioning prompts/models/rankers and offline eval
```

### 6.3 Identity & normalization

- `canonical_url` vs `original_url` preserved. Strip UTM params, resolve AMP/canonical, handle syndication.
- `content_hash = sha256(normalized_text)`, `title_hash = sha256(normalized_title)` — use for `content_fingerprint` and fast dedupe before semantic step.
- External IDs (arxiv id, DOI, GH repo) stored when available.

---

## 7. Provider Abstraction (Anti Lock-in)

All business logic depends on **Protocols**, never concrete SDKs.

```python
# app/connectors/base.py
class SourceConnector(Protocol):
    async def fetch(self, query: SourceQuery) -> list[RawDocument]: ...

# app/providers/llm/base.py
class LLMProvider(Protocol):
    async def complete(self, messages: list[Message], schema: type[BaseModel] | None, ...) -> LLMResult: ...

# similarly: EmbeddingProvider, RerankerProvider, SearchProvider, EmailProvider
```

**Model Gateway** (`app/core/providers.py`):

```
Model Gateway
├── fast     → cheap classifier / filter
├── reasoning→ evidence analysis / contradiction
├── cheap    → formatter / subject lines
├── embedding→ MiniLM default, swappable to OpenAI/Cohere
└── reranker → Cohere rerank / cross-encoder
```

Config selects provider per task:

```yaml
providers:
  llm_fast: groq:llama-3.1-8b
  llm_reasoning: openai:gpt-4o  # or groq:llama-3.1-70b
  embedding: local:all-MiniLM-L6-v2
  reranker: cohere:rerank-english-v3
  search: tavily
  email: smtp  # or resend
```

No `import groq` inside domain/intelligence code — only inside provider impl.

---

## 8. Intelligence Layer — Detailed Design

### 8.1 Document normalization

Every adapter returns `RawDocument`, then `app/graph/nodes/normalize.py` maps to canonical `Document` (see §6.3). Metadata + timestamps preserved end-to-end (never drop `published_at` / `fetched_at`).

### 8.2 Story clustering (replaces URL dedupe)

Signals: semantic similarity (embedding cosine), lexical (BM25 / Jaccard on title), entity overlap, time window (default 72h), event-type similarity, publisher diversity.

Algorithm (Phase 2 baseline, cheap → accurate):
1. Block by time window + entity overlap to reduce O(n²).
2. Embed titles+first 500 chars.
3. Agglomerative clustering with threshold `0.82` (tunable) + entity boost.
4. Emit `StoryCluster` with `cluster_confidence` + `cluster_reason` (top 2 signals).

Feedly's Deduplication (§1 ref) is the benchmark: content-level, not URL-level.

### 8.3 Event extraction

18 types: `MODEL_RELEASE, PRODUCT_RELEASE, PAPER_RELEASE, DATASET_RELEASE, GITHUB_RELEASE, FUNDING, ACQUISITION, PARTNERSHIP, BENCHMARK_RESULT, RESEARCH_RESULT, SECURITY_INCIDENT, VULNERABILITY, REGULATION, POLICY_CHANGE, COMPANY_ANNOUNCEMENT, OPEN_SOURCE_RELEASE, CONFERENCE, JOB_SIGNAL`.

Cheap classifier first (`fast` LLM or rules), then `reasoning` model only for high-importance clusters. Each event carries `confidence` + source refs.

### 8.4 Entity extraction & KG

Types: Company, Person, Model, Repository, Dataset, Benchmark, Technology, Framework, ResearchTopic, Institution, Regulator, Country, Product.

Normalization via alias map + OpenAlex/Semantic Scholar IDs where available. Store relations (`Company → released → Model`). Enables "What changed about OpenAI this week?" without a fresh web search.

### 8.5 Source quality tiers

- **Tier A** — primary: official docs, papers, company announcements, gov docs, benchmarks.
- **Tier B** — reputable journalism / technical pubs.
- **Tier C** — GitHub / HN / Reddit / social.
- **Tier D** — unknown/SEO/aggregators/AI-slop.

Dynamic score: `authority, primary_source, historical_reliability, specificity, recency, editorial_quality, community_signal` → `PRIMARY/SECONDARY/COMMUNITY/UNKNOWN`. Reddit can surface but never outranks the primary source for the same story.

### 8.6 Evidence & contradiction

Every `Claim` links to `ClaimEvidence{ document_id, span, url }`. Unsupported / stale / duplicate detection is explicit. Contradiction detector emits:

```
Claim A: "50% faster" (company blog, Tier A)
Claim B: "17% faster" (independent benchmark, Tier A)
Conflict: metrics differ; evaluation conditions differ → explain, don't pick.
```

### 8.7 Composite ranking (replaces 0.45/0.30)

**Stage 1 — cheap retrieval:** embedding + keyword + metadata (all candidates).
**Stage 2 — reranker:** reranker model on top-50 candidates only.

Score (weights configurable via `config/ranking.yaml`, exposed breakdown for debugging):

```
final_score =
    semantic_relevance
  + freshness
  + novelty
  + source_quality  (tier + reputation)
  + event_importance
  + trend_velocity
  + user_affinity    (topic/entity/source affinity + history)
  + information_gain
  + diversity_bonus
  - repetition_penalty
  - topic_fatigue
  - low_quality_penalty
  - weak_evidence_penalty
```

Output per story (for `docs/RANKING.md` debug view):

```json
{"relevance":0.91,"freshness":0.82,"novelty":0.95,"authority":0.98,"trend_velocity":0.87,"personal_affinity":0.94,"final_score":0.92}
```

Note: exact weights are seeded defaults (0.22 semantic, 0.12 freshness, 0.13 novelty, 0.14 authority, 0.10 trend, 0.14 affinity, rest penalties) and tuned via evaluation, not frozen.

### 8.8 Novelty & trend engines

- **Novelty:** new vs repeat vs minor repost vs major update; surfaced only if meaningful new evidence (`content_hash` + claim delta).
- **Trend:** per topic/entity time-series (`mentions/day, sources/day, unique publishers, GH stars, paper volume, citation velocity, community activity`) → states `emerging/accelerating/stable/declining/breaking/overhyped/re-emerging`. Charts in UI.

### 8.9 Personalization

Profiles have `long_term / recent / session` slices with **temporal decay** (exponential half-life 14d for recent, 90d for long-term). Update from feedback stream aggregated (not per-click retrain).

Affinity maps: `topic_affinity, entity_affinity, source_affinity` plus `reading_time_patterns, click/save/skip histories`. Ranking blends them via `user_affinity`.

### 8.10 Diversity & fatigue

Scoring considers publisher, perspective, geo-region, source-type, topic-subcategory. Deliberately inject `alternative perspective / primary source / independent benchmark / community reaction` when story has single-source risk. Apply `repetition_penalty` and `topic_fatigue` so 6-month-old interests don't fossilize.

---

## 9. Email & Delivery Redesign

### 9.1 Digest structure (new, replaces What/How/Insight only)

```
YOUR AI INTELLIGENCE BRIEF — August 30, 2026

THE 3-7 THINGS THAT MATTER (variable count; zero is valid)
1. Story
   What happened / Why it matters / What changed / Evidence / Who is affected / What to watch next
   Sources: 4  Confidence: High  Entities: OpenAI, GPT-5

TREND WATCH
  AI Agents ↑  Multimodal →  RAG ↓

WHAT TO WATCH NEXT
  • Benchmark expected … / Release to watch …

YOUR READING SIGNAL
  High interest: Agent Infrastructure, Open-source inference  [Adjust]

RECOMMENDED
  Deep read / Paper / Repo

Sources (cited) + Feedback controls
```

Sections are **conditional** — if no trend signal, omit `TREND WATCH`. Quality gate controls count.

### 9.2 HTML requirements

Responsive, mobile-friendly, accessible, lightweight, dark-mode aware, visually restrained, citation-rich. No gradient spam. Tested in common clients (Litmus/Email on Acid check in Phase 5).

### 9.3 Subjects

Generate 4 variants per digest (`informational/curiosity/executive/technical`). Choose via `feature_flag` strategy; track engagement. No clickbait.

### 9.4 Provider

`EmailProvider` with `SMTP` + `Resend` (or equivalent API). Persist `delivery_id/message_id/status/provider/sent_at`. Capture webhook events `delivered/opened/clicked/bounced/complained/suppressed/failed` for learning (Resend event-types model).

### 9.5 Interaction

Per-story links: `Read / Save / More like this / Less like this / Mute topic / Follow / Ask about this`. First-party routes — don't rely on open rate (privacy). Clicks/saves/feedback are primary signals. Idempotency key per `digest_id` prevents duplicates after restart.

### 9.6 Quality gates before send

Check `minimum evidence quality, ranking confidence, duplicate rate, citation coverage, source diversity, user relevance, topic fatigue`. If weak → `log skipped_digest` with reason; not an error.

---

## 10. Web App / API / Ask-Your-Intelligence

### 10.1 Delivery surfaces (newsletter is one)

Email, web dashboard, Telegram/Slack, alerts, weekly reports, "what changed?" queries, topic monitoring, company/model tracking, historical timelines, saved knowledge, API.

### 10.2 FastAPI routes

See §5 `app/api/routes/*`. All responses use Pydantic schemas; LLM outputs are schema-validated JSON.

### 10.3 UI (Phase 5 — minimal viable first)

- **Dashboard:** Your Brief / Trending in Your Topics / New Since Yesterday / Deep Research / Following / Saved
- **Story page:** Summary / Timeline / Sources / Evidence / Conflicting claims / Entities / Related / Ask about story
- **Topic page:** Overview / trend / latest stories / research / companies / repos / timeline
- **Source page:** quality / recent stories / reliability
- **Profile:** interests / excluded topics / sources / frequency / delivery / reading style

Can start with server-rendered Jinja2 + HTMX; SPA later. Keep it boring and fast.

### 10.4 Ask Your Intelligence (grounded)

```
Question → retrieve relevant stories/events/claims (hybrid search over docs/stories/entities/trends/history)
         → build evidence context
         → reasoning model with citations
         → structured answer + citations
```

Never answer from model memory when stored evidence exists. Hybrid search: keyword + semantic + entity + filters (date/source/topic).

### 10.5 Scheduler

Per-user `cron / daily / weekly / weekdays / custom` + timezone + adaptive window (learn `typical_open_time/click_time/preferred_weekday`). Use controlled experiment to adjust, not aggressive auto-shift.

---

## 11. Phased Roadmap (P0 → P3) with Atomic Commits

**Rule:** No phase starts until previous phase's tests + docs pass. Each commit is shippable and bisectable.

### Phase 0 — Stabilize & Audit (Week 1) — P0

> Goal: understand reality, add guardrails, no feature work yet.

| # | Commit (conventional) | Scope |
|---|----------------------|-------|
| 0.1 | `chore: add plan.md and phase tracking` | This plan |
| 0.2 | `docs: add UPGRADE_AUDIT.md` | Repo map, deps, graph topology, state fields, schemas, debt, failure modes, lock-in, risks |
| 0.3 | `chore: set up tooling (ruff, mypy, pytest, pre-commit)` | `pyproject.toml`, lint/type/test baseline |
| 0.4 | `test: add baseline unit tests for ranker/normalize/canonicalize` | Lock current behavior before changes |
| 0.5 | `refactor: extract typed settings (pydantic-settings)` | `app/core/config.py` replaces ad-hoc `os.getenv` |
| 0.6 | `docs: add ARCHITECTURE.md skeleton` | Target arch + current arch side-by-side |

**Exit criteria:** `pytest -q` green, `ruff check` + `mypy --ignore-missing-imports` green, audit doc complete.

### Phase 1 — Foundation: DB, Providers, Identity, Config-as-DB (Weeks 2-3) — P0

| # | Commit | Scope |
|---|--------|-------|
| 1.1 | `feat(db): add Postgres+pgvector support with Alembic` | `app/storage/models/*`, `alembic.ini`, fallback to SQLite |
| 1.2 | `feat(db): add core models (User, Document, StoryCluster, Entity, Digest, Delivery)` | Migrations `001_*` |
| 1.3 | `feat(providers): add LLM/Embedding/Reranker/Search/Email provider protocols` | `app/providers/*/base.py`, `app/core/providers.py` |
| 1.4 | `refactor(providers): migrate Groq writer to LLMProvider` | `app/providers/llm/groq.py`, keep fallback enrichment |
| 1.5 | `feat(connectors): add SourceConnector protocol + registry` | `app/connectors/base.py`, `config/sources.yaml` |
| 1.6 | `refactor(connectors): wrap existing clients behind adapters` | arxiv/tavily/hn/hf/rss adapters (no API behavior change yet) |
| 1.7 | `feat(identity): add URL canonicalization + content fingerprint` | `app/domain/documents/identity.py`, tests |
| 1.8 | `feat(config): DB-backed user preferences (profile.json → seed)` | `UserPreference` model, import path |
| 1.9 | `feat(cli): add scripts/migrate_legacy_data.py` | Import `profile.json`, sent URLs, schedules |
| 1.10 | `docs: add DATA_MODEL.md + SOURCE_CATALOG.md` |  |

**Exit criteria:** Legacy SQLite imports into Postgres, adapters pass existing source tests, no URL-only identity in new code.

### Phase 2 — Intelligence Core: Clustering, Events, Entities, Evidence, Source Scoring (Weeks 3-5) — P0

| # | Commit | Scope |
|---|--------|-------|
| 2.1 | `feat(intel): implement story clustering` | `app/intelligence/clustering.py`, `cluster_id/confidence/reason` |
| 2.2 | `feat(intel): event extraction (18 types)` | `app/intelligence/events.py`, cheap classifier first |
| 2.3 | `feat(intel): entity extraction + normalization` | `app/intelligence/entities.py`, alias map |
| 2.4 | `feat(intel): evidence/claim objects + citation` | `app/intelligence/claims.py`, `evidence.py` |
| 2.5 | `feat(intel): source quality tiers + reputation` | `app/intelligence/source_quality.py` |
| 2.6 | `feat(graph): add intelligence subgraph` | `normalize → identity → cluster → events → entities → claims → source_score` |
| 2.7 | `feat(observability): per-stage metrics` | source_success_rate, clusters_created, citation_coverage |
| 2.8 | `docs: add RANKING.md (signals + weights draft)` |  |

**Exit criteria:** 7 copies of same story → 1 cluster; evidence citations present; source tiers visible in ranking breakdown.

### Phase 3 — Ranking: Hybrid Retrieval + Reranker + Novelty + Diversity + Quality Gates (Weeks 5-7) — P0/P1

| # | Commit | Scope |
|---|--------|-------|
| 3.1 | `feat(ranking): composite scorer (configurable weights)` | `app/intelligence/ranking.py`, `config/ranking.yaml` |
| 3.2 | `feat(ranking): two-stage (cheap retrieval → reranker)` | `app/providers/reranking/*`, cost reduction verified |
| 3.3 | `feat(ranking): novelty detection` | `app/intelligence/novelty.py` (hash + claim delta) |
| 3.4 | `feat(ranking): diversity + fatigue` | `app/intelligence/diversity.py` |
| 3.5 | `feat(graph): replace fixed thresholds + min-3 fallback` | New `quality_gate` node, `digest_skipped_low_signal` path |
| 3.6 | `feat(delivery): idempotent delivery + lifecycle statuses` | `pending/sending/sent/failed/retrying`, no pre-send commit |
| 3.7 | `docs: add EVALUATION.md` | Dataset spec + NDCG/Precision methodology |
| 3.8 | `test: add ranking regression tests` | Known stories/duplicates/ranking decisions |

**Exit criteria:** No `score >= 0.45` gate in prod path; zero-send path exercised; reranker AB shows cost/quality win.

### Phase 4 — Personalization: Accounts, Feedback, Adaptive Profile (Weeks 7-9) — P0/P1

| # | Commit | Scope |
|---|--------|-------|
| 4.1 | `feat(users): user accounts + preferences tables` | `User`, `UserPreference` migrations |
| 4.2 | `feat(feedback): interactions + events (12 action types)` | `UserInteraction` model, event stream |
| 4.3 | `feat(personalization): long/recent/session profiles + decay` | `app/intelligence/personalization.py` |
| 4.4 | `feat(ranking): integrate user_affinity into final_score` |  |
| 4.5 | `feat(email): feedback controls in template` | More/Less/Mute/Follow/Save/Ask per story |
| 4.6 | `feat(graph): human-in-the-loop (draft→review→approved)` | LangGraph interrupt/resume, admin approve/edit/reject/regenerate |
| 4.7 | `feat(providers): EmailProvider + Resend + webhook ingestion` | Delivery events feed back into personalization |
| 4.8 | `docs: add PERSONALIZATION.md + SECURITY.md` |  |

**Exit criteria:** Click/save/mute changes next digest's ranking; profile evolves with decay; HITL works via checkpoint resume.

### Phase 5 — Product: API, Dashboard, Following, Saved, Ask (Weeks 9-12) — P1

| # | Commit | Scope |
|---|--------|-------|
| 5.1 | `feat(api): FastAPI factory + auth + routes skeleton` | `app/api/*` |
| 5.2 | `feat(api): digests/stories/search routes` |  |
| 5.3 | `feat(api): following + saved + topics/entities` |  |
| 5.4 | `feat(ask): grounded QA pipeline` | `app/api/routes/ask.py` + retrieval |
| 5.5 | `feat(ui): dashboard + story/topic/profile pages` | Minimal SSR (Jinja2 + HTMX) |
| 5.6 | `feat(connectors): add OpenAlex, Semantic Scholar, Crossref adapters` | Research graph |
| 5.7 | `feat(connectors): expand GitHub (releases, trending, velocity)` |  |
| 5.8 | `feat(email): new digest template (variable sections)` | Per §9 / Appendix C |
| 5.9 | `test: add API + graph integration tests` |  |

**Exit criteria:** User can browse brief, follow an entity, save a story, ask "what changed this week?" with citations.

### Phase 6 — Advanced Intelligence (Weeks 12-14) — P2

| # | Commit | Scope |
|---|--------|-------|
| 6.1 | `feat(trends): velocity + state machine` | `app/intelligence/trends.py`, `TrendObservation` series |
| 6.2 | `feat(intel): contradiction detection` | Cross-source claim conflicts with explanation |
| 6.3 | `feat(intel): knowledge gap detection + recommendations` | Explainer/paper/guide suggestions |
| 6.4 | `feat(delivery): adaptive send window + experiments` | Feature flags, subject/ordering experiments |
| 6.5 | `feat(security): regulation + CVE intelligence feeds` | `app/connectors/regulation/*` |
| 6.6 | `feat(cost): adaptive analysis budget (cheap vs deep)` | Low-value 1-2 sources; high-impact 5-10 + contradiction |

**Exit criteria:** Trend charts in UI; contradiction banner in story page; recommendation carousel after digest.

### Phase 7 — Hardening & Launch (Weeks 14-16) — P2/P3

| # | Commit | Scope |
|---|--------|-------|
| 7.1 | `feat(obs): structured logging + tracing + health dashboard` | Source latency, LLM cost/user, bounce/complaint |
| 7.2 | `feat(ops): circuit breakers + rate-limit handling per source` | One dead API never kills digest |
| 7.3 | `feat(ops): backups, migrations, disaster recovery` |  |
| 7.4 | `feat(privacy): export/delete/clear personalization` |  |
| 7.5 | `feat(security): SSRF guard, HTML sanitize, prompt-injection defense` | `UNTRUSTED_SOURCE_CONTENT` framing |
| 7.6 | `feat(admin): internal tools (health, rerun, ranking explainer)` |  |
| 7.7 | `docs: finalize MIGRATION.md + README + runbooks` |  |
| 7.8 | `chore: legacy cleanup — remove src/ shim after strangle` |  |

**Exit criteria:** All §17 gates pass; dry-run + controlled delivery test succeed; cost/user documented.

### Commit inventory

- Total planned commits: **~47 atomic commits** across 7 phases.
- Each commit touches **one concern**; no mega-commits.

---

## 12. Branching & Commit Discipline

- `main` is protected; feature branches `feat/<phase>-<slug>`, e.g. `feat/p2-clustering`.
- Squash-merge per commit row above (one PR per row is ideal; batch at most 2 rows per PR).
- Commit message format: `type(scope): summary` (types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`).
- Every `feat` commit includes: code + tests + migration (if any) + doc snippet update.
- Tag phase checkpoints: `phase-0-audit`, `phase-1-foundation`, …

---

## 13. Migration & Backwards Compatibility

- `scripts/migrate_legacy_data.py` handles: `profile.json` + `profile_news.json` → `UserPreference`, `history.db` sent URLs → `Document` + `Delivery` seeds, `schedules.json` → per-user schedules. Preserves timestamps, reports errors, is idempotent.
- `config/profile.json` stays as **bootstrap/seed/fixture** only after Phase 1. New deploys seed one demo user from it.
- `DATABASE_URL` switching: if unset, use SQLite (dev/CI); if set, use Postgres. CI runs both.
- GitHub Actions workflow updated to use new entrypoint `python -m app.main run --profile config/profile.json` (or `app/cli.py`), not Git DB sync.

---

## 14. Observability, Evaluation & Cost Control

### 14.1 Metrics (per run, per user)

`source_success_rate, source_latency, articles_fetched/normalized, clusters_created, duplicates_removed, stories_ranked/selected, LLM calls/latency/cost, citation_coverage, ranking_confidence, email delivery/bounce/complaint/open/click/save, user_feedback_score, cost_per_digest, tokens_in/out`.

### 14.2 Evaluation framework

Offline dataset: 1000 candidates × 100 profiles + labels (relevance, quality, duplicate clusters, event clusters, source quality, factuality). Metrics: `Precision@5, Recall@20, NDCG@10, duplicate_rate, coverage, novelty, citation correctness, summary faithfulness, engagement`. Every ranker/prompt change runs `scripts/eval_ranking.py` before merge.

### 14.3 Cost controls

Caching by `content_hash`, batch inference, cheap-first → expensive-last, candidate reduction before LLM, provider fallback, token accounting, daily + per-user budgets.

---

## 15. Security, Privacy & Licensing

- **Security:** secrets via env/manager (never DB), input validation, HTML sanitization, SSRF guard (allow-list + no private IP fetch), rate limiting, auth/authz, encrypted secrets, audit logs. Tests for SSRF/HTML injection/prompt injection/poisoned docs/oversized payloads.
- **Prompt injection defense:** article text is `UNTRUSTED_SOURCE_CONTENT` — model instructions: treat as evidence, never as instructions.
- **Privacy:** privacy settings, data export, deletion, clear personalization, no training on private data without consent.
- **Licensing/copyright:** store `title/publisher/author/timestamp/short excerpt when permitted/summary/URL` — not full bodies. Respect rate limits, ToS, robots where applicable, attribution.

---

## 16. Docs to Produce Per Phase

| Phase | Doc |
|-------|-----|
| 0 | `docs/UPGRADE_AUDIT.md`, `docs/ARCHITECTURE.md` (skeleton) |
| 1 | `docs/DATA_MODEL.md`, `docs/SOURCE_CATALOG.md` |
| 2-3 | `docs/RANKING.md` |
| 4 | `docs/PERSONALIZATION.md`, `docs/SECURITY.md` |
| 3-4 | `docs/EVALUATION.md` |
| 7 | `docs/MIGRATION.md`, updated `README.md` |

All docs include diagrams, schema excerpts, and operational notes.

---

## 17. Definition of Done (Product Launch Gates)

No launch until **all** pass:

**Quality**
- Duplicate rate < 5% (measured on eval set), citation coverage > 90% on important claims, summary faithfulness pass.

**Personalization**
- Feedback (save/like/mute/follow) measurably changes next ranking; topic fatigue detected; source preferences evolve.

**Reliability**
- Single source failure doesn't fail digest; delivery idempotent (restart-safe); migrations reversible; jobs resumable via LangGraph checkpoint.

**Product**
- Users can: browse stories, follow topics/entities, save stories, give feedback, ask grounded questions with citations, manage delivery prefs.

**Maintainability**
- Providers swappable via config (no code change), connectors modular, graph nodes unit-testable, ranking tested independently, prompts/models versioned.

**Security**
- External content treated as untrusted; SSRF/prompt-injection mitigated; secrets protected; export/delete works.

---

## 18. What NOT to Do

- Don't add 47 feeds without source-quality logic.
- Don't blindly scrape; respect ToS/rate limits.
- Don't freeze single-provider coupling (Groq/Tavily/SMTP only).
- Don't hard-freeze thresholds; don't force N articles; don't use URL as story identity.
- Don't let LLM invent claims; don't store full copyrighted bodies unnecessarily.
- Don't build Kafka/microservices/premature multi-DB; Postgres+pgvector is enough.
- Don't optimize only for open rate; clicks/saves/feedback are truth.

---

## Appendix A: Source Catalog

### P0 — must have (Phase 1-2)

| Class | Adapter | Provides |
|-------|---------|----------|
| Primary research | arXiv | papers, metadata |
| Primary research | OpenAlex | works/authors/institutions/topics/citations |
| Primary research | Semantic Scholar | papers/citations/embeddings/recommendations |
| Primary research | Crossref | DOI, funding, license, ORCID/ROR |
| Model ecosystem | Hugging Face | models/datasets/spaces/webhooks |
| Developer | GitHub (releases/trending) | adoption/velocity |
| Tech blogs | RSS (configurable) | implementation detail |
| News/web | Tavily (abstracted via `SearchProvider`) | events/context |
| Community | Hacker News, Reddit | practitioner signal |

### P1 — add in Phase 5-6

Official AI lab blogs, benchmark leaderboards, Product Hunt / changelogs, NIST / EU AI Office / CVE/NVD / CISA, YouTube (secondary evidence only, caption auth constraints), job signals.

### P2 — later

Patents (USPTO/PatentsView), funding DBs, Common Crawl enrichment, multilingual publishers.

All adapters behind `SourceConnector`; all independently disableable; circuit breaker per source.

---

## Appendix B: Ranking Formula

```yaml
# config/ranking.yaml — defaults (tunable, then learned)
weights:
  semantic_relevance: 0.22
  freshness:          0.12
  novelty:            0.13
  source_quality:     0.14  # tier + reputation
  event_importance:   0.09
  trend_velocity:     0.10
  user_affinity:      0.14
  information_gain:   0.06
penalties:
  repetition:         0.08
  topic_fatigue:      0.07
  low_quality:        0.10
  weak_evidence:      0.09
diversity:
  bonus:              0.05
reranker:
  enabled: true
  provider: cohere
  top_k: 50
```

Two-stage: Stage 1 cheap (embedding/keyword/metadata on all candidates) → Stage 2 reranker on top-50 → final composite. Score breakdown logged per story for debugging and for `docs/RANKING.md`.

---

## Appendix C: Digest Template Spec

```
Header: YOUR AI INTELLIGENCE BRIEF — {date} — {user display name}

Section: THE N THINGS THAT MATTER (1-7 items; zero valid → skip section, emit digest_skipped)
  Per story:
    headline
    what_happened (1-2 sent)
    why_it_matters (1 sent)
    what_changed / technical detail (bullets)
    evidence: citations with source tier badges
    who_is_affected
    what_to_watch_next
    confidence: High/Med/Low (from source agreement)
    entities + topics chips

Conditional sections (only if signal):
  RESEARCH WATCH (papers with citation graph hint)
  TREND WATCH (↑ → ↓ chips)
  IMPORTANT RELEASES
  THINGS YOU MAY HAVE MISSED

Footer:
  YOUR READING SIGNAL (recent affinities) + [Adjust interests]
  Recommended: topic / deep read / repo
  Sources (numbered)
  Feedback: Useful 👍 / Not useful 👎 / More like this / Less / Mute / Follow / Save / Ask about this

Subject variants: informational / curiosity / executive / technical
```

Template lives at `app/templates/email/base.html` + `digest.html`; responsive, accessible, dark-mode aware.

---

## Appendix D: Evaluation Dataset Spec

- `tests/evaluation/` + `data/eval/` (gitignored if large; seed small fixture in repo).
- 1000 candidates × 100 synthetic + 5-10 real profiles; human labels for relevance/quality; duplicate/event clusters; source-quality labels; summary factuality labels.
- Metrics: `Precision@5, Recall@20, NDCG@10, duplicate_rate, coverage, novelty, citation_correctness, summary_faithfulness, engagement (click/save/feedback)`.
- `scripts/eval_ranking.py` runs regression on every ranking/prompt change; results appended to `EvaluationRun` table.

---

## Appendix E: Deliverable Checklist

At project completion, the repo must contain:

- [ ] `docs/UPGRADE_AUDIT.md`
- [ ] `docs/ARCHITECTURE.md`
- [ ] `docs/DATA_MODEL.md`
- [ ] `docs/SOURCE_CATALOG.md`
- [ ] `docs/RANKING.md`
- [ ] `docs/PERSONALIZATION.md`
- [ ] `docs/EVALUATION.md`
- [ ] `docs/SECURITY.md`
- [ ] `docs/MIGRATION.md`
- [ ] Updated `README.md` (product positioning, not script)
- [ ] `app/` structure as in §5 (strangled from `src/`)
- [ ] Alembic migrations + `DATABASE_URL` dual support
- [ ] Provider abstractions + Model Gateway
- [ ] Intelligence engine (cluster/event/entity/evidence/trend/novelty/diversity)
- [ ] Composite ranker + reranker + quality gate
- [ ] User accounts + feedback + adaptive delivery
- [ ] FastAPI + minimal dashboard + Ask pipeline
- [ ] Idempotent delivery + webhook event ingestion
- [ ] Tests: `pytest -q`, `ruff check`, `mypy`, migration tests, security checks, dry-run
- [ ] Operational notes: cost/user, known limitations, remaining debt, architectural risks

---

## Final Note for the Builder

Implement **one commit row at a time** from §11. Don't batch phases. After each commit, run `pytest`, `ruff`, `mypy`, and a dry-run (`python -m app.cli digest --dry-run`). The moat is the **accumulated preference history + story graph + evidence graph + source quality + memory** — not a nicer prompt. Build that, and the product survives model churn.

