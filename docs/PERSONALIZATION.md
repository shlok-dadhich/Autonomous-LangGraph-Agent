# Personalization — Adaptive Profiles & Feedback

**Status:** Phase 4. `profile.json` is now bootstrap only; `UserPreference` + `UserInteraction` are source of truth.

## 1. Model

- `UserPreference` stores `explicit_topics, excluded_topics, preferred_sources, disliked_sources, preferred_depth/frequency/formats, topic/entity/source_affinity JSON, semantic_interest_vector (JSON/Vector), reading_time_patterns, recent_interest_shift`.
- `UserInteraction` stream: `user_id, target_type (story/document/entity), target_id, action (OPEN/CLICK/SAVE/LIKE/DISLIKE/HIDE/MUTE_TOPIC/FOLLOW/UNFOLLOW/SKIP/SHARE/READ_DURATION/SEARCH), context JSON, created_at`.

Interactions are **first aggregated, then applied** — not per-click retrain.

## 2. Aggregation (Decay)

```python
# app/intelligence/personalization.py
long  half-life 90d  weight 0.3
recent 14d          0.5
session last 2h     0.2
action weights: SAVE 1.5, LIKE 1.2, CLICK 1.0, OPEN 0.6, DISLIKE -1.2, HIDE -1.0, MUTE -1.5
affinity = tanh(sum) → 0-1 (0.5 neutral)
```

Recent behavior outweighs old; `MUTE_TOPIC` strongly suppresses.

## 3. Ranking Integration

`app/graph/nodes/personalize.py` calls `aggregate_profile()` → `user_affinity_score()` per doc and nudges `final_score = 0.85*orig + 0.15*affinity`. Re-sorts. Phase 3 composite weights include `user_affinity 0.14`.

## 4. Feedback Controls

Every digest email (see `app/templates/email/digest.html`) renders per-story links:
`Save • More like this • Less like this • Mute topic • Follow • Ask about this` + footer `Adjust interests`.

All links hit first-party routes (`/feedback/save?story=...`) which call `app/services/feedback_service.record_interaction()`. Clicks/saves are primary signals (opens unreliable due to privacy).

## 5. Human-in-the-Loop

Digest lifecycle: `draft → review → approved → scheduled → sent → failed → archived` (`Digest.status`).
`app/graph/nodes/digest.py` sets `status=review` when `cluster_confidence <0.65` or contradiction detected, or when `digest_mode=REVIEW_REQUIRED`. LangGraph `interrupt/resume` (with Postgres checkpoint) allows `approve/edit/reject/regenerate`.

## 6. Email Delivery Evolution

`EmailProvider` abstraction:
- `SmtpEmailProvider` (Gmail App Password, existing)
- `ResendEmailProvider` (API, `RESEND_API_KEY`, `requests.post` to `api.resend.com/emails`)
Webhook ingestion: `ingest_resend_webhook()` maps Resend events (`email.delivered/opened/clicked/bounced/complained`) → `DeliveryEvent` rows for learning. Use `click/save` over open rate.

## 7. API

Feedback will be exposed via `POST /feedback` and `GET /preferences` (Phase 5). For now, direct DB via `feedback_service`.

## 8. Privacy

User can clear personalization: delete `UserInteraction` rows or reset `UserPreference.affinity` maps. No private data used for model training without explicit consent.
