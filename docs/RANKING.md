# Ranking — Composite Intelligence Ranker

**Status:** Phase 2 (intelligence) + Phase 3 draft. Current Phase 2 enables clustering/evidence/source quality; composite weights become active in Phase 3.

## 1. Current (Phase 2) — Baseline + Intelligence Signals

Phase 0 ranker: single cosine `all-MiniLM-L6-v2` vs threshold 0.45/0.30 (now deprecated as final gate).
Phase 2 adds intelligence signals for ranking:

- **Semantic relevance** (MiniLM) — retained as Stage-1 retriever
- **Source quality** — `app/intelligence/source_quality.py` tier A/B/C/D → 0.95/0.75/0.45/0.25
- **Event importance** — `app/intelligence/events.py` 18 types → model release > general announcement
- **Story clustering** — `app/intelligence/clustering.py` deduplicates to one story per event (7 docs → 1 cluster)
- **Evidence** — `app/intelligence/evidence.py` source agreement → High/Medium/Low confidence
- **Entities** — extracted for affinity/diversity

## 2. Composite Formula (Phase 3 Activation — see plan.md Appendix B)

```yaml
# config/ranking.yaml
weights:
  semantic_relevance: 0.22
  freshness: 0.12
  novelty: 0.13
  source_quality: 0.14
  event_importance: 0.09
  trend_velocity: 0.10
  user_affinity: 0.14
  information_gain: 0.06
penalties:
  repetition: 0.08
  topic_fatigue: 0.07
  low_quality: 0.10
  weak_evidence: 0.09
diversity: {bonus: 0.05}
reranker: {enabled: false, provider: cohere, top_k: 50}
```

```python
final_score = (
    semantic_relevance + freshness + novelty + source_quality +
    event_importance + trend_velocity + user_affinity + information_gain +
    diversity_bonus - repetition_penalty - topic_fatigue - low_quality_penalty - weak_evidence_penalty
)
```

Two-stage: Stage 1 cheap (embedding/keyword/metadata all candidates) → Stage 2 reranker on top-50 only.

Score breakdown per story (for debugging):
```json
{"relevance":0.91,"freshness":0.82,"novelty":0.95,"authority":0.98,"trend_velocity":0.87,"personal_affinity":0.94,"final_score":0.92}
```

## 3. Freshness/Novelty/Diversity (Phase 3)

- **Freshness:** decay by `published_at` age (half-life 3 days)
- **Novelty:** `content_hash` not in recent `seen_hashes` + claim delta
- **Diversity:** bonus when story adds unseen publisher/perspective/geo/source-type

## 4. Quality Gate (Phase 2 — 0-N Stories)

`app/graph/nodes/quality_gate.py` keeps clusters with `cluster_confidence >=0.6`; allows **zero** stories → `digest_skipped_low_signal` instead of filler. Replaces legacy `check_content_threshold <3` + `fallback_search_node`.

## 5. Evaluation

Offline dataset: 1000 candidates × 100 profiles (see `docs/EVALUATION.md` Phase 3). Metrics: `Precision@5, Recall@20, NDCG@10, duplicate_rate, citation correctness, summary faithfulness`.

## 6. Observability

Per run: `source_success_rate, source_latency, articles_fetched, documents_normalized, clusters_created, duplicates_removed, stories_selected, citation_coverage, ranking_confidence` — via `app/core/metrics.py`.
