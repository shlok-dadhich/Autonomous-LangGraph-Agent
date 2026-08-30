# Evaluation — Ranking & Intelligence

**Status:** Phase 3. Offline dataset + metrics for every ranker/prompt change.
**Script:** `scripts/eval_ranking.py` (produces `EvaluationRun` row).

## 1. Dataset Spec

- `tests/evaluation/` + `data/eval/` (small fixture committed; large eval gitignored)
- 1000 candidate docs × 100 synthetic profiles + 5-10 real profiles
- Labels: `relevance (0-1)`, `quality (0-1)`, `duplicate_cluster_id`, `event_cluster_id`, `source_quality tier`, `summary_factuality`, `citation_correctness`
- Built via `scripts/eval_ranking.py --build` (samples from stored Document/StoryCluster)

## 2. Metrics

- `Precision@5`, `Recall@10`, `NDCG@10`
- `duplicate_rate` (should be <5% post-clustering)
- `coverage`, `novelty` (new hash fraction)
- `citation_coverage` (>90% on important claims)
- `summary_faithfulness` (LLM judged, schema-validated)
- `user_feedback` (click/save via `UserInteraction`)
- `cost_per_digest` (tokens_in/out × price)

## 3. Running

```bash
python scripts/eval_ranking.py --ranker app/intelligence/ranking.py --profile config/profile.json
pytest tests/evaluation -q
```

Every ranking or prompt change must run `scripts/eval_ranking.py` before merge; result appended to `EvaluationRun` table with model/prompt versions.

## 4. Regression Tests

- `tests/regression/test_ranking_regression.py` pins: known stories, known duplicates, known ranking decisions (top-3 order), known contradictory claims.
- `tests/unit/test_ranking.py` (Phase 3) checks composite scorer breakdown and quality gate allows 0.

## 5. Observability Link

Metrics feed into `app/core/metrics.py` and health dashboard (Phase 7): `source_success_rate, clusters_created, duplicates_removed, stories_selected, citation_coverage, bounce_rate, cost_per_user`.

## 6. Notes

- Reranker AB: Stage-1 only vs Stage-2 rerank — measure NDCG gain vs latency/cost.
- Do not optimize solely for open rate; clicks/saves are truth (privacy: opens unreliable).
