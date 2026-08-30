"""Ranking node — two-stage composite scorer."""

from __future__ import annotations

from loguru import logger

from app.intelligence.clustering import ClusterInput  # noqa
from app.intelligence.diversity import diversity_bonus
from app.intelligence.novelty import novelty_score
from app.intelligence.ranking import score_document
from app.providers.reranking.cohere import CohereReranker


def ranking_node(state: dict) -> dict:
    docs = state.get("documents", [])
    clusters = state.get("clusters", [])
    source_scores = {s.get("document_id"): s for s in state.get("source_scores", [])}

    # Stage 1: cheap composite scoring
    scored = []
    for d in docs:
        tier = source_scores.get(d.get("document_id"), {}).get("tier", "D")
        # novelty vs recent hashes (from state)
        seen = set(state.get("seen_hashes", []))
        novelty = novelty_score(d.get("content_hash", ""), seen)
        div_bonus = diversity_bonus([d.get("source", "")], [tier])
        breakdown = score_document(d, {"tier": tier, "novelty": novelty, "diversity_bonus": div_bonus})
        nd = dict(d)
        nd["ranking_breakdown"] = breakdown
        nd["final_score"] = breakdown["final_score"]
        scored.append(nd)

    # Stage 2: reranker on top-50 if enabled (config)
    try:
        from pathlib import Path
        import yaml

        cfg_path = Path("config/ranking.yaml")
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        rerank_cfg = cfg.get("reranker", {}) if isinstance(cfg, dict) else {}
        if rerank_cfg.get("enabled"):
            top_k = int(rerank_cfg.get("top_k", 50))
            # pick top-50 by final_score for reranking
            scored.sort(key=lambda x: x.get("final_score", 0), reverse=True)
            top = scored[:top_k]
            # reranker uses titles as docs
            reranker = CohereReranker()
            query = "latest important AI developments"
            titles = [t.get("title", "") for t in top]
            results = reranker.rerank(query, titles, top_k=len(top))
            # blend: 70% original, 30% reranker score
            for r in results:
                idx = r.index
                orig = top[idx].get("final_score", 0)
                top[idx]["final_score"] = round(0.7 * orig + 0.3 * r.score, 3)
                top[idx]["ranking_breakdown"]["reranker_score"] = round(r.score, 3)
            scored = top + scored[top_k:]
    except Exception as e:
        logger.warning(f"[ranking] reranker failed: {e}")

    scored.sort(key=lambda x: x.get("final_score", 0), reverse=True)
    logger.info(f"[ranking] scored {len(scored)} docs, top score {scored[0].get('final_score') if scored else 'n/a'}")
    return {"documents": scored, "logs": [{"level": "info", "message": f"[ranking] scored {len(scored)} docs"}]}
