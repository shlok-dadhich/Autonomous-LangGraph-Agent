"""Source quality scoring node."""

from __future__ import annotations

from loguru import logger
from app.intelligence.source_quality import score_source


def score_sources_node(state: dict) -> dict:
    docs = state.get("documents", [])
    scores = []
    for d in docs:
        src = d.get("source", "unknown")
        url = d.get("url", "")
        sc = score_source(url or src)
        scores.append(
            {
                "document_id": d.get("document_id"),
                "source": src,
                "tier": sc.tier,
                "label": sc.label,
                "score": sc.score,
                "reason": sc.reason,
            }
        )
    logger.info(f"[source_quality] scored {len(scores)} sources")
    return {"source_scores": scores, "logs": [{"level": "info", "message": f"[source_quality] {len(scores)} sources scored"}]}
