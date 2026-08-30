"""Quality gate — allows 0-N stories; never fills with filler."""

from __future__ import annotations

from loguru import logger


def quality_gate_node(state: dict) -> dict:
    clusters = state.get("clusters", [])
    documents = state.get("documents", [])
    # Prefer scored documents threshold; fallback to cluster confidence
    # threshold for final_score >=0.55 (configurable via ranking.yaml quality gate)
    min_score = 0.55
    try:
        # if documents have final_score, filter clusters that contain high-score docs
        doc_scores = {d.get("document_id"): d.get("final_score", 0) for d in documents}
        kept = []
        for c in clusters:
            ids = c.get("document_ids", [])
            best = max((doc_scores.get(i, 0) for i in ids), default=c.get("cluster_confidence", 0))
            if best >= min_score or c.get("cluster_confidence", 0) >= 0.7:
                kept.append(c)
        # allow 0-N: if no candidate passes, return empty (no filler)
        if not kept:
            logger.info(f"[quality_gate] digest_skipped_low_signal: {len(clusters)} clusters, 0 passed (min_score {min_score})")
            return {
                "clusters": [],
                "logs": [{"level": "info", "message": "[quality_gate] digest_skipped_low_signal"}],
                "metrics": {"skipped": True, "reason": "low_signal"},
            }
        logger.info(f"[quality_gate] {len(clusters)} -> {len(kept)} clusters (allow 0-N, no filler)")
        return {"clusters": kept, "logs": [{"level": "info", "message": f"[quality_gate] {len(kept)}/{len(clusters)} kept"}]}
    except Exception as e:
        logger.warning(f"[quality_gate] fallback due to {e}")
        # fallback to old confidence threshold
        kept = [c for c in clusters if c.get("cluster_confidence", 0) >= 0.6]
        return {"clusters": kept, "logs": [{"level": "info", "message": f"[quality_gate] fallback {len(kept)}/{len(clusters)}"}]}
