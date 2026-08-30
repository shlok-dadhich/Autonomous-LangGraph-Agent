"""Quality gate — allows 0-N stories."""

from __future__ import annotations

from loguru import logger

def quality_gate_node(state: dict) -> dict:
    clusters = state.get("clusters", [])
    # Phase 2: keep clusters with confidence >=0.6; allow zero
    kept = [c for c in clusters if c.get("cluster_confidence",0) >= 0.6]
    if not kept and clusters:
        # keep top 1 if all low but we have signal
        kept = [max(clusters, key=lambda x: x.get("cluster_confidence",0))]
    logger.info(f"[quality_gate] {len(clusters)} -> {len(kept)} clusters (allow 0-N)")
    if not kept:
        logger.info("[quality_gate] digest_skipped_low_signal")
    return {"clusters": kept, "logs": [{"level": "info", "message": f"[quality_gate] {len(kept)}/{len(clusters)} clusters kept"}]}
