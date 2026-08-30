"""Digest generation node — supports HITL lifecycle."""

from __future__ import annotations

from loguru import logger


def generate_digest_node(state: dict) -> dict:
    clusters = state.get("clusters", [])
    documents = state.get("documents", [])
    # Determine HITL mode from state or config
    mode = state.get("digest_mode", "AUTO")  # AUTO | REVIEW_REQUIRED | MANUAL
    # If any cluster has low confidence or contradiction, require review
    needs_review = any(c.get("cluster_confidence", 1) < 0.65 for c in clusters) or bool(state.get("evidence", []))
    if mode == "REVIEW_REQUIRED" or (needs_review and mode == "AUTO" and len(clusters) > 0):
        # limit auto review trigger to high-stakes
        status = "review" if needs_review else "draft"
    else:
        status = "approved" if clusters else "draft"

    # Build digest payload (simplified)
    digest = {"story_ids": [c.get("cluster_id") for c in clusters], "status": status, "needs_review": needs_review}

    logger.info(f"[digest] status={status} needs_review={needs_review} stories={len(clusters)} mode={mode}")

    # If needs_review, the graph would interrupt here (LangGraph interrupt). For now just mark.
    if status == "review":
        logger.info("[digest] Human-in-the-loop: awaiting approval (approve/edit/reject/regenerate)")

    return {"digest": digest, "clusters": clusters, "logs": [{"level": "info", "message": f"[digest] {status} with {len(clusters)} stories"}]}
