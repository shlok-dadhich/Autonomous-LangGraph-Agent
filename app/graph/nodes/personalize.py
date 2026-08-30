"""Personalization node — blends user affinity into ranking."""

from __future__ import annotations

from loguru import logger

from app.intelligence.personalization import Interaction, aggregate_profile, user_affinity_score


def personalize_node(state: dict) -> dict:
    docs = state.get("documents", [])
    # interactions can be passed in state for testing; in prod loaded from DB
    raw_interactions = state.get("interactions", [])
    interactions = []
    for r in raw_interactions:
        interactions.append(
            Interaction(
                action=r.get("action", "CLICK"),
                target_type=r.get("target_type", "document"),
                target_id=r.get("target_id", ""),
                timestamp=r.get("timestamp", 0),
                meta=r.get("meta", r.get("context", {})),
            )
        )

    slices = aggregate_profile(interactions) if interactions else None

    for d in docs:
        topics = d.get("topics", []) or [d.get("title", "")[:30]]
        entities = [e.get("canonical_name") for e in state.get("entities", []) if e.get("document_id") == d.get("document_id")] or d.get("entities", [])
        source = d.get("source", "")
        if slices:
            aff = user_affinity_score(slices, topics, entities, source)
        else:
            aff = 0.5  # neutral
        # Update ranking breakdown
        breakdown = d.get("ranking_breakdown", {})
        breakdown["personal_affinity"] = aff
        # Recompute final_score with updated affinity (simple blend)
        # we keep original final_score but nudge toward affinity
        orig = d.get("final_score", 0.5)
        d["final_score"] = round(0.85 * orig + 0.15 * aff, 3)
        d["ranking_breakdown"] = breakdown
        d["user_affinity"] = aff

    # Re-sort by final_score
    docs.sort(key=lambda x: x.get("final_score", 0), reverse=True)
    logger.info(f"[personalize] applied affinity to {len(docs)} docs")
    return {"documents": docs, "logs": [{"level": "info", "message": f"[personalize] {len(docs)} docs personalized"}]}
