"""Recommendation + knowledge gap detection."""

from __future__ import annotations

from collections import Counter


def recommend_topics(user_affinity: dict[str, float], seen_topics: set[str], all_topics: list[str], top_k: int = 3) -> list[str]:
    # recommend high-affinity topics not yet followed but seen in corpus
    candidates = [t for t in all_topics if t not in seen_topics]
    # simple: sort by affinity if exists else alphabetical
    candidates.sort(key=lambda t: user_affinity.get(t, 0), reverse=True)
    return candidates[:top_k]


def detect_knowledge_gap(viewed_topics: Counter, corpus_topics: Counter, threshold: int = 5) -> list[str]:
    """Gap when user sees topic B many times but rarely clicks."""
    gaps = []
    for topic, count in corpus_topics.items():
        if count >= threshold and viewed_topics.get(topic, 0) < 2:
            gaps.append(topic)
    return gaps[:3]


def explain_gap(topic: str) -> dict:
    return {
        "topic": topic,
        "recommendation": f"Background explainer for {topic}",
        "resources": [f"Foundational paper on {topic}", f"Tech guide: {topic} implementation", f"Timeline: evolution of {topic}"],
    }
