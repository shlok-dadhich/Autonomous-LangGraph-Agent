"""Diversity + fatigue."""

from __future__ import annotations

from collections import Counter


def diversity_bonus(publishers: list[str], source_types: list[str]) -> float:
    """Bonus when story adds perspective diversity."""
    unique_pub = len(set(publishers))
    unique_types = len(set(source_types))
    bonus = 0.0
    if unique_pub >= 3:
        bonus += 0.03
    if unique_types >= 2:
        bonus += 0.02
    return round(min(0.05, bonus), 3)


def topic_fatigue_penalty(topic_counts: Counter, topic: str, threshold: int = 5) -> float:
    """Penalty when topic has been overexposed recently."""
    count = topic_counts.get(topic, 0)
    if count >= threshold:
        return min(1.0, (count - threshold + 1) * 0.3)
    return 0.0


def repetition_penalty(content_hash: str, recent_hashes: list[str]) -> float:
    return 1.0 if content_hash in recent_hashes else 0.0
