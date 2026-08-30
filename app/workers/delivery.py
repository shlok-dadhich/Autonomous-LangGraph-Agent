"""Delivery worker — adaptive send window + experiments + cost budget."""

from __future__ import annotations

import random

from app.core.feature_flags import is_enabled


def pick_subject_variants() -> dict:
    variants = {
        "informational": "Your AI Brief — Top developments today",
        "curiosity": "What you missed in AI today",
        "executive": "AI Intelligence Brief — Executive Summary",
        "technical": "RAG / Agents / LLMs — Technical Digest",
    }
    # experiment A/B: feature flag chooses variant
    chosen = "informational"
    if is_enabled("new_email_template"):
        chosen = random.choice(list(variants.keys()))
    return {"chosen": chosen, "title": variants[chosen], "variants": variants}


def adaptive_send_window(open_times: list[str]) -> str:
    """Learn typical open time; return recommended HH:MM."""
    if not open_times:
        return "08:00"
    # naive: most common hour
    from collections import Counter

    hours = [t.split(":")[0] for t in open_times]
    most = Counter(hours).most_common(1)[0][0]
    return f"{most}:00"


def analysis_budget(final_score: float) -> dict:
    """Cost controls: cheap vs deep evidence verification."""
    if final_score >= 0.85:
        return {"sources": 5, "model": "reasoning", "contradiction_check": True}
    if final_score >= 0.65:
        return {"sources": 3, "model": "reasoning", "contradiction_check": False}
    return {"sources": 1, "model": "fast", "contradiction_check": False}
