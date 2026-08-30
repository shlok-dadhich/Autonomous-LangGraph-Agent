"""Diversity + fatigue."""

from __future__ import annotations

def diversity_bonus(sources: list[str]) -> float:
    return 0.05 if len(set(sources)) > 2 else 0.0
