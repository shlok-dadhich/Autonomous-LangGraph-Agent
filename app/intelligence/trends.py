"""Trend detection — velocity + state machine."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta


@dataclass
class TrendPoint:
    target_id: str
    timestamp: datetime
    mentions: int
    unique_sources: int


def velocity(prev: list[TrendPoint], curr: TrendPoint) -> float:
    if not prev:
        return 0.0
    # avg mentions/day over last 7d
    week_ago = curr.timestamp - timedelta(days=7)
    recent = [p for p in prev if p.timestamp >= week_ago]
    if not recent:
        return float(curr.mentions)
    avg = sum(p.mentions for p in recent) / len(recent)
    return round((curr.mentions - avg) / max(1, avg), 3)


def classify_state(velocity: float, mentions: int) -> str:
    if mentions >= 10 and velocity > 0.5:
        return "ACCELERATING"
    if mentions >= 5 and velocity > 0.2:
        return "RISING"
    if mentions >= 5 and abs(velocity) <= 0.2:
        return "STABLE"
    if mentions >= 3 and velocity > 1.0:
        return "BREAKING"
    if velocity < -0.3:
        return "DECLINING"
    if velocity > 0.3 and mentions < 3:
        return "EMERGING"
    return "STABLE"


def detect_trend(target_id: str, history: list[TrendPoint], current: TrendPoint) -> dict:
    v = velocity(history, current)
    state = classify_state(v, current.mentions)
    return {"target_id": target_id, "velocity": v, "state": state, "mentions": current.mentions, "explanation": f"{state} ({v:+.2f} vs 7d avg, {current.mentions} mentions)"}
