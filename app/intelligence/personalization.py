"""Personalization — long/recent/session with decay."""

from __future__ import annotations

import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field

# decay half-lives
HALF_LIFE_RECENT_DAYS = 14
HALF_LIFE_LONG_DAYS = 90


def _decay_weight(age_days: float, half_life: float) -> float:
    return math.pow(0.5, age_days / half_life)


@dataclass
class Interaction:
    action: str
    target_type: str
    target_id: str
    timestamp: float  # epoch seconds
    meta: dict = field(default_factory=dict)


@dataclass
class UserProfileState:
    topic_affinity: dict[str, float] = field(default_factory=dict)
    entity_affinity: dict[str, float] = field(default_factory=dict)
    source_affinity: dict[str, float] = field(default_factory=dict)


def aggregate_profile(
    interactions: list[Interaction],
    now: float | None = None,
) -> dict[str, UserProfileState]:
    """Aggregate into long/recent/session slices with decay."""
    now = now or time.time()
    long_state = UserProfileState()
    recent_state = UserProfileState()
    session_state = UserProfileState()

    # weights per action (explicit feedback stronger)
    action_weight = {
        "SAVE": 1.5,
        "LIKE": 1.2,
        "FOLLOW": 1.5,
        "CLICK": 1.0,
        "OPEN": 0.6,
        "READ_DURATION": 0.8,
        "DISLIKE": -1.2,
        "HIDE": -1.0,
        "MUTE_TOPIC": -1.5,
        "SKIP": -0.4,
    }

    for inter in interactions:
        age_days = (now - inter.timestamp) / 86400
        w_long = _decay_weight(age_days, HALF_LIFE_LONG_DAYS)
        w_recent = _decay_weight(age_days, HALF_LIFE_RECENT_DAYS)
        # session = last 2 hours only
        w_session = 1.0 if age_days < 2 / 24 else 0.0

        aw = action_weight.get(inter.action.upper(), 0.5)
        topic = inter.meta.get("topic")
        entity = inter.meta.get("entity")
        source = inter.meta.get("source")

        for state, w in [(long_state, w_long), (recent_state, w_recent), (session_state, w_session)]:
            if w == 0:
                continue
            if topic:
                state.topic_affinity[topic] = state.topic_affinity.get(topic, 0) + aw * w
            if entity:
                state.entity_affinity[entity] = state.entity_affinity.get(entity, 0) + aw * w
            if source:
                state.source_affinity[source] = state.source_affinity.get(source, 0) + aw * w

    return {"long": long_state, "recent": recent_state, "session": session_state}


def user_affinity_score(
    profile_slices: dict[str, UserProfileState],
    doc_topics: list[str],
    doc_entities: list[str],
    doc_source: str,
) -> float:
    """Blend long/recent/session affinities into 0-1 affinity."""
    # weights: recent matters most, then long, session is bonus
    long_w, recent_w, session_w = 0.3, 0.5, 0.2
    scores = []
    for name, w in [("long", long_w), ("recent", recent_w), ("session", session_w)]:
        state = profile_slices.get(name)
        if not state:
            continue
        s = 0.0
        for t in doc_topics:
            s += state.topic_affinity.get(t, 0)
        for e in doc_entities:
            s += state.entity_affinity.get(e, 0)
        s += state.source_affinity.get(doc_source, 0)
        # normalize via tanh to 0-1
        norm = (math.tanh(s) + 1) / 2  # -inf..inf -> 0..1 (0.5 when s=0)
        scores.append(norm * w)
    if not scores:
        return 0.5
    return round(sum(scores) / sum([long_w, recent_w, session_w][: len(scores)]), 3)
