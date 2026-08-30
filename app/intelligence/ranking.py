"""Composite ranking — see docs/RANKING.md."""

from __future__ import annotations

import math
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

DEFAULT_WEIGHTS = {
    "semantic_relevance": 0.22,
    "freshness": 0.12,
    "novelty": 0.13,
    "source_quality": 0.14,
    "event_importance": 0.09,
    "trend_velocity": 0.10,
    "user_affinity": 0.14,
    "information_gain": 0.06,
}
DEFAULT_PENALTIES = {
    "repetition": 0.08,
    "topic_fatigue": 0.07,
    "low_quality": 0.10,
    "weak_evidence": 0.09,
}
DEFAULT_DIVERSITY_BONUS = 0.05


def _load_config(path: str = "config/ranking.yaml") -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def _freshness_score(published_at: str | None) -> float:
    if not published_at:
        return 0.5
    try:
        # handle ISO strings
        dt = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        age_hours = (datetime.now(UTC) - dt).total_seconds() / 3600
        # exponential decay half-life 72h
        return max(0.0, min(1.0, math.exp(-age_hours / 72)))
    except Exception:
        return 0.5


def _event_importance(event_type: str | None) -> float:
    weights = {
        "MODEL_RELEASE": 0.95,
        "RESEARCH_RESULT": 0.90,
        "BENCHMARK_RESULT": 0.88,
        "PAPER_RELEASE": 0.85,
        "SECURITY_INCIDENT": 0.95,
        "VULNERABILITY": 0.90,
        "REGULATION": 0.85,
        "ACQUISITION": 0.80,
        "FUNDING": 0.70,
        "PRODUCT_RELEASE": 0.75,
        "OPEN_SOURCE_RELEASE": 0.80,
        "COMPANY_ANNOUNCEMENT": 0.60,
    }
    return weights.get((event_type or "").upper(), 0.55)


def _source_quality_score(tier: str | None) -> float:
    return {"A": 0.95, "B": 0.75, "C": 0.45, "D": 0.25}.get((tier or "D"), 0.25)


def _information_gain(text: str) -> float:
    # cheap proxy: unique token ratio + length
    toks = re.findall(r"\w+", text.lower())
    if not toks:
        return 0.0
    uniq = len(set(toks)) / len(toks)
    length_factor = min(1.0, len(text) / 800)
    return round(0.5 * uniq + 0.5 * length_factor, 3)


def composite_score(
    *,
    semantic_relevance: float = 0.5,
    freshness: float | None = None,
    novelty: float = 0.5,
    source_quality: float = 0.5,
    event_importance: float = 0.5,
    trend_velocity: float = 0.5,
    user_affinity: float = 0.5,
    information_gain: float = 0.5,
    diversity_bonus: float = 0.0,
    repetition_penalty: float = 0.0,
    topic_fatigue: float = 0.0,
    low_quality_penalty: float = 0.0,
    weak_evidence_penalty: float = 0.0,
    weights: dict | None = None,
    penalties: dict | None = None,
) -> dict[str, float]:
    w = weights or DEFAULT_WEIGHTS
    p = penalties or DEFAULT_PENALTIES
    # weighted sum
    score = (
        w.get("semantic_relevance", 0.22) * semantic_relevance
        + w.get("freshness", 0.12) * (freshness if freshness is not None else 0.5)
        + w.get("novelty", 0.13) * novelty
        + w.get("source_quality", 0.14) * source_quality
        + w.get("event_importance", 0.09) * event_importance
        + w.get("trend_velocity", 0.10) * trend_velocity
        + w.get("user_affinity", 0.14) * user_affinity
        + w.get("information_gain", 0.06) * information_gain
        + (diversity_bonus)
        - p.get("repetition", 0.08) * repetition_penalty
        - p.get("topic_fatigue", 0.07) * topic_fatigue
        - p.get("low_quality", 0.10) * low_quality_penalty
        - p.get("weak_evidence", 0.09) * weak_evidence_penalty
    )
    final = max(0.0, min(1.0, score))
    return {
        "relevance": round(semantic_relevance, 3),
        "freshness": round(freshness if freshness is not None else 0.5, 3),
        "novelty": round(novelty, 3),
        "authority": round(source_quality, 3),
        "trend_velocity": round(trend_velocity, 3),
        "personal_affinity": round(user_affinity, 3),
        "information_gain": round(information_gain, 3),
        "final_score": round(final, 3),
    }


def score_document(doc: dict[str, Any], context: dict[str, Any] | None = None) -> dict[str, float]:
    """Convenience: compute breakdown from a Document dict + context."""
    ctx = context or {}
    cfg = _load_config()
    weights = cfg.get("weights")
    penalties = cfg.get("penalties")
    freshness = _freshness_score(doc.get("published_at"))
    info_gain = _information_gain(doc.get("title", "") + " " + doc.get("description", ""))
    event_imp = _event_importance(ctx.get("event_type"))
    src_q = _source_quality_score(doc.get("source_tier") or ctx.get("tier"))
    return composite_score(
        semantic_relevance=float(doc.get("relevance_score", doc.get("semantic_relevance", 0.5))),
        freshness=freshness,
        novelty=float(ctx.get("novelty", 0.7)),
        source_quality=src_q,
        event_importance=event_imp,
        trend_velocity=float(ctx.get("trend_velocity", 0.5)),
        user_affinity=float(ctx.get("user_affinity", 0.5)),
        information_gain=info_gain,
        diversity_bonus=float(ctx.get("diversity_bonus", 0)),
        repetition_penalty=float(ctx.get("repetition_penalty", 0)),
        topic_fatigue=float(ctx.get("topic_fatigue", 0)),
        low_quality_penalty=float(ctx.get("low_quality_penalty", 0)),
        weak_evidence_penalty=float(ctx.get("weak_evidence_penalty", 0)),
        weights=weights,
        penalties=penalties,
    )
