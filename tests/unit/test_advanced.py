"""Tests for Phase 6 advanced intelligence."""

from datetime import datetime, timezone, timedelta

from app.intelligence.trends import TrendPoint, velocity, classify_state
from app.intelligence.evidence import detect_contradiction
from app.services.recommendation_service import recommend_topics, detect_knowledge_gap
from app.workers.delivery import analysis_budget, adaptive_send_window


def test_trend_velocity():
    now = datetime.now(timezone.utc)
    hist = [TrendPoint(target_id="AI Agents", timestamp=now - timedelta(days=d), mentions=5, unique_sources=3) for d in [6, 5, 4, 3, 2, 1]]
    curr = TrendPoint(target_id="AI Agents", timestamp=now, mentions=12, unique_sources=6)
    v = velocity(hist, curr)
    assert v > 0.5
    assert classify_state(v, curr.mentions) in ("ACCELERATING", "RISING", "BREAKING")


def test_contradiction():
    assert detect_contradiction(["50% faster", "17% faster"]) is not None
    assert detect_contradiction(["no numbers"]) is None


def test_recommendations():
    rec = recommend_topics({"RAG": 1.0, "LLM": 0.5}, {"LLM"}, ["RAG", "LLM", "Agents"])
    assert "RAG" in rec
    from collections import Counter
    gaps = detect_knowledge_gap(Counter({"LLM": 1}), Counter({"RAG": 6, "LLM": 6}))
    assert "RAG" in gaps


def test_adaptive_budget():
    assert analysis_budget(0.9)["sources"] == 5
    assert analysis_budget(0.5)["sources"] == 1
    assert adaptive_send_window(["09:15", "09:30", "10:00"]) == "09:00"
