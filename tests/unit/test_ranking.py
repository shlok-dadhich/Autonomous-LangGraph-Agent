"""Tests for Phase 3 ranking."""

from app.intelligence.ranking import composite_score, score_document
from app.intelligence.novelty import novelty_score
from app.intelligence.diversity import diversity_bonus, topic_fatigue_penalty
from app.graph.nodes.quality_gate import quality_gate_node
from app.providers.reranking.cohere import CohereReranker


def test_composite_score_breakdown():
    br = composite_score(semantic_relevance=0.9, freshness=0.8, novelty=0.9, source_quality=0.95, event_importance=0.9, trend_velocity=0.6, user_affinity=0.8)
    assert 0 <= br["final_score"] <= 1
    assert br["relevance"] == 0.9
    assert "authority" in br


def test_score_document_uses_freshness():
    doc = {"title": "Test paper", "description": "Some content about AI", "published_at": "2026-08-30T00:00:00+00:00", "relevance_score": 0.8, "source_tier": "A"}
    br = score_document(doc, {"event_type": "PAPER_RELEASE"})
    assert br["final_score"] > 0
    assert br["authority"] == 0.95


def test_novelty_and_diversity():
    assert novelty_score("hash123", set()) > 0.5
    assert novelty_score("hash123", {"hash123"}) < 0.5
    assert diversity_bonus(["a", "b", "c"], ["A", "B", "C"]) == 0.05
    assert diversity_bonus(["a"], ["A"]) == 0.0
    from collections import Counter
    assert topic_fatigue_penalty(Counter({"RAG": 6}), "RAG") > 0


def test_reranker_fallback():
    rr = CohereReranker(api_key=None)
    docs = ["AI agents breakthrough", "cooking recipe", "LLM reasoning paper"]
    res = rr.rerank("AI agents", docs, top_k=2)
    assert len(res) == 2
    assert res[0].score >= res[1].score


def test_quality_gate_allows_zero():
    state = {"clusters": [{"cluster_id": "1", "cluster_confidence": 0.4}, {"cluster_id": "2", "cluster_confidence": 0.3}], "documents": []}
    out = quality_gate_node(state)
    assert out["clusters"] == []  # no filler, allows 0
    assert "digest_skipped_low_signal" in out["logs"][0]["message"]


def test_quality_gate_keeps_high():
    state = {
        "clusters": [{"cluster_id": "1", "cluster_confidence": 0.9, "document_ids": ["d1"]}],
        "documents": [{"document_id": "d1", "final_score": 0.9}],
    }
    out = quality_gate_node(state)
    assert len(out["clusters"]) == 1
