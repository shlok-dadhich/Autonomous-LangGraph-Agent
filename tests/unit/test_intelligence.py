"""Tests for Phase 2 intelligence modules."""

from app.intelligence.clustering import ClusterInput, cluster_documents
from app.intelligence.events import classify_event
from app.intelligence.entities import extract_entities
from app.intelligence.evidence import evaluate_evidence, detect_contradiction
from app.intelligence.source_quality import score_source
from app.intelligence.claims import extract_claims


def test_clustering_groups_same_story():
    docs = [
        ClusterInput(id="1", title="OpenAI releases GPT-5", text="official announcement", entities=["OpenAI"]),
        ClusterInput(id="2", title="OpenAI releases GPT-5 — TechCrunch coverage", text="official announcement", entities=["OpenAI"]),
        ClusterInput(id="3", title="Anthropic releases new model", text="different", entities=["Anthropic"]),
    ]
    clusters = cluster_documents(docs, title_threshold=0.5)
    # first two should cluster, third separate
    assert len(clusters) == 2
    # check multi-doc cluster has higher confidence
    multi = max(clusters, key=lambda c: len(c.document_ids))
    assert len(multi.document_ids) == 2
    assert multi.cluster_confidence >= 0.82


def test_event_classification():
    ev = classify_event("OpenAI releases GPT-5 model", "We are introducing GPT-5 today")
    assert ev.event_type == "MODEL_RELEASE"
    ev2 = classify_event("New paper on arxiv: RAG breakthrough", "arxiv preprint shows...")
    assert ev2.event_type == "PAPER_RELEASE"


def test_entity_extraction():
    ents = extract_entities("OpenAI and Anthropic announce partnership", "GitHub repo https://github.com/openai/gpt-5")
    names = {e.canonical_name for e in ents}
    assert "OpenAI" in names
    assert "Anthropic" in names
    # repo captured
    assert any("github.com" in n for n in names)


def test_evidence_high_confidence():
    bundle = evaluate_evidence(["arxiv.org", "techcrunch.com", "reuters.com", "github.com"], ["A", "B", "B", "C"])
    assert bundle.confidence in ("High", "Medium")


def test_contradiction_detection():
    assert detect_contradiction(["Company reports 50% faster", "Independent benchmark reports 17% faster"]) is not None
    assert detect_contradiction(["No numbers here", "Also none"]) is None


def test_source_quality():
    s = score_source("https://arxiv.org/abs/1234")
    assert s.tier == "A"
    assert s.label == "PRIMARY"
    s2 = score_source("https://reddit.com/r/MachineLearning")
    assert s2.tier == "C"


def test_claims_extraction():
    claims = extract_claims("Title here", "Sentence one. Sentence two with 50% improvement. Short.", "doc1", "https://example.com")
    assert len(claims) >= 1
    assert all(c.evidence_refs for c in claims)
    assert any("http" in r["url"] for c in claims for r in c.evidence_refs)
