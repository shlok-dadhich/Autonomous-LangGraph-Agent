from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest


@dataclass
class _DummyResponse:
    text: str

    def raise_for_status(self) -> None:
        return None


def test_rss_sources_include_anthropic_and_hf(monkeypatch):
    from src.tools import rss_client

    def fake_get(url, timeout=15):
        assert url == "https://www.anthropic.com/news"
        return _DummyResponse(
            text=(
                '<html><body>'
                '<a href="https://www.anthropic.com/news/alpha">Anthropic Alpha</a>'
                '<a href="https://www.anthropic.com/news/beta">Anthropic Beta</a>'
                '</body></html>'
            )
        )

    def fake_parse(url):
        assert url == "https://huggingface.co/blog/feed.xml"
        return SimpleNamespace(
            entries=[
                SimpleNamespace(
                    title="Hugging Face Post",
                    link="https://huggingface.co/blog/example",
                    summary="HF summary",
                    published="Wed, 01 Apr 2026 00:00:00 GMT",
                )
            ],
            bozo=False,
        )

    monkeypatch.setattr(rss_client.requests, "get", fake_get)
    monkeypatch.setattr(rss_client.feedparser, "parse", fake_parse)

    result = rss_client.fetch_rss_sources(
        feed_specs=[
            {
                "name": "anthropic_news",
                "url": "https://www.anthropic.com/news",
                "source": "anthropic-newsroom",
                "parser": "html",
                "limit": 2,
            },
            {
                "name": "huggingface_blog",
                "url": "https://huggingface.co/blog/feed.xml",
                "source": "huggingface-blog",
                "parser": "rss",
                "limit": 1,
            },
        ]
    )

    articles = result["raw_articles"]
    sources = {article["source"] for article in articles}

    assert "anthropic-newsroom" in sources
    assert "huggingface-blog" in sources
    assert len(articles) == 3
    assert all(article["url"].startswith("https://") for article in articles)


def test_filter_node_keeps_diversity_and_dedupes(monkeypatch):
    from src.graph import nodes

    class DummyRanker:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def score_articles(self, interest_profile_text, unique_articles):
            return [dict(article) for article in unique_articles]

    def fake_similarity_prune(scored_articles, similarity_threshold=0.9):
        kept = []
        seen_titles = set()
        for article in scored_articles:
            title = article["title"].strip().lower()
            if title in seen_titles:
                continue
            seen_titles.add(title)
            kept.append(article)
        return kept

    monkeypatch.setattr(nodes, "RelevanceRanker", DummyRanker)
    monkeypatch.setattr(nodes, "_prune_by_similarity_with_source_preference", fake_similarity_prune)

    state = {
        "interest_profile": {
            "topics": ["RAG"],
            "keywords": ["retrieval"],
            "max_filtered_articles": 6,
        },
        "unique_articles": [
            {"title": "RAG pipeline", "url": "https://example.com/1", "description": "a", "source": "arxiv", "relevance_score": 0.93},
            {"title": "RAG pipeline", "url": "https://example.com/2", "description": "b", "source": "arxiv", "relevance_score": 0.91},
            {"title": "HF release", "url": "https://example.com/3", "description": "c", "source": "huggingface-blog", "relevance_score": 0.42},
            {"title": "Tavily note", "url": "https://example.com/4", "description": "d", "source": "tavily", "relevance_score": 0.44},
        ],
    }

    result = nodes.filter_node(state)
    filtered = result["filtered_articles"]

    assert len(filtered) <= 3
    assert len({article["title"] for article in filtered}) == len(filtered)
    assert any(article["source"] in {"huggingface-blog", "tavily"} for article in filtered)


def test_fallback_search_relaxes_threshold_before_broad_search(monkeypatch):
    from src.graph import nodes

    class DummyRanker:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name

        def score_articles(self, interest_profile_text, unique_articles):
            return [dict(article) for article in unique_articles]

    monkeypatch.setattr(nodes, "RelevanceRanker", DummyRanker)
    monkeypatch.setattr(nodes, "_prune_by_similarity_with_source_preference", lambda scored_articles, similarity_threshold=0.9: scored_articles)

    def fail_if_called():
        raise AssertionError("broader Tavily fallback should not run when relaxed ranking restores content")

    monkeypatch.setattr(nodes, "_fetch_tavily_buffer_safe", fail_if_called)

    state = {
        "interest_profile": {"topics": ["RAG"], "keywords": ["retrieval"]},
        "filtered_articles": [
            {"title": "Existing 1", "url": "https://example.com/a", "description": "a", "source": "arxiv", "relevance_score": 0.9},
            {"title": "Existing 2", "url": "https://example.com/b", "description": "b", "source": "arxiv", "relevance_score": 0.88},
        ],
        "unique_articles": [
            {"title": "Existing 1", "url": "https://example.com/a", "description": "a", "source": "arxiv", "relevance_score": 0.9},
            {"title": "Existing 2", "url": "https://example.com/b", "description": "b", "source": "arxiv", "relevance_score": 0.88},
            {"title": "Relaxed match", "url": "https://example.com/c", "description": "c", "source": "huggingface-blog", "relevance_score": 0.36},
        ],
    }

    result = nodes.fallback_search_node(state)

    assert len(result["filtered_articles"]) >= 3
    assert any(article["url"] == "https://example.com/c" for article in result["filtered_articles"])


def test_writer_preserves_original_url_and_tracks_links(monkeypatch, tmp_path):
    from src.core.writer import NewsletterWriter
    from src.services.template_service import TemplateService

    writer = NewsletterWriter(api_key="test-key")

    def fake_invoke_groq(messages, max_tokens):
        return (
            '[{"title":"Modeling RAG","url":"https://malicious.example/guess","source":"arxiv","relevance_score":0.9,'
            '"what":"What: Introduces a better retriever.",'
            '"how":"How: Uses dual encoders and reranking.",'
            '"personalized_insight":"**Personalized Insight:** This improves your RAG stack."}]'
        )

    monkeypatch.setattr(writer, "_invoke_groq", fake_invoke_groq)

    article = {
        "title": "Modeling RAG",
        "url": "https://example.com/original-paper",
        "source": "arxiv",
        "description": "A paper about RAG systems.",
        "relevance_score": 0.9,
    }

    enriched = writer.generate_analysis({"topics": ["RAG"], "keywords": ["retrieval"]}, [article])

    assert enriched[0]["url"] == "https://example.com/original-paper"
    assert enriched[0]["summary_lines"][0].startswith("What:")
    assert enriched[0]["summary_lines"][1].startswith("How:")
    assert enriched[0]["personalized_insight"].startswith("**Personalized Insight:**")

    template_dir = tmp_path / "templates"
    template_dir.mkdir()
    (template_dir / "email_body.html").write_text(
        '<a href="{{ articles[0].tracked_url or articles[0].url }}">Link</a>',
        encoding="utf-8",
    )

    rendered = TemplateService(template_dir=template_dir).render_newsletter(
        enriched_articles=enriched,
        current_date="Apr 02, 2026",
        newsletter_title="AI Weekly Intelligence",
        top_topic="RAG",
        build_label="2026-04-02",
    )

    assert "utm_source=arxiv" in rendered
    assert "utm_campaign=ai_weekly_intelligence" in rendered
    assert "utm_medium=email" in rendered


def test_database_connection_uses_wal(tmp_path):
    from src.core.database import create_sqlite_connection

    db_path = tmp_path / "history.db"
    conn = create_sqlite_connection(db_path)
    try:
        journal_mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
        assert journal_mode.lower() == "wal"
    finally:
        conn.close()
