"""Baseline tests for URL identity / canonicalization helpers.

These pin the current behavior in src/graph/nodes.py so that the
future app/domain/documents/identity.py can be verified against them.
"""

from __future__ import annotations


def test_has_verified_url_accepts_http_and_https():
    from src.graph.nodes import _has_verified_url

    assert _has_verified_url({"url": "https://example.com/a"}) is True
    assert _has_verified_url({"url": "http://example.com/a"}) is True
    assert _has_verified_url({"url": "ftp://example.com/a"}) is False
    assert _has_verified_url({"url": ""}) is False
    assert _has_verified_url({"url": "not-a-url"}) is False
    assert _has_verified_url({"url": "https://"} ) is False
    assert _has_verified_url({}) is False


def test_normalize_allowed_domains_strips_www_and_lowercases():
    from src.graph.nodes import _normalize_allowed_domains

    assert _normalize_allowed_domains(["https://WWW.Reuters.com/path", "reuters.com", ""]) == {"reuters.com"}
    assert _normalize_allowed_domains(["www.example.com"]) == {"example.com"}
    assert _normalize_allowed_domains(["HTTP://Example.COM"]) == {"example.com"}
    assert _normalize_allowed_domains([]) == set()
    assert _normalize_allowed_domains(["   "]) == set()


def test_is_url_allowed_with_empty_allowlist():
    from src.graph.nodes import _is_url_allowed

    assert _is_url_allowed("https://anything.example/page", set()) is True
    assert _is_url_allowed("https://reuters.com/world", set()) is True


def test_is_url_allowed_matches_exact_and_subdomain():
    from src.graph.nodes import _is_url_allowed

    allowed = {"reuters.com"}
    assert _is_url_allowed("https://reuters.com/world", allowed) is True
    assert _is_url_allowed("https://www.reuters.com/world", allowed) is True
    assert _is_url_allowed("https://sub.reuters.com/page", allowed) is True
    assert _is_url_allowed("https://random-blog.example/post", allowed) is False
    assert _is_url_allowed("https://notreuters.com/", allowed) is False
    assert _is_url_allowed("https://reuters.com.evil.com/", allowed) is False
    assert _is_url_allowed("", allowed) is False


def test_is_url_allowed_case_insensitive():
    from src.graph.nodes import _is_url_allowed

    allowed = {"example.com"}
    assert _is_url_allowed("https://EXAMPLE.COM/page", allowed) is True
    assert _is_url_allowed("https://Sub.Example.COM/page", allowed) is True


def test_profile_to_text_joins_topics_and_keywords():
    from src.graph.nodes import _profile_to_text

    assert _profile_to_text({"topics": ["RAG"], "keywords": ["retrieval"]}) == "RAG. retrieval"
    assert _profile_to_text({"topics": [], "keywords": []}) == ""
    assert _profile_to_text({"topics": ["A", "B"], "keywords": []}) == "A. B"
    # falsy items are filtered
    assert _profile_to_text({"topics": ["", None, "LLM"], "keywords": [""]}) == "LLM"


def test_is_diversity_source():
    from src.graph.nodes import _is_diversity_source

    assert _is_diversity_source("huggingface-daily") is True
    assert _is_diversity_source("huggingface-blog") is True
    assert _is_diversity_source("anthropic-newsroom") is True
    assert _is_diversity_source("tavily") is True
    assert _is_diversity_source("reddit") is True
    assert _is_diversity_source("arxiv") is False
    assert _is_diversity_source("rss-feed") is False


def test_select_articles_filters_unverified_urls():
    from src.graph.nodes import _select_articles_for_newsletter

    scored = [
        {"title": "good", "url": "https://example.com/1", "source": "arxiv", "relevance_score": 0.9},
        {"title": "bad url", "url": "not-a-url", "source": "arxiv", "relevance_score": 0.99},
        {"title": "no url", "url": "", "source": "arxiv", "relevance_score": 0.99},
    ]
    selected = _select_articles_for_newsletter(
        scored_articles=scored,
        interest_profile={"topics": ["RAG"]},
        threshold=0.45,
        max_filtered_articles=6,
    )
    assert len(selected) == 1
    assert selected[0]["url"] == "https://example.com/1"
