"""Tests for app/domain/documents/identity."""

from app.domain.documents.identity import canonicalize_url, content_hash, document_id, title_hash


def test_canonicalize_strips_tracking():
    assert canonicalize_url("https://example.com/page?utm_source=arxiv&x=1") == "https://example.com/page?x=1"
    assert canonicalize_url("https://example.com/page?gclid=abc&fbclid=123") == "https://example.com/page"


def test_canonicalize_strips_www_and_lowercases():
    assert canonicalize_url("https://WWW.Example.COM/page") == "https://example.com/page"


def test_canonicalize_sort_and_slash():
    assert canonicalize_url("https://example.com/page?b=2&a=1") == "https://example.com/page?a=1&b=2"
    assert canonicalize_url("https://example.com/page/") == "https://example.com/page"
    assert canonicalize_url("https://example.com/") == "https://example.com/"


def test_hashes():
    assert len(content_hash("hello")) == 64
    assert content_hash("hello") != content_hash("Hello")
    assert title_hash("  Hello  World  ") == title_hash("hello world")


def test_document_id_deterministic():
    c = canonicalize_url("https://example.com/a")
    h = content_hash("some content")
    assert document_id(c, h) == document_id(c, h)
    assert document_id(c) != document_id(canonicalize_url("https://example.com/b"))
