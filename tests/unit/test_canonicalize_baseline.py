"""Baseline canonicalization expectations (pre-identity module).

These tests document the *desired* canonicalization contract that
app/domain/documents/identity.py must satisfy. They currently test
the helper shim that will be promoted.
"""

from __future__ import annotations

import hashlib
import re
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse


TRACKING_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_content",
    "utm_term",
    "utm_id",
    "utm_source_platform",
    "utm_creative_format",
    "utm_marketing_tactic",
    "gclid",
    "fbclid",
    "igshid",
    "mc_cid",
    "mc_eid",
}


def canonicalize_url(url: str) -> str:
    """Reference implementation for test expectations (mirrors future app helper)."""
    parsed = urlparse(url.strip())
    scheme = parsed.scheme.lower() or "https"
    netloc = parsed.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    # strip tracking params
    q = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True) if k.lower() not in TRACKING_PARAMS]
    q.sort()
    path = parsed.path or "/"
    # remove trailing slash except root
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    return urlunparse((scheme, netloc, path, "", urlencode(q), ""))


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def test_canonicalize_strips_utm():
    assert canonicalize_url("https://example.com/page?utm_source=arxiv&utm_medium=email&x=1") == "https://example.com/page?x=1"


def test_canonicalize_lowercases_and_strips_www():
    assert canonicalize_url("https://WWW.Example.COM/page") == "https://example.com/page"


def test_canonicalize_removes_trailing_slash():
    assert canonicalize_url("https://example.com/page/") == "https://example.com/page"
    assert canonicalize_url("https://example.com/") == "https://example.com/"


def test_canonicalize_sorts_query_params():
    assert canonicalize_url("https://example.com/page?b=2&a=1") == "https://example.com/page?a=1&b=2"


def test_content_hash_is_stable():
    assert content_hash("hello") == hashlib.sha256(b"hello").hexdigest()
    assert content_hash("hello") != content_hash("Hello")
    assert len(content_hash("x")) == 64


def test_title_hash_normalizes_whitespace_and_case():
    def title_hash(title: str) -> str:
        normalized = re.sub(r"\s+", " ", title.strip().lower())
        return hashlib.sha256(normalized.encode()).hexdigest()

    assert title_hash("  Hello  World  ") == title_hash("hello world")
    assert title_hash("A") != title_hash("B")
