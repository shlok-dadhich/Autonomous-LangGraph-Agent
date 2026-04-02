"""Feed harvesting helpers for RSS and newsroom-style HTML sources."""

from __future__ import annotations

import html
import os
import re
import sys
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.reliability import safe_execute

try:
    import feedparser
except ImportError:
    logger.warning("feedparser not installed. Install with: pip install feedparser")
    feedparser = None

try:
    import requests
except ImportError:
    logger.warning("requests not installed. Install with: pip install requests")
    requests = None


DEFAULT_RSS_SOURCES: List[Dict[str, Any]] = [
    {
        "name": "anthropic_news",
        "url": "https://www.anthropic.com/news",
        "source": "anthropic-newsroom",
        "parser": "html",
        "limit": 5,
    },
    {
        "name": "huggingface_blog",
        "url": "https://huggingface.co/blog/feed.xml",
        "source": "huggingface-blog",
        "parser": "rss",
        "limit": 5,
    },
]


class _AnthropicNewsParser(HTMLParser):
    """Minimal parser for Anthropic newsroom article cards."""

    def __init__(self) -> None:
        super().__init__()
        self._capture_depth = 0
        self._capture_href: Optional[str] = None
        self._capture_text: List[str] = []
        self.items: List[Dict[str, str]] = []

    def handle_starttag(self, tag: str, attrs):  # type: ignore[override]
        if tag != "a":
            return

        href = dict(attrs).get("href", "")
        if not href:
            return

        if not ("/news/" in href or "/research/" in href):
            return

        self._capture_href = href
        self._capture_depth = 1
        self._capture_text = []

    def handle_data(self, data: str) -> None:
        if self._capture_href:
            text = data.strip()
            if text:
                self._capture_text.append(text)

    def handle_endtag(self, tag: str) -> None:
        if tag != "a" or not self._capture_href:
            return

        text = html.unescape(" ".join(self._capture_text)).strip()
        if text:
            self.items.append({"title": text, "url": self._capture_href})

        self._capture_href = None
        self._capture_text = []
        self._capture_depth = 0


def _normalize_feed_spec(feed_spec: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": str(feed_spec.get("name", feed_spec.get("source", "rss_feed"))),
        "url": str(feed_spec.get("url", "")).strip(),
        "source": str(feed_spec.get("source", feed_spec.get("name", "rss-feed"))),
        "parser": str(feed_spec.get("parser", "rss")).strip().lower(),
        "limit": int(feed_spec.get("limit", 5)),
    }


def _canonical_source_name(source_name: str) -> str:
    lowered = source_name.lower()
    if "anthropic" in lowered:
        return "anthropic-newsroom"
    if "huggingface" in lowered:
        return "huggingface-blog"
    return source_name


def _parse_rss_feed(feed_url: str, source_name: str, limit: int) -> List[Dict[str, Any]]:
    if feedparser is None:
        raise ImportError("feedparser is required for RSS parsing")

    feed = feedparser.parse(feed_url)
    articles: List[Dict[str, Any]] = []
    entries = list(getattr(feed, "entries", []))[:limit]

    for entry in entries:
        title = str(getattr(entry, "title", "")).strip()
        url = str(getattr(entry, "link", "")).strip()
        if not title or not url:
            continue

        description = str(getattr(entry, "summary", "")).strip() or title
        published = getattr(entry, "published", "") or getattr(entry, "updated", "")
        articles.append(
            {
                "title": title,
                "url": url,
                "description": description[:500],
                "source": _canonical_source_name(source_name),
                "published_date": str(published).strip() or datetime.now(timezone.utc).isoformat(),
            }
        )

    return articles


def _parse_html_index(feed_url: str, source_name: str, limit: int) -> List[Dict[str, Any]]:
    if requests is None:
        raise ImportError("requests is required for HTML feed parsing")

    response = requests.get(feed_url, timeout=15)
    response.raise_for_status()

    parser = _AnthropicNewsParser()
    parser.feed(response.text)

    articles: List[Dict[str, Any]] = []
    seen_urls = set()
    for item in parser.items:
        url = item["url"]
        if url in seen_urls:
            continue
        seen_urls.add(url)
        articles.append(
            {
                "title": item["title"],
                "url": url,
                "description": item["title"],
                "source": _canonical_source_name(source_name),
                "published_date": datetime.now(timezone.utc).isoformat(),
            }
        )
        if len(articles) >= limit:
            break

    return articles


@safe_execute(source_name="rss_sources", max_retries=2)
def fetch_rss_sources(
    feed_specs: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Fetch RSS or newsroom HTML feeds into the canonical article schema."""
    specs = [_normalize_feed_spec(feed_spec) for feed_spec in (feed_specs or DEFAULT_RSS_SOURCES)]
    all_articles: List[Dict[str, Any]] = []

    for spec in specs:
        feed_url = spec["url"]
        source_name = spec["source"]
        parser_name = spec["parser"]
        limit = spec["limit"]

        if not feed_url:
            logger.warning(f"[rss] Skipping empty feed URL for {source_name}")
            continue

        logger.info(f"[rss] Fetching {source_name} from {feed_url} using {parser_name}")

        if parser_name == "html":
            items = _parse_html_index(feed_url, source_name, limit)
        else:
            items = _parse_rss_feed(feed_url, source_name, limit)

        all_articles.extend(items)

    return all_articles
