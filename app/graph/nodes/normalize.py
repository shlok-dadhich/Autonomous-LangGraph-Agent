"""Normalize RawDocument -> canonical Document dict."""

from __future__ import annotations

from datetime import UTC, datetime

from loguru import logger


def normalize_documents_node(state: dict) -> dict:
    raw = state.get("raw_articles", [])
    docs = []
    for a in raw:
        if not isinstance(a, dict):
            continue
        title = str(a.get("title","")).strip()
        url = str(a.get("url","")).strip()
        if not title or not url:
            continue
        docs.append({
            "title": title,
            "url": url,
            "description": str(a.get("description","")).strip(),
            "source": str(a.get("source","")).strip() or "unknown",
            "published_at": a.get("published_date") or a.get("published_at"),
            "fetched_at": datetime.now(UTC).isoformat(),
            "author": a.get("author"),
        })
    logger.info(f"[normalize] {len(raw)} raw -> {len(docs)} documents")
    return {"documents": docs, "logs": [{"level": "info", "message": f"[normalize] normalized {len(docs)} documents"}]}
