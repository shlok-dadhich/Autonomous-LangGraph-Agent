"""Novelty detection — new vs repeat."""

from __future__ import annotations

def is_novel(content_hash: str, seen_hashes: set[str]) -> bool:
    return content_hash not in seen_hashes
