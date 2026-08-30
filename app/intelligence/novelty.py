"""Novelty detection — new vs repeat vs update."""

from __future__ import annotations

import re


def novelty_score(content_hash: str, seen_hashes: set[str], claim_overlap: float = 0.0) -> float:
    """0=repost, 1=novel."""
    if content_hash in seen_hashes:
        # may be update if claim delta high, but base is low
        return 0.15 + 0.3 * max(0.0, 1 - claim_overlap)
    # unseen hash is novel; if high claim overlap still somewhat novel due to new evidence
    return 0.85 + 0.10 * (1 - claim_overlap)


def is_duplicate(content_hash: str, seen_hashes: set[str]) -> bool:
    return content_hash in seen_hashes


def classify_novelty(content_hash: str, seen_hashes: set[str], text_changed: bool = False) -> str:
    if content_hash not in seen_hashes:
        return "new"
    if text_changed:
        return "updated"
    return "repost"
