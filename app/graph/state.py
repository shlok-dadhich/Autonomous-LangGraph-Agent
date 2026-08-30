"""Typed graph state — extends legacy GraphState with intelligence fields."""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict


class GraphState(TypedDict, total=False):
    # legacy
    interest_profile: dict
    profile: dict
    mode: str
    trusted_domains: list[str]
    raw_articles: Annotated[list, operator.add]
    unique_articles: list
    filtered_articles: list
    email_draft_content: list
    email_html_content: str | None
    sent_article_ids: list[str]
    thread_id: str | None
    logs: Annotated[list, operator.add]
    error: str | None

    # intelligence (Phase 2+)
    documents: list  # normalized Document dicts with canonical_url, content_hash, etc.
    clusters: list  # StoryClusterResult dicts
    events: list  # EventResult dicts
    entities: list  # EntityMention dicts
    claims: list  # Claim dicts
    evidence: list  # EvidenceBundle dicts
    source_scores: list  # SourceScore dicts
    metrics: dict  # per-stage metrics
