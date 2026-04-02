"""
GraphState definition for the newsletter agent LangGraph workflow.

Defines the shared state structure that flows through all nodes in the graph,
with support for parallel node execution via Annotated list merging.
"""

from typing import Annotated, List, Optional, TypedDict
import operator


class GraphState(TypedDict):
    """
    Shared state for the newsletter agent graph.
    
    Attributes:
        interest_profile: User's interests and preferences for filtering articles.
        raw_articles: Accumulated articles from all sources (parallel merge).
        logs: Accumulated log entries from all processing steps (parallel merge).
        unique_articles: Deduplicated articles ready for ranking.
        filtered_articles: Ranked and threshold-filtered articles.
        email_draft_content: Enriched article summaries for final email draft.
        email_html_content: Rendered HTML newsletter ready for delivery.
        sent_article_ids: URLs already sent in previous newsletters.
        thread_id: Run identifier for traceability and alert correlation.
        error: Error message if any node encounters an unrecoverable error.
    """

    interest_profile: dict
    raw_articles: Annotated[list, operator.add]
    unique_articles: list
    filtered_articles: list
    email_draft_content: list
    email_html_content: Optional[str]
    sent_article_ids: List[str]
    thread_id: Optional[str]
    logs: Annotated[list, operator.add]
    error: Optional[str]
