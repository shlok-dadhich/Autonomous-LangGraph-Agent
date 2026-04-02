"""Conditional edge routing helpers for the LangGraph workflow."""

from src.graph.state import GraphState


def check_content_threshold(state: GraphState) -> str:
	"""
	Route based on filtered article count.

	Returns:
		fallback_search_node: if filtered count is less than 3.
		writer_node: if filtered count is 3 or more.
	"""
	filtered_articles = state.get("filtered_articles", [])
	if len(filtered_articles) < 3:
		return "fallback_search_node"
	return "writer_node"


def check_article_count(state: GraphState) -> str:
	"""Backward-compatible alias for legacy graph imports."""
	return check_content_threshold(state)
