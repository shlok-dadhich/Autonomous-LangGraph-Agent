"""Intelligence subgraph — normalize → identity → cluster → events/entities/claims/quality → evidence."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.graph.nodes.claims import extract_claims_node
from app.graph.nodes.cluster import cluster_documents_node
from app.graph.nodes.entities import extract_entities_node
from app.graph.nodes.events import detect_events_node
from app.graph.nodes.identity import resolve_identity_node
from app.graph.nodes.normalize import normalize_documents_node
from app.graph.nodes.source_quality import score_sources_node
from app.graph.state import GraphState


def build_intelligence_subgraph():
    graph = StateGraph(GraphState)
    graph.add_node("normalize", normalize_documents_node)
    graph.add_node("identity", resolve_identity_node)
    graph.add_node("cluster", cluster_documents_node)
    graph.add_node("events", detect_events_node)
    graph.add_node("entities", extract_entities_node)
    graph.add_node("source_quality", score_sources_node)
    graph.add_node("claims", extract_claims_node)

    graph.add_edge(START, "normalize")
    graph.add_edge("normalize", "identity")
    graph.add_edge("identity", "cluster")
    graph.add_edge("cluster", "events")
    graph.add_edge("events", "entities")
    graph.add_edge("entities", "source_quality")
    graph.add_edge("source_quality", "claims")
    graph.add_edge("claims", END)

    return graph.compile()
