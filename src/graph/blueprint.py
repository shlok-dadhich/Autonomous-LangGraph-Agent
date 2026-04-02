"""
LangGraph blueprint for the newsletter agent research workflow.

Defines the graph architecture with parallel data collection nodes,
synchronization, and state management using Annotated reducers.
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
from loguru import logger

from src.graph.state import GraphState
from src.graph.nodes import (
    research_arxiv_node,
    research_web_node,
    research_hf_node,
    research_rss_node,
    merge_node,
    deduplicate_node,
    filter_node,
    fallback_search_node,
    writer_node,
    delivery_node,
)
from src.graph.edges import check_content_threshold
from src.core.database import create_sqlite_connection, ensure_data_dir


CHECKPOINT_DB_PATH = ensure_data_dir() / "checkpoints.db"
_checkpoint_conn = create_sqlite_connection(CHECKPOINT_DB_PATH)
checkpointer = SqliteSaver(_checkpoint_conn)


def build_fanout_blueprint():
    """
    Build the minimal Phase 5 fan-out blueprint.

    Wiring:
    - START -> research_arxiv_node
    - START -> research_web_node
    - START -> research_hf_node
    - START -> research_rss_node
    - START -> research_rss_node
    - research_arxiv_node -> merge_node
    - research_web_node -> merge_node
    - research_hf_node -> merge_node
    - research_rss_node -> merge_node
    - research_rss_node -> merge_node

    Returns:
        Compiled StateGraph with SqliteSaver checkpointer.
    """

    logger.info("Building Phase 5 fan-out blueprint...")

    graph = StateGraph(GraphState)

    # Register only the orchestration entry and sync nodes for fan-out/fan-in.
    graph.add_node("research_arxiv_node", research_arxiv_node)
    graph.add_node("research_web_node", research_web_node)
    graph.add_node("research_hf_node", research_hf_node)
    graph.add_node("research_rss_node", research_rss_node)
    graph.add_node("merge_node", merge_node)

    # Fan-out from START.
    graph.add_edge(START, "research_arxiv_node")
    graph.add_edge(START, "research_web_node")
    graph.add_edge(START, "research_hf_node")
    graph.add_edge(START, "research_rss_node")

    # Fan-in to merge synchronization node.
    graph.add_edge("research_arxiv_node", "merge_node")
    graph.add_edge("research_web_node", "merge_node")
    graph.add_edge("research_hf_node", "merge_node")
    graph.add_edge("research_rss_node", "merge_node")

    compiled_graph = graph.compile(checkpointer=checkpointer)
    logger.success("Phase 5 fan-out blueprint compiled successfully")
    return compiled_graph


def build_research_graph():
    """
    Build the LangGraph state machine for the research phase.
    
    Architecture:
    - START branches to research_arxiv_node, research_web_node, research_hf_node, and research_rss_node in parallel
    - All four nodes feed into merge_node for synchronization
    - merge_node validates counts and flows to deduplicate_node
    - deduplicate_node removes already-sent URLs and flows to filter_node
    - filter_node applies semantic ranking threshold
    - if filtered count < 3 -> fallback_search_node, else -> writer_node
    - fallback_search_node enriches buffer content and flows to writer_node
    - writer_node generates draft content and flows to delivery_node
    - delivery_node sends email and commits sent URLs, then flows to END
    
    Three Intelligence Layers:
    - Arxiv: Academic pre-print layer (formal research)
    - Tavily/HN: Market/News layer (business & startups)
    - Hugging Face: Implementation layer (what developers are actually using)
    
    Returns:
        Compiled StateGraph ready for execution.
    """
    
    logger.info("Building research graph...")
    
    # Create state graph with GraphState schema
    graph = StateGraph(GraphState)
    
    # ========================================================================
    # ADD NODES
    # ========================================================================
    
    graph.add_node("research_arxiv_node", research_arxiv_node)
    graph.add_node("research_web_node", research_web_node)
    graph.add_node("research_hf_node", research_hf_node)
    graph.add_node("merge_node", merge_node)
    graph.add_node("deduplicate_node", deduplicate_node)
    graph.add_node("filter_node", filter_node)
    graph.add_node("fallback_search_node", fallback_search_node)
    graph.add_node("writer_node", writer_node)
    graph.add_node("delivery_node", delivery_node)
    
    logger.debug(
        "Nodes added: research_arxiv_node, research_web_node, research_hf_node, research_rss_node, merge_node, deduplicate_node, filter_node, fallback_search_node, writer_node, delivery_node"
    )
    
    # ========================================================================
    # ADD EDGES
    # ========================================================================
    
    # Fan-out from START to parallel nodes
    graph.add_edge(START, "research_arxiv_node")
    graph.add_edge(START, "research_web_node")
    graph.add_edge(START, "research_hf_node")
    
    logger.debug("Edges added: START -> research_arxiv_node, START -> research_web_node, START -> research_hf_node, START -> research_rss_node")
    
    # All parallel nodes converge into merge_node
    graph.add_edge("research_arxiv_node", "merge_node")
    graph.add_edge("research_web_node", "merge_node")
    graph.add_edge("research_hf_node", "merge_node")
    
    logger.debug("Edges added: research_arxiv_node -> merge_node, research_web_node -> merge_node, research_hf_node -> merge_node, research_rss_node -> merge_node")
    
    # merge_node flows to deduplication
    graph.add_edge("merge_node", "deduplicate_node")
    
    logger.debug("Edge added: merge_node -> deduplicate_node")

    # deduplicate_node flows to semantic filter
    graph.add_edge("deduplicate_node", "filter_node")

    logger.debug("Edge added: deduplicate_node -> filter_node")

    # Minimum Content Guard: conditionally route after filtering
    graph.add_conditional_edges(
        "filter_node",
        check_content_threshold,
        {
            "fallback_search_node": "fallback_search_node",
            "writer_node": "writer_node",
        },
    )

    logger.debug(
        "Conditional edges added: filter_node -> fallback_search_node | writer_node"
    )

    # Fallback enriches content, then proceed to writer
    graph.add_edge("fallback_search_node", "writer_node")

    logger.debug("Edge added: fallback_search_node -> writer_node")

    # writer_node flows to delivery
    graph.add_edge("writer_node", "delivery_node")

    logger.debug("Edge added: writer_node -> delivery_node")

    # delivery_node flows to END
    graph.add_edge("delivery_node", END)

    logger.debug("Edge added: delivery_node -> END")
    
    # ========================================================================
    # COMPILE
    # ========================================================================
    
    compiled_graph = graph.compile(checkpointer=checkpointer)
    logger.success("Research graph compiled successfully")
    
    return compiled_graph


# Instantiate the compiled graph at module level for easy imports
research_graph = build_research_graph()
