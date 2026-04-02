"""
Graph nodes for the newsletter agent research phase.

Each node executes a parallel data-collection task, wrapped with resilience logic.
Nodes return state updates compatible with the Annotated list reducers.
"""

import os
from collections import Counter
from datetime import datetime
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
from loguru import logger
from dotenv import load_dotenv

from src.graph.state import GraphState
from src.tools.arxiv_client import fetch_arxiv_papers
from src.tools.tavily_client import fetch_tavily_results
from src.tools.social_signal_client import fetch_social_signals
from src.tools.hn_client import fetch_hn_stories
from src.tools.hf_client import fetch_hf_daily_papers
from src.tools.rss_client import fetch_rss_sources
from src.utils.reliability import safe_execute
from src.core.database import DatabaseManager
from src.core.ranker import RelevanceRanker
from src.core.writer import NewsletterWriter
from src.services.email_service import EmailService
from src.services.template_service import TemplateService
from src.services.telegram_bot import AlertService

# Load environment variables
load_dotenv()


# ============================================================================
# WRAPPED TOOL FUNCTIONS WITH RESILIENCE
# ============================================================================


@safe_execute(source_name="arxiv", max_retries=2)
def _fetch_arxiv_safe(**kwargs) -> list:
    """Wrapped Arxiv fetch with retries and logging."""
    result = fetch_arxiv_papers(**kwargs)
    if isinstance(result, dict):
        return result.get("raw_articles", [])
    return result if isinstance(result, list) else []


@safe_execute(source_name="tavily", max_retries=2)
def _fetch_tavily_safe(interest_profile: Dict[str, Any], max_results: int = 10) -> list:
    """Wrapped Tavily fetch with retries and logging."""
    api_key = os.getenv("TAVILY_API_KEY")
    result = fetch_tavily_results(
        interest_profile=interest_profile,
        max_results=max_results,
        api_key=api_key,
    )
    if isinstance(result, dict):
        return result.get("raw_articles", [])
    return result if isinstance(result, list) else []


@safe_execute(source_name="tavily_fallback", max_retries=2)
def _fetch_tavily_buffer_safe() -> list:
    """Wrapped broad Tavily fallback search with retries and logging."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY not provided for fallback search")

    from tavily import TavilyClient

    client = TavilyClient(api_key=api_key)
    response = client.search(
        query="latest important AI breakthroughs",
        max_results=6,
        include_answer=False,
        topic="general",
    )

    return [
        {
            "title": result.get("title", ""),
            "url": result.get("url", ""),
            "description": result.get("content", "")[:500],
            "source": "tavily_fallback",
            "relevance_score": 0.45,
        }
        for result in response.get("results", [])
    ]


@safe_execute(source_name="social_signals", max_retries=2)
def _fetch_social_signals_safe(
    interest_profile: Dict[str, Any],
    max_results: int = 15,
) -> list:
    """Wrapped social signals fetch with retries and logging."""
    api_key = os.getenv("TAVILY_API_KEY")
    result = fetch_social_signals(
        interest_profile=interest_profile,
        max_results=max_results,
        api_key=api_key,
    )
    if isinstance(result, dict):
        return result.get("raw_articles", [])
    return result if isinstance(result, list) else []


@safe_execute(source_name="hackernews", max_retries=2)
def _fetch_hn_safe(
    interest_profile: Dict[str, Any],
    min_score: int = 50,
    max_items: int = 100,
) -> list:
    """Wrapped HN fetch with retries and logging."""
    result = fetch_hn_stories(
        interest_profile=interest_profile,
        min_score=min_score,
        max_items=max_items,
    )
    if isinstance(result, dict):
        return result.get("raw_articles", [])
    return result if isinstance(result, list) else []


@safe_execute(source_name="huggingface_daily", max_retries=2)
def _fetch_hf_daily_safe(limit: int = 5) -> list:
    """Wrapped HF Daily Papers fetch with retries and logging."""
    result = fetch_hf_daily_papers(limit=limit)
    if isinstance(result, dict):
        return result.get("raw_articles", [])
    return result if isinstance(result, list) else []


@safe_execute(source_name="rss_sources", max_retries=2)
def _fetch_rss_safe(feed_specs: list[Dict[str, Any]] | None = None) -> list:
    """Wrapped RSS / newsroom HTML source fetch with retries and logging."""
    result = fetch_rss_sources(feed_specs=feed_specs)
    if isinstance(result, dict):
        return result.get("raw_articles", [])
    return result if isinstance(result, list) else []


def _has_verified_url(article: Dict[str, Any]) -> bool:
    """Return True when article has an http(s) URL with a host."""
    url = str(article.get("url", "")).strip()
    if not url:
        return False

    parsed = urlparse(url)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _is_diversity_source(source_name: str) -> bool:
    """Identify sources that should be guaranteed representation when available."""
    normalized = source_name.lower()
    return normalized.startswith("huggingface") or normalized in {
        "anthropic-newsroom",
        "reddit",
        "twitter",
        "x",
        "linkedin",
        "social_signals",
        "tavily",
        "tavily_fallback",
    }


def _prune_by_similarity_with_source_preference(
    scored_articles: list[Dict[str, Any]],
    similarity_threshold: float = 0.9,
) -> list[Dict[str, Any]]:
    """Drop near-duplicates while preferring diversity sources when they conflict."""
    if not scored_articles:
        return []

    candidate_articles = [article for article in scored_articles if isinstance(article, dict)]
    if not candidate_articles:
        return []

    candidate_articles.sort(key=lambda article: float(article.get("relevance_score", 0.0)), reverse=True)

    try:
        import torch
        from sentence_transformers import SentenceTransformer
        from sentence_transformers import util

        ranker = RelevanceRanker(model_name="all-MiniLM-L6-v2")
        model = SentenceTransformer(ranker.model_name, device="cpu")
        model = model.to("cpu")
        texts = [ranker._article_text(article) for article in candidate_articles]

        with torch.no_grad():
            embeddings = model.encode(
                texts,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )

        kept_indices: list[int] = []

        for index, article in enumerate(candidate_articles):
            if not kept_indices:
                kept_indices.append(index)
                continue

            current_embedding = embeddings[index : index + 1]
            kept_embeddings = embeddings[kept_indices]
            similarities = util.cos_sim(current_embedding, kept_embeddings)[0]
            max_similarity = float(similarities.max().item())

            if max_similarity < similarity_threshold:
                kept_indices.append(index)
                continue

            if _is_diversity_source(str(article.get("source", ""))):
                best_match_position = int(similarities.argmax().item())
                best_match_index = kept_indices[best_match_position]
                best_match_article = candidate_articles[best_match_index]

                if not _is_diversity_source(str(best_match_article.get("source", ""))):
                    kept_indices[best_match_position] = index

        selected_articles = [candidate_articles[index] for index in kept_indices]
        selected_articles.sort(key=lambda article: float(article.get("relevance_score", 0.0)), reverse=True)
        return selected_articles
    except Exception:
        # Fall back to score-based dedupe if transformer loading fails.
        ranker = RelevanceRanker(model_name="all-MiniLM-L6-v2")
        try:
            return ranker.prune_similar_articles(candidate_articles, similarity_threshold=similarity_threshold)
        except Exception:
            return candidate_articles


def _select_articles_for_newsletter(
    scored_articles: list[Dict[str, Any]],
    interest_profile: Dict[str, Any],
    threshold: float,
    diversity_threshold: float = 0.40,
    max_filtered_articles: int = 6,
) -> list[Dict[str, Any]]:
    verified_articles = [article for article in scored_articles if _has_verified_url(article)]
    if not verified_articles:
        return []

    baseline_articles = [
        article
        for article in verified_articles
        if float(article.get("relevance_score", 0.0)) >= threshold
    ]

    baseline_articles.sort(key=lambda article: float(article.get("relevance_score", 0.0)), reverse=True)

    diversity_candidates = [
        article
        for article in verified_articles
        if _is_diversity_source(str(article.get("source", "")))
        and float(article.get("relevance_score", 0.0)) >= diversity_threshold
    ]
    diversity_candidates.sort(key=lambda article: float(article.get("relevance_score", 0.0)), reverse=True)

    selected_articles = baseline_articles[:max_filtered_articles]
    if diversity_candidates and not any(
        _is_diversity_source(str(article.get("source", ""))) for article in selected_articles
    ):
        diversity_candidate = diversity_candidates[0]
        selected_urls = {str(article.get("url", "")).strip() for article in selected_articles}
        diversity_url = str(diversity_candidate.get("url", "")).strip()

        if diversity_url and diversity_url not in selected_urls:
            if len(selected_articles) >= max_filtered_articles and selected_articles:
                selected_articles = selected_articles[:-1]
            selected_articles.append(diversity_candidate)

    selected_articles = _prune_by_similarity_with_source_preference(selected_articles, similarity_threshold=0.9)

    if len(selected_articles) > max_filtered_articles:
        selected_articles = selected_articles[:max_filtered_articles]

    selected_articles.sort(key=lambda article: float(article.get("relevance_score", 0.0)), reverse=True)
    return selected_articles


# ============================================================================
# NODE FUNCTIONS
# ============================================================================


def research_arxiv_node(state: GraphState) -> Dict[str, Any]:
    """
    Fetch recent research papers from Arxiv.
    
    Queries cs.AI and cs.LG categories for papers from the last 7 days.
    Updates the state with raw_articles and logs from the operation.
    
    Args:
        state: Current GraphState
    
    Returns:
        Dict with updates to raw_articles and logs
    """
    logger.info("=" * 60)
    logger.info("NODE: research_arxiv_node")
    logger.info("=" * 60)

    start_ts = datetime.now().isoformat(timespec="milliseconds")
    start_msg = f"[arxiv] Started Arxiv Search at {start_ts}"
    logger.info(start_msg)
    
    interest_profile = state.get("interest_profile", {})
    arxiv_cfg = interest_profile.get("sources", {}).get("arxiv", {})

    if arxiv_cfg.get("enabled", True) is False:
        msg = "[arxiv] Source disabled in interest profile."
        logger.info(msg)
        return {
            "raw_articles": [],
            "logs": [
                {"level": "info", "message": start_msg, "timestamp": start_ts},
                {"level": "info", "message": msg, "timestamp": datetime.now().isoformat(timespec="milliseconds")},
            ],
        }

    categories = arxiv_cfg.get("categories", ["cs.AI", "cs.LG"])
    days_back = arxiv_cfg.get("days_back", 7)
    max_results = arxiv_cfg.get("max_results", 20)

    logger.debug(
        f"Arxiv config -> categories={categories}, days_back={days_back}, "
        f"max_results={max_results}"
    )

    result = _fetch_arxiv_safe(
        categories=categories,
        days_back=days_back,
        max_results=max_results,
    )

    paper_count = len(result.get("raw_articles", []))
    summary_log = {
        "level": "info",
        "message": f"[arxiv] Completed research_arxiv_node with {paper_count} papers.",
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
    }
    logs = [{"level": "info", "message": start_msg, "timestamp": start_ts}]
    logs.extend(list(result.get("logs", [])))
    logs.append(summary_log)
    
    logger.info(f"Arxiv node complete: {paper_count} papers")

    return {
        "raw_articles": list(result.get("raw_articles", [])),
        "logs": logs,
    }


def research_web_node(state: GraphState) -> Dict[str, Any]:
    """
    Fetch content from web sources: Tavily, Reddit, and Hacker News.
    
    Executes three parallel-capable sub-fetches to collect diverse web content
    filtered by interest profile. Aggregates all results into state.
    
    Args:
        state: Current GraphState
    
    Returns:
        Dict with updates to raw_articles and logs
    """
    logger.info("=" * 60)
    logger.info("NODE: research_web_node")
    logger.info("=" * 60)

    start_ts = datetime.now().isoformat(timespec="milliseconds")
    start_msg = f"[web] Started Web Search at {start_ts}"
    logger.info(start_msg)
    
    interest_profile = state.get("interest_profile", {})
    keywords = interest_profile.get("keywords", [])
    sources_cfg = interest_profile.get("sources", {})
    logger.debug(f"Interest profile keywords: {keywords}")

    tavily_cfg = sources_cfg.get("tavily", {})
    social_signals_cfg = sources_cfg.get("social_signals", sources_cfg.get("reddit", {}))
    hn_cfg = sources_cfg.get("hackernews", {})

    all_articles = []
    all_logs = []
    succeeded_sources = []

    def _run_tavily() -> tuple[str, Dict[str, Any]]:
        if tavily_cfg.get("enabled", True) is False:
            return "tavily", {"raw_articles": [], "logs": [{"level": "info", "message": "[tavily] Source disabled in interest profile."}]}
        guided_profile = {
            "topics": interest_profile.get("topics", []),
            "keywords": keywords,
        }
        return "tavily", _fetch_tavily_safe(
            interest_profile=guided_profile,
            max_results=tavily_cfg.get("max_results", 10),
        )

    def _run_social_signals() -> tuple[str, Dict[str, Any]]:
        if social_signals_cfg.get("enabled", True) is False:
            return "social_signals", {"raw_articles": [], "logs": [{"level": "info", "message": "[social_signals] Source disabled in interest profile."}]}
        guided_profile = {
            "topics": interest_profile.get("topics", []),
            "keywords": keywords,
        }
        return "social_signals", _fetch_social_signals_safe(
            interest_profile=guided_profile,
            max_results=social_signals_cfg.get("max_results", 15),
        )

    def _run_hn() -> tuple[str, Dict[str, Any]]:
        if hn_cfg.get("enabled", True) is False:
            return "hackernews", {"raw_articles": [], "logs": [{"level": "info", "message": "[hackernews] Source disabled in interest profile."}]}
        guided_profile = {
            "topics": interest_profile.get("topics", []),
            "keywords": keywords,
        }
        return "hackernews", _fetch_hn_safe(
            interest_profile=guided_profile,
            min_score=hn_cfg.get("min_score", 50),
            max_items=hn_cfg.get("max_results", 20),
        )

    source_runners = [_run_tavily, _run_social_signals, _run_hn]

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(runner) for runner in source_runners]
        for future in as_completed(futures):
            source_name, result = future.result()
            source_articles = list(result.get("raw_articles", []))
            source_logs = list(result.get("logs", []))

            all_articles.extend(source_articles)
            all_logs.extend(source_logs)

            if source_articles:
                succeeded_sources.append(source_name)

    summary = (
        f"[web] Completed web fan-out. Successful sources: "
        f"{', '.join(succeeded_sources) if succeeded_sources else 'none'}. "
        f"Total web articles: {len(all_articles)}."
    )
    all_logs.insert(0, {"level": "info", "message": start_msg, "timestamp": start_ts})
    all_logs.append({"level": "info", "message": summary, "timestamp": datetime.now().isoformat(timespec="milliseconds")})

    logger.info(summary)

    return {
        "raw_articles": list(all_articles),
        "logs": list(all_logs),
    }


def research_hf_node(state: GraphState) -> Dict[str, Any]:
    """
    Fetch trending papers from Hugging Face Daily Papers feed.

    The "Implementation Layer" - captures what developers are actually building
    and discussing today. Direct community validation through engagement metrics.
    Public API, no authentication required.

    Args:
        state: Current GraphState

    Returns:
        Dict with updates to raw_articles and logs
    """
    logger.info("=" * 60)
    logger.info("NODE: research_hf_node")
    logger.info("=" * 60)

    start_ts = datetime.now().isoformat(timespec="milliseconds")
    start_msg = f"[huggingface] Started HF Daily Papers fetch at {start_ts}"
    logger.info(start_msg)

    interest_profile = state.get("interest_profile", {})
    hf_cfg = interest_profile.get("sources", {}).get("huggingface", {})

    if hf_cfg.get("enabled", True) is False:
        msg = "[huggingface] Source disabled in interest profile."
        logger.info(msg)
        return {
            "raw_articles": [],
            "logs": [
                {"level": "info", "message": start_msg, "timestamp": start_ts},
                {"level": "info", "message": msg, "timestamp": datetime.now().isoformat(timespec="milliseconds")},
            ],
        }

    limit = hf_cfg.get("limit", 5)
    
    logger.debug(f"HF config -> limit={limit}")

    result = _fetch_hf_daily_safe(limit=limit)
    hf_articles = list(result.get("raw_articles", []))

    # Keep only merge-compatible records with required keys and a verified URL.
    normalized_hf_articles = []
    for article in hf_articles:
        if not isinstance(article, dict):
            continue

        normalized = {
            "title": str(article.get("title", "")).strip(),
            "url": str(article.get("url", "")).strip(),
            "description": str(article.get("description", "")).strip(),
            "source": str(article.get("source", "huggingface-daily")).strip() or "huggingface-daily",
            "relevance_score": float(article.get("relevance_score", 0.8)),
        }

        if normalized["title"] and _has_verified_url(normalized):
            normalized_hf_articles.append(normalized)

    paper_count = len(normalized_hf_articles)
    summary_log = {
        "level": "info",
        "message": f"[huggingface] Completed research_hf_node with {paper_count} papers.",
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
    }
    logs = [{"level": "info", "message": start_msg, "timestamp": start_ts}]
    logs.extend(list(result.get("logs", [])))
    logs.append(summary_log)

    logger.info(f"HF node complete: {paper_count} papers")

    return {
        "raw_articles": normalized_hf_articles,
        "logs": logs,
    }


def research_rss_node(state: GraphState) -> Dict[str, Any]:
    """Fetch high-signal RSS and newsroom-style sources."""
    logger.info("=" * 60)
    logger.info("NODE: research_rss_node")
    logger.info("=" * 60)

    start_ts = datetime.now().isoformat(timespec="milliseconds")
    start_msg = f"[rss] Started RSS source fetch at {start_ts}"
    logger.info(start_msg)

    interest_profile = state.get("interest_profile", {})
    rss_cfg = interest_profile.get("sources", {}).get("rss", {})

    if rss_cfg.get("enabled", True) is False:
        msg = "[rss] Source disabled in interest profile."
        logger.info(msg)
        return {
            "raw_articles": [],
            "logs": [
                {"level": "info", "message": start_msg, "timestamp": start_ts},
                {"level": "info", "message": msg, "timestamp": datetime.now().isoformat(timespec="milliseconds")},
            ],
        }

    feed_specs = rss_cfg.get("feeds", []) or []
    if not feed_specs:
        feed_specs = [
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

    logger.debug(f"RSS feed count -> {len(feed_specs)}")
    result = _fetch_rss_safe(feed_specs=feed_specs)
    rss_articles = list(result.get("raw_articles", []))

    normalized_articles = []
    for article in rss_articles:
        if not isinstance(article, dict):
            continue
        normalized = {
            "title": str(article.get("title", "")).strip(),
            "url": str(article.get("url", "")).strip(),
            "description": str(article.get("description", "")).strip(),
            "source": str(article.get("source", "rss-feed")).strip() or "rss-feed",
            "relevance_score": float(article.get("relevance_score", 0.75)),
            "published_date": str(article.get("published_date", "")).strip(),
        }
        if normalized["title"] and _has_verified_url(normalized):
            normalized_articles.append(normalized)

    paper_count = len(normalized_articles)
    summary_log = {
        "level": "info",
        "message": f"[rss] Completed research_rss_node with {paper_count} items.",
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
    }
    logs = [{"level": "info", "message": start_msg, "timestamp": start_ts}]
    logs.extend(list(result.get("logs", [])))
    logs.append(summary_log)

    logger.info(f"RSS node complete: {paper_count} items")

    return {
        "raw_articles": normalized_articles,
        "logs": logs,
    }


def merge_node(state: GraphState) -> Dict[str, Any]:
    """
    Merge and validate articles from parallel research nodes.
    
    Logs the total count of harvested articles and checks if the collection
    is empty. Acts as a synchronization point for parallel nodes.
    
    Args:
        state: Current GraphState with merged articles from all sources
    
    Returns:
        Dict with optional error update if no articles found
    """
    logger.info("=" * 60)
    logger.info("NODE: merge_node")
    logger.info("=" * 60)
    
    raw_articles = state.get("raw_articles", [])
    source_counts = Counter(
        article.get("source", "unknown")
        for article in raw_articles
        if isinstance(article, dict)
    )

    total_count = len(raw_articles)
    breakdown = (
        ", ".join(
            f"{source}={count}"
            for source, count in sorted(source_counts.items())
        )
        if source_counts
        else "none"
    )

    banner = (
        "\n"
        + "#" * 74
        + "\n"
        + "# HARVEST PHASE COMPLETE"
        + "\n"
        + f"# Total Items: {total_count}"
        + "\n"
        + f"# Source Breakdown: {breakdown}"
        + "\n"
        + "#" * 74
    )
    logger.info(banner)

    merge_log = {
        "level": "info",
        "message": (
            f"[merge] Harvest Phase Complete | total_items={total_count} | "
            f"breakdown={breakdown}"
        ),
    }

    if total_count == 0:
        error_msg = "Zero articles harvested"
        logger.error(f"[merge] {error_msg}")
        return {
            "error": error_msg,
            "logs": [
                merge_log,
                {"level": "error", "message": f"[merge] {error_msg}"},
            ],
        }

    logger.success(f"Merge complete with {total_count} articles ready for ranking")
    return {"logs": [merge_log]}


def deduplicate_node(state: GraphState) -> Dict[str, Any]:
    """
    Remove already-sent articles using persistent URL history.

    Loads sent URLs from SQLite, filters raw_articles, and stores survivors
    in unique_articles for downstream ranking.
    """
    logger.info("=" * 60)
    logger.info("NODE: deduplicate_node")
    logger.info("=" * 60)

    db = DatabaseManager()
    sent_article_ids_set = set(db.get_sent_ids())

    raw_articles = state.get("raw_articles", [])
    unique_articles = []
    skipped_count = 0

    for article in raw_articles:
        url = article.get("url", "")
        if not url or url in sent_article_ids_set:
            skipped_count += 1
            continue
        unique_articles.append(article)

    log_entry = {
        "level": "info",
        "message": (
            "[deduplicate] "
            f"Loaded {len(sent_article_ids_set)} sent URLs; "
            f"filtered {skipped_count} duplicates; "
            f"kept {len(unique_articles)} unique articles."
        ),
    }

    logger.info(log_entry["message"])

    return {
        "sent_article_ids": sorted(sent_article_ids_set),
        "unique_articles": unique_articles,
        "logs": [log_entry],
    }


def _profile_to_text(interest_profile: Dict[str, Any]) -> str:
    """Convert profile dict into a single text query for embeddings."""
    topics = interest_profile.get("topics", [])
    keywords = interest_profile.get("keywords", [])
    return ". ".join([str(item) for item in topics + keywords if item]).strip()


def filter_node(state: GraphState) -> Dict[str, Any]:
    """
    Rank unique articles semantically and keep only relevant results.

    - Scores with sentence-transformers cosine similarity.
    - Drops articles with score < 0.45.
    - Sorts remaining items by descending relevance_score.
    """
    logger.info("=" * 60)
    logger.info("NODE: filter_node")
    logger.info("=" * 60)

    unique_articles = state.get("unique_articles", [])
    interest_profile = state.get("interest_profile", {})
    profile_text = _profile_to_text(interest_profile)

    if not unique_articles:
        msg = "[filter] No unique articles to rank."
        logger.warning(msg)
        return {
            "filtered_articles": [],
            "logs": [{"level": "warning", "message": msg}],
        }

    if not profile_text:
        msg = "[filter] Empty interest profile text; skipping semantic ranking."
        logger.warning(msg)
        return {
            "filtered_articles": [],
            "logs": [{"level": "warning", "message": msg}],
        }

    ranker = RelevanceRanker(model_name="all-MiniLM-L6-v2")

    try:
        scored_articles = ranker.score_articles(
            interest_profile_text=profile_text,
            unique_articles=unique_articles,
        )

        threshold = 0.45
        max_filtered_articles = int(interest_profile.get("max_filtered_articles", 6))
        filtered_articles = _select_articles_for_newsletter(
            scored_articles=scored_articles,
            interest_profile=interest_profile,
            threshold=threshold,
            diversity_threshold=0.40,
            max_filtered_articles=max_filtered_articles,
        )

        msg = (
            "[filter] "
            f"Ranked {len(scored_articles)} articles; "
            f"kept {len(filtered_articles)} with score >= {threshold} "
            f"(diversity floor=0.4, cap={max_filtered_articles}) "
            "and verified URLs only."
        )
        logger.info(msg)

        return {
            "filtered_articles": filtered_articles,
            "logs": [{"level": "info", "message": msg}],
        }
    except Exception as exc:
        error_msg = f"[filter] Ranking failed: {type(exc).__name__}: {str(exc)}"
        logger.error(error_msg)
        return {
            "filtered_articles": [],
            "logs": [{"level": "error", "message": error_msg}],
            "error": error_msg,
        }


def fallback_search_node(state: GraphState) -> Dict[str, Any]:
    """
    Fetch broader buffer content when semantic filter is too strict.

    Uses a less restrictive Tavily query to ensure minimum newsletter content.
    """
    logger.info("=" * 60)
    logger.info("NODE: fallback_search_node")
    logger.info("=" * 60)

    current_filtered = list(state.get("filtered_articles", []))
    needed = max(0, 3 - len(current_filtered))

    if needed == 0:
        msg = "[fallback] Minimum content already satisfied; skipping fallback search."
        logger.info(msg)
        return {"logs": [{"level": "info", "message": msg}]}

    unique_articles = state.get("unique_articles", [])
    interest_profile = state.get("interest_profile", {})

    if unique_articles:
        try:
            ranker = RelevanceRanker(model_name="all-MiniLM-L6-v2")
            profile_text = _profile_to_text(interest_profile)
            rescored_articles = ranker.score_articles(
                interest_profile_text=profile_text,
                unique_articles=unique_articles,
            )
            relaxed_articles = _select_articles_for_newsletter(
                scored_articles=rescored_articles,
                interest_profile=interest_profile,
                threshold=0.35,
                diversity_threshold=0.35,
                max_filtered_articles=6,
            )

            current_urls = {article.get("url", "") for article in current_filtered}
            for article in relaxed_articles:
                article_url = article.get("url", "")
                if article_url and article_url not in current_urls:
                    current_filtered.append(article)
                    current_urls.add(article_url)
                if len(current_filtered) >= 3:
                    break

            needed = max(0, 3 - len(current_filtered))
        except Exception as exc:
            logger.warning(f"[fallback] Relaxed re-rank failed: {type(exc).__name__}: {str(exc)}")

    if needed == 0:
        msg = "[fallback] Relaxed semantic re-rank restored minimum content; skipping broader search."
        logger.info(msg)
        return {
            "filtered_articles": current_filtered,
            "logs": [{"level": "info", "message": msg}],
        }

    result = _fetch_tavily_buffer_safe()
    buffer_articles = result.get("raw_articles", [])
    fallback_logs = result.get("logs", [])

    if buffer_articles:
        current_urls = {article.get("url", "") for article in current_filtered}
        appended = 0
        for article in buffer_articles:
            article_url = article.get("url", "")
            if article_url and article_url not in current_urls:
                current_filtered.append(article)
                current_urls.add(article_url)
                appended += 1
            if len(current_filtered) >= 3:
                break

        msg = (
            "[fallback] "
            f"Needed {needed} items, added {appended} fallback articles, "
            f"new filtered count: {len(current_filtered)}."
        )
    else:
        msg = "[fallback] No fallback articles returned from Tavily broad search."

    logger.info(msg)
    fallback_logs.append({"level": "info", "message": msg})

    return {
        "filtered_articles": current_filtered,
        "logs": fallback_logs,
    }


def writer_node(state: GraphState) -> Dict[str, Any]:
    """
    Generate persona-driven intelligence and render the newsletter HTML.

    Produces per-article 3-line summaries, a tailored personalized insight,
    and the final HTML email artifact.
    """
    logger.info("=" * 60)
    logger.info("NODE: writer_node")
    logger.info("=" * 60)

    filtered_articles = state.get("filtered_articles", [])
    interest_profile = state.get("interest_profile", {})
    writer_cfg = interest_profile.get("sources", {}).get("writer", {})
    batch_size = int(writer_cfg.get("batch_size", 3))

    if not filtered_articles:
        msg = "[writer] No filtered articles available for drafting."
        logger.warning(msg)
        return {
            "email_draft_content": [],
            "email_html_content": "",
            "logs": [{"level": "warning", "message": msg}],
        }

    writer = NewsletterWriter()
    email_draft_content = writer.generate_analysis(
        interest_profile=interest_profile,
        article_list=filtered_articles,
        batch_size=batch_size,
    )

    template_service = TemplateService()
    profile_topics = [str(item).strip() for item in interest_profile.get("topics", []) if item]
    profile_keywords = [str(item).strip() for item in interest_profile.get("keywords", []) if item]
    top_topic = profile_topics[0] if profile_topics else (profile_keywords[0] if profile_keywords else "RAG")
    build_label = datetime.now().strftime("%Y-%m-%d")
    email_html_content = template_service.render_newsletter(
        enriched_articles=email_draft_content,
        current_date=datetime.now().strftime("%b %d, %Y"),
        newsletter_title="AI Weekly Intelligence",
        top_topic=top_topic,
        build_label=build_label,
        managed_by="GitHub Copilot",
        feedback_url="mailto:newsletter-feedback@example.com?subject=AI%20Weekly%20Intelligence%20Feedback",
    )

    msg = (
        f"[writer] Generated enriched draft content for "
        f"{len(email_draft_content)} article(s) using batch_size={batch_size}."
    )
    logger.info(msg)
    logger.info(f"[writer] Rendered newsletter HTML ({len(email_html_content)} characters).")

    return {
        "email_draft_content": email_draft_content,
        "email_html_content": email_html_content,
        "logs": [{"level": "info", "message": msg}],
    }


def delivery_node(state: GraphState) -> Dict[str, Any]:
    """
    Dispatch final newsletter and commit sent URLs after successful delivery.

    URLs are persisted to history.db only when SMTP send returns success,
    preventing false "sent" memory when delivery fails.
    """
    logger.info("=" * 60)
    logger.info("NODE: delivery_node")
    logger.info("=" * 60)

    draft_articles = state.get("email_draft_content", [])
    email_html_content = state.get("email_html_content", "")
    filtered_articles = state.get("filtered_articles", [])
    thread_id = state.get("thread_id", "unknown")
    recipient_email = state.get("interest_profile", {}).get("delivery", {}).get("recipient_email")

    if not draft_articles:
        msg = "[delivery] No draft content available for email dispatch."
        logger.warning(msg)
        return {
            "logs": [{"level": "warning", "message": msg}],
            "error": msg,
        }

    html_content = email_html_content or TemplateService().render_newsletter(
        enriched_articles=draft_articles,
        current_date=datetime.now().strftime("%b %d, %Y"),
    )

    subject = f"Your AI Weekly — {datetime.now().strftime('%Y-%m-%d')}"
    email_service = EmailService(recipient_email=recipient_email)

    try:
        sent_ok = email_service.send_newsletter(
            html_content=html_content,
            subject=subject,
            graph_state=state,
            thread_id=thread_id,
        )
        if not sent_ok:
            raise RuntimeError("SMTP delivery returned False")

        sent_urls = [article.get("url", "") for article in filtered_articles if article.get("url")]
        db = DatabaseManager()
        db.add_sent_ids(sent_urls)

        msg = (
            f"Newsletter dispatched to {recipient_email or email_service.recipient_email} "
            f"and {len(sent_urls)} URLs committed to history."
        )
        logger.success(msg)

        alert_service = AlertService()
        try:
            alert_service.send_success_notification(
                delivered_items=len(draft_articles),
                thread_id=thread_id,
            )
        except Exception as alert_exc:
            logger.warning(
                f"[delivery] Telegram success alert failed: {type(alert_exc).__name__}: {str(alert_exc)}"
            )

        return {
            "logs": [{"level": "success", "message": msg}],
        }

    except Exception as exc:
        error_msg = f"[delivery] Email dispatch failed: {type(exc).__name__}: {str(exc)}"
        logger.error(error_msg)

        alert_service = AlertService()
        try:
            alert_service.send_error(
                error_message=error_msg,
                thread_id=thread_id,
            )
        except Exception as alert_exc:
            logger.warning(
                f"[delivery] Telegram error alert failed: {type(alert_exc).__name__}: {str(alert_exc)}"
            )

        return {
            "logs": [{"level": "error", "message": error_msg}],
            "error": error_msg,
        }
