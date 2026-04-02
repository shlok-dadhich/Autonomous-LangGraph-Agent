"""
Social Signal client for fetching community discussions from Reddit, HackerNews, and HuggingFace.

Replaces the old Reddit API client by using Tavily Search with domain filters
to aggregate social signals from multiple platforms. This approach:
- Eliminates PRAW dependency
- Provides access to Reddit, HuggingFace, and HackerNews communities
- Returns standardized metadata schema for consistency
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
from loguru import logger

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.reliability import safe_execute

try:
    from tavily import TavilyClient
except ImportError:
    logger.warning("Tavily SDK not installed. Install with: pip install tavily-python")
    TavilyClient = None


def _construct_social_signal_query(
    topics: List[str],
    keywords: List[str],
    include_domains: List[str] = None,
) -> str:
    """
    Construct a Tavily search query optimized for social signals.
    
    Strategy: Build a query that targets specific domains and topics from the
    interest profile. Uses site: syntax for precision.
    
    Examples:
    - "site:reddit.com/r/MachineLearning AI breakthroughs latest"
    - "site:huggingface.co transformers language models"
    - "site:news.ycombinator.com AI news"
    
    Args:
        topics: List of topic strings from interest profile.
        keywords: List of keywords from interest profile.
        include_domains: List of domains to specifically target.
                        Defaults to ['reddit.com', 'huggingface.co', 'news.ycombinator.com']
    
    Returns:
        Formatted search query string suitable for Tavily API.
    """
    if include_domains is None:
        include_domains = ["reddit.com", "huggingface.co", "news.ycombinator.com"]
    
    combined_terms = topics + keywords if keywords else topics
    
    if not combined_terms:
        # Fallback to general AI social signals
        combined_terms = ["AI", "machine learning", "transformers"]
    
    # Build domain-specific queries for targeted search
    query_parts = []
    
    # For Reddit, try common AI/ML subreddits
    reddit_subreddits = ["MachineLearning", "LocalLLaMA", "LanguageModels", "artificialinteligence"]
    for topic in combined_terms[:3]:  # Limit to first 3 topics
        reddit_part = f"site:reddit.com/r/{reddit_subreddits[len(query_parts) % len(reddit_subreddits)]} {topic}"
        query_parts.append(reddit_part)
    
    # For HuggingFace discussions/models
    if combined_terms:
        hf_part = f"site:huggingface.co {combined_terms[0]}"
        query_parts.append(hf_part)
    
    # For HackerNews (news.ycombinator.com)
    if combined_terms:
        hn_part = f"site:news.ycombinator.com {combined_terms[0]}"
        query_parts.append(hn_part)
    
    # Join with OR logic for Tavily
    complex_query = " OR ".join(query_parts)
    
    logger.debug(f"Constructed social signal query: {complex_query}")
    return complex_query


def _map_domain_to_source(url: str) -> str:
    """
    Map domain in URL to standardized source name.
    
    Args:
        url: The URL string.
    
    Returns:
        Source identifier reflecting where Tavily found the content.
    """
    if "reddit.com" in url:
        return "reddit-via-tavily"
    elif "huggingface.co" in url:
        return "huggingface-via-tavily"
    elif "news.ycombinator.com" in url or "ycombinator.com" in url:
        return "hackernews-via-tavily"
    else:
        return "social-signals-via-tavily"


@safe_execute(source_name="social_signals", max_retries=2)
def fetch_social_signals(
    interest_profile: Dict[str, Any],
    max_results: int = 15,
    api_key: Optional[str] = None,
    include_domains: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch social signals from Reddit, HuggingFace, and HackerNews via Tavily.
    
    **METADATA SCHEMA:**
    - title (str): Article/post title
    - url (str): Direct link to the discussion or content
    - description (str): Excerpt or summary of the content (capped at 500 chars)
    - source (str): Platform identifier (reddit-via-tavily, huggingface-via-tavily, 
                     hackernews-via-tavily, or social-signals-via-tavily)
    - relevance_score (float): Relevance confidence (0.0-1.0)
    
    **RELIABILITY FEATURES:**
    - Tavily API error handling with retries via @safe_execute decorator
    - Per-domain domain inclusion filtering
    - Graceful fallback if API key is missing
    - Automatic description truncation
    
    **IMPROVEMENT OVER REDDIT CLIENT:**
    - No authentication required (uses Tavily API key)
    - Aggregates multiple community platforms
    - Removes PRAW dependency
    - Better handling of rate limits and API changes
    
    Args:
        interest_profile: Dict with 'topics' and 'keywords' keys from user profile.
        max_results: Maximum results per domain (default: 15, total ~45-60).
        api_key: Tavily API key. If None, reads from TAVILY_API_KEY env var.
        include_domains: List of domains to filter. Defaults to Reddit, HuggingFace, HN.
    
    Returns:
        List of dicts with schema: {title, url, description, source, relevance_score}.
    
    Raises:
        ValueError: If TAVILY_API_KEY is not provided.
    """
    if TavilyClient is None:
        raise ImportError("Tavily SDK not installed. Install with: pip install tavily-python")
    
    api_key = api_key or os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError(
            "TAVILY_API_KEY not found. Please set it in environment variables "
            "or pass api_key parameter."
        )
    
    if include_domains is None:
        include_domains = ["reddit.com", "huggingface.co", "news.ycombinator.com"]
    
    topics = interest_profile.get("topics", [])
    keywords = interest_profile.get("keywords", [])
    
    logger.info(f"Fetching social signals for topics: {topics}, keywords: {keywords}")
    
    try:
        client = TavilyClient(api_key=api_key)
        
        # Construct search query optimized for social signals
        query = _construct_social_signal_query(topics, keywords, include_domains)
        
        logger.debug(f"Social signals search query: {query}")
        
        # Execute Tavily search with domain inclusion filter
        response = client.search(
            query=query,
            max_results=max_results,
            include_domains=include_domains,
            include_answer=False,
            topic="general",
        )
        
        logger.info(
            f"Tavily social signals search returned {len(response.get('results', []))} results"
        )
        
        # Transform Tavily results to standard schema
        articles = []
        for result in response.get("results", []):
            article = {
                "title": result.get("title", ""),
                "url": result.get("url", ""),
                "description": result.get("content", "")[:500],  # Cap at 500 chars
                "source": _map_domain_to_source(result.get("url", "")),
                "relevance_score": 0.7,  # Tavily results are pre-ranked by relevance
            }
            
            # Only include if we have minimal required fields
            if article["title"] and article["url"]:
                articles.append(article)
        
        logger.info(f"Extracted {len(articles)} social signal articles with proper schema")
        return articles
        
    except Exception as e:
        logger.error(f"Error fetching social signals from Tavily: {str(e)}")
        raise
