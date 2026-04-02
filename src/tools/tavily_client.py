"""
Tavily Web Search client for fetching context-aware web results.

Performs semantic searches on Tavily platform based on user interest profile,
with dynamic search depth fallback and robust error handling for phase 4
research instrumentation.
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


def _construct_complex_query(interest_profile: Dict[str, Any]) -> str:
    """
    Construct a complex search query from interest profile.
    
    Strategy: Combines topics and keywords into a sophisticated query that
    captures both breadth and specificity.
    
    Example query: "AI agents latest news, LLM breakthroughs, reasoning startups 2026"
    
    Args:
        interest_profile: Dict with 'topics' and 'keywords' keys.
    
    Returns:
        Formatted search query string.
    """
    topics = interest_profile.get("topics", [])
    keywords = interest_profile.get("keywords", [])
    
    if not topics and not keywords:
        return "artificial intelligence latest news 2026"
    
    query_parts = []
    
    # Add topics with varied keywords (news, breakthroughs, developments)
    if topics:
        for i, topic in enumerate(topics):
            if i % 3 == 0:
                query_parts.append(f"{topic} latest news")
            elif i % 3 == 1:
                query_parts.append(f"{topic} breakthroughs")
            else:
                query_parts.append(f"{topic} startups 2026")
    
    # Add keywords as direct OR terms
    if keywords:
        query_parts.extend(keywords)
    
    # Join into semicolon-separated query for semantic precision
    complex_query = "; ".join(query_parts)
    
    logger.debug(f"Constructed complex query: {complex_query}")
    return complex_query


@safe_execute(source_name="tavily", max_retries=2)
def fetch_tavily_results(
    interest_profile: Dict[str, Any],
    max_results: int = 10,
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch web search results from Tavily API based on interest profile.
    
    **METADATA SCHEMA:**
    - title (str): Page title from search result
    - url (str): Full direct link to the page
    - description (str): Content snippet or excerpt from page
    - source (str): Hardcoded as 'tavily'
    - published_date (str): ISO 8601 timestamp of fetch (approximation)
    
    **RELIABILITY FEATURES:**
    - Dynamic Search Depth: If first search returns < 3 results, automatically
      retries with search_depth='advanced' for deeper crawling.
    - Wrapped in @safe_execute decorator with retry logic.
    - Secure API key handling: Loads from TAVILY_API_KEY env var.
    
    Args:
        interest_profile: Dict containing 'topics' and 'keywords' keys.
                         Example: {'topics': ['LLMs', 'agents'], 
                                  'keywords': ['reasoning', 'planning']}
        max_results: Maximum results per search (default: 10).
        api_key: Tavily API key. If None, reads from TAVILY_API_KEY env var.
    
    Returns:
        List of dicts with schema: {title, url, description, source, published_date}.
    
    Handles:
        - Missing API key: Raises ValueError early.
        - Empty results: Returns empty list (logged as warning).
        - Low result count: Retries with advanced search depth.
        - API failures: Caught by @safe_execute decorator with retry logic.
    """
    
    if TavilyClient is None:
        raise ImportError(
            "Tavily SDK required. Install with: pip install tavily-python"
        )
    
    # Secure API key loading
    api_key = api_key or os.getenv("TAVILY_API_KEY")
    if not api_key or api_key.startswith("your_tavily"):
        raise ValueError(
            "TAVILY_API_KEY not configured. Set it in .env file "
            "or provide api_key parameter."
        )
    
    client = TavilyClient(api_key=api_key)
    results = []
    
    # Construct complex query from profile
    query = _construct_complex_query(interest_profile)
    logger.debug(f"Tavily search query: {query}")
    
    # First search attempt with standard depth
    try:
        logger.debug("Tavily search: Initial attempt (standard depth)")
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth="basic",
            include_answer=False,
            topic="general",
        )
        
        for result in response.get("results", []):
            article = {
                "title": result.get("title", "").strip(),
                "url": result.get("url", "").strip(),
                "description": (result.get("content", "") or result.get("snippet", ""))[:500].strip(),
                "source": "tavily",
                "published_date": datetime.utcnow().isoformat(),
            }
            results.append(article)
        
        logger.debug(
            f"Tavily initial search returned {len(results)} results"
        )
    
    except Exception as e:
        logger.error(f"Tavily first search attempt failed: {type(e).__name__}: {str(e)}")
        raise
    
    # Dynamic Search Depth: If too few results, retry with advanced depth
    if len(results) < 3:
        try:
            logger.info(
                f"Tavily: Got {len(results)} results, retrying with advanced search depth..."
            )
            response = client.search(
                query=query,
                max_results=max_results,
                search_depth="advanced",
                include_answer=False,
                topic="general",
            )
            
            for result in response.get("results", []):
                article = {
                    "title": result.get("title", "").strip(),
                    "url": result.get("url", "").strip(),
                    "description": (result.get("content", "") or result.get("snippet", ""))[:500].strip(),
                    "source": "tavily",
                    "published_date": datetime.utcnow().isoformat(),
                }
                results.append(article)
            
            logger.debug(
                f"Tavily advanced search returned {len(results)} total results"
            )
        
        except Exception as e:
            logger.warning(
                f"Tavily advanced retry failed: {type(e).__name__}: {str(e)}. "
                f"Continuing with {len(results)} results from initial search."
            )
            # Don't re-raise; continue with whatever results we have
    
    logger.info(
        f"Tavily fetch complete: {len(results)} results across "
        f"{'basic' if len(results) < 3 else 'basic + advanced'} search"
    )
    
    return results


# ============================================================================
# Unit Testing Block
# ============================================================================

if __name__ == "__main__":
    """
    Standalone test: Verify Tavily client can fetch results from web based
    on a sample interest profile.
    
    Run: python src/tools/tavily_client.py
    
    Expected output: List of web results with title, url, description, source, published_date.
    """
    
    print("\n" + "=" * 80)
    print("TAVILY CLIENT - UNIT TEST")
    print("=" * 80)
    
    # Sample interest profile
    sample_profile = {
        "topics": ["AI agents", "Large Language Models", "reasoning"],
        "keywords": ["breakthroughs", "startups", "2026"],
    }
    
    print(f"Interest Profile: {sample_profile}\n")
    print("Fetching web results from Tavily...\n")
    
    try:
        result = fetch_tavily_results(
            interest_profile=sample_profile,
            max_results=5,
        )
        
        raw_articles = result.get("raw_articles", [])
        logs = result.get("logs", [])
        
        print(f"Status: Retrieved {len(raw_articles)} web results\n")
        
        if raw_articles:
            print("Sample Output (first result):")
            print("-" * 80)
            article = raw_articles[0]
            print(f"Title: {article['title']}")
            print(f"URL: {article['url']}")
            print(f"Source: {article['source']}")
            print(f"Fetched: {article['published_date']}")
            print(f"Description:\n  {article['description'][:100]}...\n")
            
            print(f"Total Web Results Retrieved: {len(raw_articles)}")
            print("\nAll Results Downloaded:")
            print("-" * 80)
            for i, article in enumerate(raw_articles, 1):
                print(
                    f"{i}. {article['title'][:70]}..."
                    f"\n   URL: {article['url']}"
                    f"\n   Source: {article['source']}\n"
                )
        else:
            print("⚠️  No results retrieved. Check logs and ensure TAVILY_API_KEY is set.\n")
        
        print("Execution Logs:")
        print("-" * 80)
        for log in logs:
            level = log.get("level", "info").upper()
            message = log.get("message", "")
            print(f"[{level}] {message}")
    
    except ValueError as e:
        print(f"⚠️  Configuration Error: {str(e)}")
        print("\nPlease set TAVILY_API_KEY in your .env file:")
        print("  TAVILY_API_KEY=your_actual_api_key_here")
    
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {str(e)}")
    
    print("\n" + "=" * 80)
