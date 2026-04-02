"""
Arxiv API client for fetching recent papers in AI and ML domains.

Queries the Arxiv RSS feed for papers from cs.AI and cs.LG categories
published in the last 7 days. Includes robust error handling, metadata
extraction, and independent test capability.
"""

import feedparser
import re
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any
from loguru import logger

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.reliability import safe_execute


def _extract_first_sentences(text: str, num_sentences: int = 3) -> str:
    """
    Extract the first N sentences from a text block.
    
    Args:
        text: The full text to extract from.
        num_sentences: Number of sentences to extract (default: 3).
    
    Returns:
        String containing the first N sentences, or full text if fewer available.
    """
    if not text:
        return ""
    
    # Split on sentence boundaries (period, exclamation, question mark)
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    selected = sentences[:num_sentences]
    result = ' '.join(selected)
    
    # Ensure consistent ending
    if result and not result.endswith(('.', '!', '?')):
        result += '.'
    
    return result


@safe_execute(source_name="arxiv", max_retries=2)
def fetch_arxiv_papers(
    categories: List[str] = None,
    days_back: int = 7,
    max_results: int = 20,
) -> List[Dict[str, Any]]:
    """
    Fetch recent papers from Arxiv in specified categories.
    
    **METADATA SCHEMA:**
    - title (str): Full paper title
    - url (str): Link to abstract page
    - description (str): First 3 sentences of abstract
    - source (str): Hardcoded as 'arxiv'
    - published_date (str): ISO 8601 format, filtered to last N days
    
    Args:
        categories: List of Arxiv categories (e.g., ['cs.AI', 'cs.LG']).
                   Defaults to ['cs.AI', 'cs.LG'].
        days_back: Number of days to look back (default: 7).
        max_results: Maximum number of papers to retrieve (default: 20).
    
    Returns:
        List of dicts with schema: {title, url, description, source, published_date}.
    
    Handles:
        - 503 Service Unavailable: Returns empty list, logs status.
        - Feed parsing warnings: Logs but continues.
        - Network errors: Caught by @safe_execute decorator with retry logic.
    """
    
    if categories is None:
        categories = ["cs.AI", "cs.LG"]
    
    papers = []
    cutoff_date = datetime.utcnow() - timedelta(days=days_back)
    
    for category in categories:
        try:
            # Arxiv RSS feed URL for category
            url = f"http://arxiv.org/rss/{category}"
            logger.debug(f"Fetching Arxiv papers from category: {category}")
            
            feed = feedparser.parse(url)
            
            # Handle bozo feeds (parsing errors, 503s, etc.)
            if feed.bozo:
                if "503" in str(feed.bozo_exception) or "Service Unavailable" in str(feed.bozo_exception):
                    logger.warning(
                        f"Arxiv API returned 503 (Maintenance) for {category}. "
                        "Skipping category and returning empty list."
                    )
                    return []
                else:
                    logger.warning(
                        f"Arxiv feed parsing warning for {category}: {feed.bozo_exception}"
                    )
            
            # Extract entries, respecting max_results per category
            entries_to_process = feed.entries[:max_results]
            
            for entry in entries_to_process:
                # Parse published date
                try:
                    pub_date = datetime(*entry.published_parsed[:6])
                except (AttributeError, TypeError, ValueError):
                    pub_date = datetime.utcnow()
                
                # Filter by date threshold
                if pub_date < cutoff_date:
                    logger.debug(
                        f"Skipping paper '{entry.title[:50]}...' "
                        f"(published {pub_date.date()}, before cutoff {cutoff_date.date()})"
                    )
                    continue
                
                # Extract arxiv ID from entry ID for consistent paper URL
                arxiv_id = entry.id.split("/abs/")[-1]
                paper_url = f"https://arxiv.org/abs/{arxiv_id}"
                
                # Extract first 3 sentences of abstract (description)
                description = _extract_first_sentences(entry.summary, num_sentences=3)
                
                paper = {
                    "title": entry.title.strip(),
                    "url": paper_url,
                    "description": description,
                    "source": "arxiv",
                    "published_date": pub_date.isoformat(),
                }
                
                papers.append(paper)
                logger.debug(
                    f"Added paper: {paper['title'][:60]}... "
                    f"(published {pub_date.date()})"
                )
                
                if len(papers) >= max_results:
                    break
            
            logger.debug(
                f"Arxiv category '{category}': {len(papers)} papers collected"
            )
        
        except Exception as e:
            logger.error(
                f"Error fetching Arxiv papers from {category}: "
                f"{type(e).__name__}: {str(e)}"
            )
            # Re-raise to trigger @safe_execute retry logic
            raise
    
    logger.info(
        f"Arxiv fetch complete: {len(papers)} papers across {len(categories)} categories"
    )
    return papers


# ============================================================================
# Unit Testing Block
# ============================================================================

if __name__ == "__main__":
    """
    Standalone test: Verify Arxiv client can fetch papers from the last 3 days
    in the 'cs.AI' category.
    
    Run: python src/tools/arxiv_client.py
    
    Expected output: List of papers with title, url, description, source, published_date.
    """
    
    print("\n" + "=" * 80)
    print("ARXIV CLIENT - UNIT TEST")
    print("=" * 80)
    print("Fetching papers from last 3 days in cs.AI category...\n")
    
    result = fetch_arxiv_papers(
        categories=["cs.AI"],
        days_back=3,
        max_results=5,
    )
    
    raw_articles = result.get("raw_articles", [])
    logs = result.get("logs", [])
    
    print(f"Status: Retrieved {len(raw_articles)} papers\n")
    
    if raw_articles:
        print("Sample Output (first paper):")
        print("-" * 80)
        paper = raw_articles[0]
        print(f"Title: {paper['title']}")
        print(f"URL: {paper['url']}")
        print(f"Source: {paper['source']}")
        print(f"Published: {paper['published_date']}")
        print(f"Description:\n  {paper['description']}\n")
        
        print(f"Total Papers Retrieved: {len(raw_articles)}")
        print("\nAll Papers Downloaded:")
        print("-" * 80)
        for i, paper in enumerate(raw_articles, 1):
            print(
                f"{i}. {paper['title'][:70]}..."
                f"\n   URL: {paper['url']}"
                f"\n   Published: {paper['published_date']}\n"
            )
    else:
        print("⚠️  No papers retrieved. Check logs below.\n")
    
    print("Execution Logs:")
    print("-" * 80)
    for log in logs:
        level = log.get("level", "info").upper()
        message = log.get("message", "")
        print(f"[{level}] {message}")
    
    print("\n" + "=" * 80)
