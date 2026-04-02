"""
Hacker News API client for finding high-signal AI/tech stories.

Uses the official HN Firebase API with parallel fetching to efficiently
retrieve top stories filtered by keywords from interest profile.
Implements robust timeout handling and signal filtering for Phase 4.
"""

import os
import sys
from pathlib import Path
import requests
from typing import List, Dict, Any, Optional
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from loguru import logger

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.reliability import safe_execute

# HN Firebase API endpoints
HN_API_BASE = "https://hacker-news.firebaseio.com/v0"
HN_TOP_STORIES_URL = f"{HN_API_BASE}/topstories.json"
HN_ITEM_URL = f"{HN_API_BASE}/item"


def _fetch_hn_item(story_id: int, timeout: int = 10) -> Optional[Dict[str, Any]]:
    """
    Fetch a single HN item by ID.
    
    Args:
        story_id: HN story ID.
        timeout: Request timeout in seconds.
    
    Returns:
        Dict with item data, or None if fetch fails.
    """
    try:
        item_url = f"{HN_ITEM_URL}/{story_id}.json"
        response = requests.get(item_url, timeout=timeout)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.debug(f"Failed to fetch HN item {story_id}: {type(e).__name__}")
        return None


def _extract_domain(url: str) -> str:
    """
    Extract domain name from URL.
    
    Args:
        url: Full URL string.
    
    Returns:
        Domain name or "news.ycombinator.com" as fallback.
    """
    try:
        from urllib.parse import urlparse
        domain = urlparse(url).netloc if url else "news.ycombinator.com"
        return domain if domain else "news.ycombinator.com"
    except Exception:
        return "news.ycombinator.com"


def _matches_interest_profile(
    title: str,
    url: str,
    interest_profile: Dict[str, Any]
) -> bool:
    """
    Check if story matches interest profile keywords.
    
    Args:
        title: Story title.
        url: Story URL.
        interest_profile: Dict with 'keywords' and/or 'topics' lists.
    
    Returns:
        True if title or URL contains at least one keyword/topic.
    """
    keywords = interest_profile.get("keywords", [])
    topics = interest_profile.get("topics", [])
    all_terms = keywords + topics
    
    if not all_terms:
        return True  # Accept all if no filter specified
    
    title_lower = title.lower()
    url_lower = url.lower()
    
    for term in all_terms:
        term_lower = term.lower()
        if term_lower in title_lower or term_lower in url_lower:
            return True
    
    return False


@safe_execute(source_name="hackernews", max_retries=2)
def fetch_hn_stories(
    interest_profile: Dict[str, Any],
    min_score: int = 50,
    max_items: int = 100,
    timeout: int = 10,
    max_workers: int = 5,
) -> List[Dict[str, Any]]:
    """
    Fetch high-signal stories from Hacker News using parallel fetching.
    
    **METADATA SCHEMA:**
    - title (str): Story title
    - url (str): External link (not HN discussion URL)
    - description (str): Title + 'Score: X | Comments: Y'
    - source (str): Hardcoded as 'hackernews'
    - score (int): Story score for ranking
    - published_date (str): ISO 8601 timestamp of story creation
    
    **RELIABILITY FEATURES:**
    - Parallel fetching: Uses ThreadPoolExecutor to fetch up to 5 items concurrently
    - Hard cap: Max 100 story IDs fetched from top stories list (prevents runaway calls)
    - Timeout protection: Individual 10s timeouts on each request
    - Interest-based filtering: Matches story title/URL against interest_profile keywords
    - Score thresholding: Min score filter (default 50)
    - Wrapped in @safe_execute decorator with retry logic
    
    Args:
        interest_profile: Dict with 'keywords' and/or 'topics' lists.
                         Example: {'keywords': ['AI', 'python'], 
                                  'topics': ['machine learning']}
        min_score: Minimum story score to include (default: 50).
        max_items: Hard cap on story IDs to fetch (default: 100, max 100).
        timeout: Request timeout in seconds (default: 10).
        max_workers: Max concurrent threads for fetching stories (default: 5).
    
    Returns:
        List of dicts with schema: {title, url, description, score, source, published_date}.
    
    Handles:
        - Network failures: Retried via @safe_execute decorator
        - Timeout on top stories list: Returns empty list
        - Per-item fetch failures: Skipped; continues with next item
        - Interest filtering: Only stories matching keywords are returned
    """
    
    stories = []
    
    # Enforce hard cap on max items to fetch
    max_items = min(max_items, 100)
    
    try:
        # Step 1: Fetch top story IDs
        logger.debug(f"Fetching HN top {max_items} story IDs...")
        response = requests.get(HN_TOP_STORIES_URL, timeout=timeout)
        response.raise_for_status()
        
        top_story_ids = response.json()[:max_items]
        logger.debug(f"Retrieved {len(top_story_ids)} top story IDs from HN")
        
        if not top_story_ids:
            logger.warning("HN returned empty story list")
            return stories
        
        # Step 2: Fetch details in parallel
        logger.debug(
            f"Fetching {len(top_story_ids)} story details with {max_workers} workers..."
        )
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all fetch tasks
            future_to_id = {
                executor.submit(_fetch_hn_item, story_id, timeout): story_id
                for story_id in top_story_ids
            }
            
            # Process results as they complete
            for future in as_completed(future_to_id):
                story_id = future_to_id[future]
                
                try:
                    item = future.result()
                except Exception as e:
                    logger.debug(f"Future failed for story {story_id}: {str(e)}")
                    continue
                
                if not item or not item.get("title"):
                    continue
                
                # Filter by score
                item_score = item.get("score", 0)
                if item_score < min_score:
                    logger.debug(
                        f"Skipped '{item['title'][:40]}' (score {item_score} < {min_score})"
                    )
                    continue
                
                # Filter by interest profile
                url = item.get("url", "")
                if not _matches_interest_profile(item["title"], url, interest_profile):
                    logger.debug(
                        f"Skipped '{item['title'][:40]}' (no keyword match)"
                    )
                    continue
                
                # Build external URL (or fallback to HN discussion)
                external_url = url if url else f"https://news.ycombinator.com/item?id={story_id}"
                
                # Extract metadata
                domain = _extract_domain(url)
                comments = item.get("descendants", 0)
                pub_date = datetime.utcfromtimestamp(item.get("time", 0)).isoformat()
                
                story = {
                    "title": item.get("title", "").strip(),
                    "url": external_url,
                    "description": f"{item['title']} | Score: {item_score}",
                    "score": item_score,
                    "source": "hackernews",
                    "published_date": pub_date,
                }
                
                stories.append(story)
                logger.debug(
                    f"Added HN story: {item['title'][:60]}... "
                    f"(score={item_score}, comments={comments})"
                )
        
        logger.info(
            f"HN fetch complete: {len(stories)} stories after filtering "
            f"({len(top_story_ids)} candidates, min_score={min_score})"
        )
    
    except requests.RequestException as e:
        logger.error(f"HN API error fetching top stories: {type(e).__name__}: {str(e)}")
        raise
    
    except Exception as e:
        logger.error(f"Unexpected error in HN fetch: {type(e).__name__}: {str(e)}")
        raise
    
    return stories


# ============================================================================
# Unit Testing Block
# ============================================================================

if __name__ == "__main__":
    """
    Standalone test: Verify HN client can fetch high-signal stories matching
    an interest profile with parallel fetching.
    
    Run: python src/tools/hn_client.py
    
    Expected output: List of stories with title, url, description, score, source, published_date.
    """
    
    print("\n" + "=" * 80)
    print("HACKER NEWS CLIENT - UNIT TEST")
    print("=" * 80)
    
    # Sample interest profile
    sample_profile = {
        "keywords": ["AI", "LLM", "python", "machine learning"],
        "topics": ["neural networks", "agents"],
    }
    
    print(f"Interest Profile: {sample_profile}")
    print("Fetching high-signal HN stories with min_score=50...\n")
    
    result = fetch_hn_stories(
        interest_profile=sample_profile,
        min_score=50,
        max_items=30,
        max_workers=5,
    )
    
    raw_articles = result.get("raw_articles", [])
    logs = result.get("logs", [])
    
    print(f"Status: Retrieved {len(raw_articles)} stories\n")
    
    if raw_articles:
        print("Sample Output (first story):")
        print("-" * 80)
        story = raw_articles[0]
        print(f"Title: {story['title']}")
        print(f"URL: {story['url']}")
        print(f"Score: {story['score']} | Comments: {story['description'].split('|')[1] if '|' in story['description'] else 'N/A'}")
        print(f"Source: {story['source']}")
        print(f"Published: {story['published_date']}\n")
        
        print(f"Total Stories Retrieved (after filtering): {len(raw_articles)}")
        print("\nAll Stories Downloaded:")
        print("-" * 80)
        for i, story in enumerate(raw_articles, 1):
            print(
                f"{i}. {story['title'][:70]}..."
                f"\n   Score: {story['score']} | URL: {story['url'][:60]}...\n"
            )
    else:
        print("⚠️  No stories retrieved after filtering.")
        print("This may happen if HN API is slow or no stories match your interest profile.\n")
    
    print("Execution Logs (last 10):")
    print("-" * 80)
    for log in logs[-10:]:
        level = log.get("level", "info").upper()
        message = log.get("message", "")
        print(f"[{level}] {message}")
    
    print("\n" + "=" * 80)
