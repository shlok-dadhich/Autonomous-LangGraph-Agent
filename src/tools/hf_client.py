"""
Hugging Face Daily Papers API client for fetching trending research.

Fetches the most discussed and trending papers from the Hugging Face community
via the public daily_papers API endpoint. Returns high-signal implementation-layer
research directly from developers and practitioners.
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from loguru import logger

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils.reliability import safe_execute

try:
    import requests
except ImportError:
    logger.warning("requests not installed. Install with: pip install requests")
    requests = None


def _extract_hf_paper_id(item: Dict[str, Any]) -> str:
    """Extract a stable paper id from known HF daily_papers payload shapes."""
    paper_field = item.get("paper", {})

    if isinstance(paper_field, dict):
        return str(paper_field.get("id", "")).strip() or str(item.get("id", "")).strip()

    if isinstance(paper_field, str):
        return paper_field.strip() or str(item.get("id", "")).strip()

    return str(item.get("id", "")).strip()


@safe_execute(source_name="huggingface_daily", max_retries=2)
def fetch_hf_daily_papers(
    limit: int = 5,
    timeout: int = 10,
) -> List[Dict[str, Any]]:
    """
    Fetch the most discussed papers from Hugging Face daily papers feed.
    
    **METADATA SCHEMA:**
    - title (str): Paper title
    - url (str): Link to Hugging Face paper page
    - description (str): Paper abstract/summary
    - source (str): Hardcoded as 'huggingface-daily'
    - relevance_score (float): Pre-ranked by HF community engagement (0.8 default)
    
    **PUBLIC API:**
    - No authentication required (public endpoint)
    - Rate limit: Very generous (community-driven)
    - Endpoint: https://huggingface.co/api/daily_papers
    
    **RELIABILITY FEATURES:**
    - Wrapped in @safe_execute decorator with retry logic
    - Graceful handling of network timeouts and API downtime
    - Returns empty list on failure (logged)
    - JSON parsing validation
    - URL construction safety checks
    
    **THE "IMPLEMENTATION LAYER":**
    Unlike Arxiv (academia) or news (hype), HF papers represent what practitioners
    are actually building with today. High-quality signal with instant community
    validation (retweets, discussions). This is the "secret sauce" for catching
    emerging trends before mainstream media.
    
    Args:
        limit: Maximum papers to fetch (default: 5).
               Typical HF daily feed has 8-15 papers.
        timeout: HTTP request timeout in seconds (default: 10).
    
    Returns:
        List of dicts with schema: {title, url, description, source, relevance_score}.
    
    Raises:
        ImportError: If requests library is not installed.
        ValueError: If API response is invalid (logged, returns empty list via decorator).
    """
    if requests is None:
        raise ImportError("requests library not installed. Install with: pip install requests")
    
    api_url = "https://huggingface.co/api/daily_papers"
    logger.info(f"Fetching Hugging Face daily papers from {api_url}")
    
    try:
        # Fetch the daily papers JSON
        response = requests.get(api_url, timeout=timeout)
        response.raise_for_status()
        
        data = response.json()
        logger.debug(f"HF API returned {len(data)} papers")
        
        # Transform HF API response to standard schema
        articles = []
        
        for item in data[:limit]:
            try:
                # HF API wraps paper info; extract from the right level
                # Structure: item has top-level keys like 'paper', 'title', 'summary', 'publishedAt'
                paper_id = _extract_hf_paper_id(item)
                title = str(item.get("title", "")).strip()
                summary = str(item.get("summary", "")).strip()
                
                # Skip papers with missing critical fields
                if not paper_id or not title:
                    logger.debug(f"Skipping paper with missing id or title")
                    continue
                
                # Construct URL per HF paper pattern
                paper_url = f"https://huggingface.co/papers/{paper_id}"
                
                article = {
                    "title": title,
                    "url": paper_url,
                    "description": summary[:500] if summary else "",  # Cap at 500 chars
                    "source": "huggingface-daily",
                    "relevance_score": 0.8,  # HF papers are pre-ranked by community
                }
                
                articles.append(article)
                logger.debug(f"Extracted paper: {title[:60]}... -> {paper_url}")
                
            except (KeyError, ValueError, AttributeError) as e:
                logger.debug(f"Error processing HF paper: {str(e)}")
                continue
        
        logger.info(f"Successfully extracted {len(articles)} HF daily papers")
        return articles
        
    except requests.exceptions.Timeout:
        logger.error(f"HF API request timed out after {timeout}s")
        return []
    except requests.exceptions.ConnectionError as e:
        logger.error(f"HF API connection error: {str(e)}")
        return []
    except requests.exceptions.HTTPError as e:
        logger.error(f"HF API returned HTTP error: {response.status_code}")
        return []
    except json.JSONDecodeError:
        logger.error("HF API returned invalid JSON")
        return []
    except Exception as e:
        logger.error(f"Unexpected error fetching HF papers: {type(e).__name__}: {str(e)}")
        return []


if __name__ == "__main__":
    """
    Self-contained test harness for verifying HF Daily Papers client.
    
    Run independently to validate output schema without running the full graph:
        python src/tools/hf_client.py
    
    Expected output:
    - 5 papers fetched from https://huggingface.co/api/daily_papers
    - Each paper has: title, url, description, source, relevance_score
    - All URLs follow https://huggingface.co/papers/{paper_id} pattern
    - source field is always "huggingface-daily"
    """
    
    print("\n" + "=" * 80)
    print("HF Daily Papers Client - Independent Verification")
    print("=" * 80)
    
    try:
        result = fetch_hf_daily_papers(limit=5)
        
        # Handle both dict (from @safe_execute decorator) and list (direct call)
        if isinstance(result, dict):
            papers = result.get("raw_articles", [])
        else:
            papers = result
        
        if not papers:
            print("⚠️  No papers returned (API may be down or requests library missing)")
            sys.exit(1)
        
        print(f"\n✅ Successfully fetched {len(papers)} papers\n")
        
        # Validate schema and display results
        required_keys = {"title", "url", "description", "source", "relevance_score"}
        
        all_valid = True
        for i, paper in enumerate(papers, 1):
            print(f"Paper {i}:")
            print(f"  Title: {paper.get('title', 'N/A')[:70]}")
            print(f"  URL: {paper.get('url', 'N/A')}")
            print(f"  Description: {paper.get('description', 'N/A')[:80]}...")
            print(f"  Source: {paper.get('source', 'N/A')}")
            print(f"  Relevance Score: {paper.get('relevance_score', 'N/A')}")
            
            # Validate schema
            missing_keys = required_keys - set(paper.keys())
            if missing_keys:
                print(f"  ❌ Missing keys: {missing_keys}")
                all_valid = False
            else:
                print(f"  ✓ Schema valid")
            
            # Validate URL format
            url = paper.get("url", "")
            if not url.startswith("https://huggingface.co/papers/"):
                print(f"  ❌ Invalid URL format: {url}")
                all_valid = False
            
            print()
        
        if all_valid:
            print("✅ All papers pass schema validation!")
            print("\nMediaSchema:")
            print(json.dumps(papers[0], indent=2))
        else:
            print("⚠️  Some papers failed schema validation")
            sys.exit(1)
            
    except Exception as e:
        print(f"❌ Error: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
