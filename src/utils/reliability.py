"""
Reliability and resilience utilities for the newsletter agent.

Provides safe execution wrappers with retry logic, exponential backoff,
and structured error logging for robust data harvesting operations.
"""

import time
import functools
from typing import Callable
from datetime import datetime
from loguru import logger


def _timestamp_ms() -> str:
    """Return an ISO timestamp with millisecond precision."""
    return datetime.now().isoformat(timespec="milliseconds")


def safe_execute(
    source_name: str,
    max_retries: int = 2,
) -> Callable:
    """
    Decorator for safe execution of data-harvesting functions with retry logic.
    
    Wraps a function to handle failures gracefully by:
    - Retrying with exponential backoff on failure
    - Logging all attempts and errors
    - Returning state-compatible dict: {raw_articles: [...], logs: [...]}
    
    Args:
        source_name: Identifier for the data source (e.g., "arxiv", "hackernews").
        max_retries: Number of retry attempts after initial failure (default: 2).
    
    Returns:
        Decorated function that returns dict with raw_articles and logs.
    
    Example:
        @safe_execute(source_name="arxiv", max_retries=2)
        def fetch_arxiv_papers(query: str) -> List[dict]:
            # your fetching logic
            return papers
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> dict:
            logs = []
            
            for attempt in range(max_retries + 1):
                try:
                    ts = _timestamp_ms()
                    logger.info(
                        f"[{source_name}] Attempt {attempt + 1}/{max_retries + 1}: "
                        f"Executing {func.__name__}()"
                    )
                    
                    result = func(*args, **kwargs)

                    if isinstance(result, list):
                        item_count = len(result)
                    else:
                        item_count = 0
                        result = []

                    if item_count == 0:
                        empty_msg = (
                            f"[{source_name}] Empty return from {func.__name__}(): "
                            "no items found (not an API failure)."
                        )
                        logger.warning(empty_msg)
                        logs.append({"level": "warning", "message": empty_msg, "timestamp": ts})
                    else:
                        success_msg = (
                            f"[{source_name}] ✓ Successfully executed {func.__name__}(). "
                            f"Retrieved {item_count} items."
                        )
                        logger.success(success_msg)
                        logs.append({"level": "success", "message": success_msg, "timestamp": ts})
                    
                    # Return state-compatible dict
                    return {
                        "raw_articles": result,
                        "logs": logs,
                    }
                
                except Exception as e:
                    ts = _timestamp_ms()
                    error_msg = (
                        f"[{source_name}] Attempt {attempt + 1}/{max_retries + 1} failed: "
                        f"{type(e).__name__}: {str(e)}"
                    )
                    logger.warning(error_msg)
                    logs.append({"level": "warning", "message": error_msg, "timestamp": ts})
                    
                    # Calculate exponential backoff: 2^attempt seconds
                    if attempt < max_retries:
                        backoff_time = 2 ** attempt
                        wait_msg = f"[{source_name}] Waiting {backoff_time}s before retry..."
                        logger.debug(wait_msg)
                        logs.append({"level": "debug", "message": wait_msg, "timestamp": _timestamp_ms()})
                        time.sleep(backoff_time)
            
            # All retries exhausted
            final_error = (
                f"[{source_name}] ✗ Failed after {max_retries + 1} attempts. "
                "API failure after retries. Returning empty list."
            )
            logger.error(final_error)
            logs.append({"level": "error", "message": final_error, "timestamp": _timestamp_ms()})
            
            # Return state-compatible dict with empty articles
            return {
                "raw_articles": [],
                "logs": logs,
            }
        
        return wrapper
    
    return decorator
