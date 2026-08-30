"""GitHub adapter — releases + trending."""

from __future__ import annotations

import os
import requests

from app.connectors import register
from app.connectors.base import RawDocument, SourceQuery


@register("github")
class GithubConnector:
    async def fetch(self, query: SourceQuery) -> list[RawDocument]:
        # Phase 5: use GitHub Search API via trending or releases if token available
        token = os.getenv("GITHUB_TOKEN")
        headers = {"Accept": "application/vnd.github+json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        q = " ".join(query.keywords or query.topics or ["AI"]) or "AI"
        try:
            # Search trending repos via stars
            resp = requests.get(
                "https://api.github.com/search/repositories",
                params={"q": q, "sort": "stars", "order": "desc", "per_page": query.max_results},
                headers=headers,
                timeout=15,
            )
            if resp.status_code != 200:
                return []
            items = resp.json().get("items", [])
            out: list[RawDocument] = []
            for it in items[: query.max_results]:
                out.append(
                    RawDocument(
                        title=it.get("full_name", ""),
                        url=it.get("html_url", ""),
                        description=it.get("description", "") or "",
                        source="github",
                        external_id=it.get("full_name"),
                        metadata={"stars": it.get("stargazers_count"), "forks": it.get("forks_count")},
                    )
                )
            return out
        except Exception:
            return []
