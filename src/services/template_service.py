"""Newsletter HTML rendering helpers."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from jinja2 import Template
from loguru import logger


class TemplateService:
    """Render the newsletter HTML using the shared Jinja2 template."""

    def __init__(self, template_dir: Optional[Path] = None) -> None:
        self.template_dir = template_dir or Path(__file__).resolve().parents[1] / "templates"
        self.template_path = self.template_dir / "email_body.html"

    @staticmethod
    def _tracking_url(article: Dict[str, Any], build_label: str) -> str:
        """Attach newsletter tracking parameters while preserving the absolute URL."""
        original_url = str(article.get("url", "")).strip()
        if not original_url:
            return ""

        parsed = urlparse(original_url)
        if not parsed.scheme or not parsed.netloc:
            return original_url

        existing_params = dict(parse_qsl(parsed.query, keep_blank_values=True))
        source_name = str(article.get("source", "newsletter")).strip().lower().replace(" ", "-")
        title_text = str(article.get("title", "")).strip().lower()
        content_slug = "-".join(part for part in title_text.split()[:6] if part)

        existing_params.update(
            {
                "utm_source": source_name or "newsletter",
                "utm_medium": "email",
                "utm_campaign": "ai_weekly_intelligence",
                "utm_content": content_slug or "article-link",
                "utm_term": build_label,
            }
        )

        return urlunparse(parsed._replace(query=urlencode(existing_params)))

    def render_newsletter(
        self,
        enriched_articles: List[Dict[str, Any]],
        current_date: Optional[str] = None,
        newsletter_title: str = "AI Weekly Intelligence",
        top_topic: Optional[str] = None,
        build_label: Optional[str] = None,
        managed_by: str = "GitHub Copilot",
        feedback_url: str = "mailto:newsletter-feedback@example.com?subject=AI%20Weekly%20Intelligence%20Feedback",
    ) -> str:
        """Render the premium newsletter template with enriched article payload."""
        template = Template(self.template_path.read_text(encoding="utf-8"))
        rendered_date = current_date or datetime.now().strftime("%b %d, %Y")
        sources_used = sorted(
            {
                article.get("source", "unknown")
                for article in enriched_articles
                if article.get("source")
            }
        )
        tracked_articles = []
        for article in enriched_articles:
            enriched_article = dict(article)
            enriched_article["tracked_url"] = self._tracking_url(enriched_article, build_label or rendered_date)
            tracked_articles.append(enriched_article)

        logger.info(
            f"[template] Tagged {len(tracked_articles)} article link(s) with newsletter tracking parameters."
        )

        return template.render(
            current_date=rendered_date,
            newsletter_title=newsletter_title,
            articles=tracked_articles,
            sources_used=sources_used,
            total_sources=len(sources_used),
            top_topic=top_topic or "RAG",
            build_label=build_label or rendered_date,
            managed_by=managed_by,
            feedback_url=feedback_url,
        )

    def render_email_html(
        self,
        email_draft_content: List[Dict[str, Any]],
        current_date: Optional[str] = None,
        newsletter_title: str = "AI Weekly Intelligence",
    ) -> str:
        """Backward-compatible alias for existing callers."""
        return self.render_newsletter(
            enriched_articles=email_draft_content,
            current_date=current_date,
            newsletter_title=newsletter_title,
        )