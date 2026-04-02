"""LLM-backed newsletter synthesis utilities."""

import json
import os
import re
import time
from typing import Any, Dict, List, Sequence

from loguru import logger


DEFAULT_GROQ_MODEL = "llama-3.1-70b-versatile"
DEFAULT_BATCH_SIZE = 3
DEFAULT_MAX_RETRIES = 2
DEFAULT_SINGLE_PROMPT_CHAR_LIMIT = 12000

SYSTEM_PROMPT = (
    "You are a Senior AI Research Lead at a top lab. "
    "Write for technical readers who care about implementation details and model mechanics. "
    "Be analytical, concise, and personalized to the user's interests. "
    "Avoid all introductory fluff, casual filler, and marketing language. "
    "Return only valid JSON."
)


class NewsletterWriter:
    """Generate concise executive intelligence summaries via Groq."""

    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_retries: int = DEFAULT_MAX_RETRIES,
        single_prompt_char_limit: int = DEFAULT_SINGLE_PROMPT_CHAR_LIMIT,
    ) -> None:
        self.model_name = model_name or os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.batch_size = max(1, int(batch_size))
        self.max_retries = max(0, int(max_retries))
        self.single_prompt_char_limit = max(1000, int(single_prompt_char_limit))

    @staticmethod
    def _profile_context(interest_profile: Dict[str, Any]) -> str:
        topics = [str(item).strip() for item in interest_profile.get("topics", []) if item]
        keywords = [str(item).strip() for item in interest_profile.get("keywords", []) if item]
        return (
            "User topics: " + (", ".join(topics) if topics else "none") + "\n"
            "User keywords: " + (", ".join(keywords) if keywords else "none")
        )

    @staticmethod
    def _topic_hint(interest_profile: Dict[str, Any], limit: int = 3) -> str:
        topics = [str(item).strip() for item in interest_profile.get("topics", []) if item]
        keywords = [str(item).strip() for item in interest_profile.get("keywords", []) if item]
        combined = topics[:limit] + keywords[:limit]
        if not combined:
            return "your AI priorities"
        return ", ".join(combined[: limit + 1])

    @staticmethod
    def _normalize_lines(lines: Any, fallback_text: str) -> List[str]:
        if isinstance(lines, list):
            normalized = [str(line).strip() for line in lines if str(line).strip()]
        else:
            normalized = []

        if not normalized:
            summary = " ".join(fallback_text.split()) or "No description provided."
            segments = [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", summary) if segment.strip()]
            normalized = [segment[:160] for segment in segments[:3]]

        while len(normalized) < 3:
            normalized.append("")

        return normalized[:3]

    @staticmethod
    def _to_executive_sentence(text: str, prefix: str, fallback: str) -> str:
        """Translate snippet fragments into concise executive language."""
        cleaned = " ".join(str(text).split())
        if not cleaned:
            return fallback

        sentence = re.split(r"(?<=[.!?])\s+", cleaned)[0].strip()[:220]
        if not sentence:
            return fallback

        sentence = sentence[0].lower() + sentence[1:] if len(sentence) > 1 else sentence.lower()
        return f"{prefix} {sentence.rstrip('.')} .".replace(" .", ".")

    @staticmethod
    def _format_personalized_insight(insight: str, topic_hint: str) -> str:
        base = " ".join(str(insight).split())
        if not base:
            base = f"This is directly actionable for {topic_hint}."
        return f"**Personalized Insight:** {base}"

    @staticmethod
    def _fallback_enrichment(article: Dict[str, Any], interest_profile: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback summary that translates source snippets into executive What/How/Impact."""
        description = " ".join(str(article.get("description", "")).split())
        topic_hint = NewsletterWriter._topic_hint(interest_profile)
        what_line = NewsletterWriter._to_executive_sentence(
            description,
            "What:",
            "What: Introduces a practical AI system update relevant to production teams.",
        )
        how_line = NewsletterWriter._to_executive_sentence(
            description,
            "How:",
            "How: Uses a concrete implementation approach with measurable engineering tradeoffs.",
        )
        summary_lines = [what_line, how_line, ""]

        return {
            "title": article.get("title", ""),
            "url": article.get("url", ""),
            "source": article.get("source", ""),
            "relevance_score": article.get("relevance_score"),
            "summary": "\n".join(summary_lines),
            "summary_lines": summary_lines,
            "personalized_insight": NewsletterWriter._format_personalized_insight(
                f"Connect this to {topic_hint} to improve reliability, latency, or model quality in your stack.",
                topic_hint,
            ),
        }

    def _system_prompt(self, interest_profile: Dict[str, Any]) -> str:
        topic_hint = self._topic_hint(interest_profile)
        return (
            f"{SYSTEM_PROMPT} Prioritize the user's interests: {topic_hint}. "
            "For each article, produce an elite 3-layer summary for a senior engineer. "
            "The output must include: what (one sentence technical objective), how (one sentence architecture/method), "
            "and personalized_insight (one sentence impact tied to user topics/keywords). "
            "Context is limited to title + description + source, so infer cautiously and avoid hallucinations. "
            "If context is thin, translate the original snippet into executive language instead of repeating it verbatim. "
            "personalized_insight must start with '**Personalized Insight:**'. "
            "The URL must be passed through exactly as provided in the input JSON. "
            "Return only valid JSON."
        )

    def _build_single_messages(self, interest_profile: Dict[str, Any], article: Dict[str, Any]) -> List[Dict[str, str]]:
        profile_context = self._profile_context(interest_profile)
        user_prompt = (
            f"{profile_context}\n\n"
            "Article context bundle (title + description + source):\n"
            f"Title: {article.get('title', '')}\n"
            f"URL: {article.get('url', '')}\n"
            f"Source: {article.get('source', '')}\n"
            f"Description: {article.get('description', '')}\n\n"
            "The URL must be passed through exactly as provided in the input JSON.\n"
            "Return a JSON list with exactly one object in this schema:\n"
            '[{"title":"...","url":"...","source":"...","relevance_score":0.0,"what":"...","how":"...","personalized_insight":"**Personalized Insight:** ..."}]'
        )

        return [
            {"role": "system", "content": self._system_prompt(interest_profile)},
            {"role": "user", "content": user_prompt},
        ]

    def _build_batch_messages(
        self,
        interest_profile: Dict[str, Any],
        articles: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, str]]:
        profile_context = self._profile_context(interest_profile)
        article_blocks = []
        for index, article in enumerate(articles, start=1):
            article_blocks.append(
                f"Article {index}\n"
                f"Title: {article.get('title', '')}\n"
                f"URL: {article.get('url', '')}\n"
                f"Source: {article.get('source', '')}\n"
                f"Description: {article.get('description', '')}"
            )

        user_prompt = (
            f"{profile_context}\n\n"
            "Summarize each article in the same order as provided.\n"
            "Each output item must include title, url, source, relevance_score, what, how, and personalized_insight.\n"
            "Use the article context bundle (title + description + source) for synthesis.\n"
            "The URL must be passed through exactly as provided in the input JSON.\n"
            "what and how must each be exactly one sentence.\n"
            "personalized_insight must explicitly reference the user's topics or keywords and begin with '**Personalized Insight:**'.\n"
            "If context is insufficient, translate the snippet into executive language rather than repeating phrases.\n\n"
            + "\n\n".join(article_blocks)
            + "\n\nReturn a JSON-valid list of objects in this exact schema:\n"
            '[{"title":"...","url":"...","source":"...","relevance_score":0.0,"what":"...","how":"...","personalized_insight":"**Personalized Insight:** ..."}]'
        )

        return [
            {"role": "system", "content": self._system_prompt(interest_profile)},
            {"role": "user", "content": user_prompt},
        ]

    @staticmethod
    def _is_retryable_error(exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        if status_code in {429, 500, 502, 503, 504}:
            return True

        name = exc.__class__.__name__.lower()
        message = str(exc).lower()
        return any(
            token in name or token in message
            for token in ("rate", "timeout", "temporarily", "overloaded", "unavailable")
        )

    def _invoke_groq(self, messages: List[Dict[str, str]], max_tokens: int) -> str:
        if not self.api_key:
            raise ValueError("GROQ_API_KEY is not configured")

        from groq import Groq

        client = Groq(api_key=self.api_key)
        last_exc: Exception | None = None

        for attempt in range(self.max_retries + 1):
            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content or "{}"
            except Exception as exc:  # pragma: no cover - vendor exceptions vary
                last_exc = exc
                if attempt >= self.max_retries or not self._is_retryable_error(exc):
                    raise

                delay_seconds = min(2 ** attempt, 8)
                logger.warning(
                    f"[writer] Groq request failed on attempt {attempt + 1}/{self.max_retries + 1}: "
                    f"{type(exc).__name__}: {str(exc)}. Retrying in {delay_seconds}s."
                )
                time.sleep(delay_seconds)

        if last_exc:
            raise last_exc
        raise RuntimeError("Groq request failed without raising a captured exception")

    @staticmethod
    def _extract_article_list(parsed: Any) -> List[Dict[str, Any]]:
        """Accept either a list payload or an object wrapper containing a list."""
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]

        if isinstance(parsed, dict):
            payload = parsed.get("articles")
            if isinstance(payload, list):
                return [item for item in payload if isinstance(item, dict)]

        return []

    @staticmethod
    def _normalize_single_response(
        original_article: Dict[str, Any],
        parsed: Dict[str, Any],
        interest_profile: Dict[str, Any],
    ) -> Dict[str, Any]:
        what_line = str(parsed.get("what", "")).strip()
        how_line = str(parsed.get("how", "")).strip()

        if not what_line:
            what_line = NewsletterWriter._to_executive_sentence(
                original_article.get("description", ""),
                "What:",
                "What: Introduces a practical AI system update relevant to production teams.",
            )
        elif not what_line.lower().startswith("what:"):
            what_line = f"What: {what_line}"

        if not how_line:
            how_line = NewsletterWriter._to_executive_sentence(
                original_article.get("description", ""),
                "How:",
                "How: Uses a concrete implementation approach with measurable engineering tradeoffs.",
            )
        elif not how_line.lower().startswith("how:"):
            how_line = f"How: {how_line}"

        summary_lines = [what_line, how_line, ""]

        personalized_insight = str(
            parsed.get("personalized_insight", parsed.get("impact", parsed.get("why_this_matters", "")))
        ).strip()
        if not personalized_insight:
            personalized_insight = NewsletterWriter._fallback_enrichment(original_article, interest_profile)[
                "personalized_insight"
            ]
        else:
            personalized_insight = NewsletterWriter._format_personalized_insight(
                personalized_insight.replace("**Personalized Insight:**", "").strip(),
                NewsletterWriter._topic_hint(interest_profile),
            )

        return {
            "title": str(parsed.get("title") or original_article.get("title", "")),
            # URL is immutable: always preserve the verified upstream value.
            "url": str(original_article.get("url", "")),
            "source": str(parsed.get("source") or original_article.get("source", "")),
            "relevance_score": parsed.get("relevance_score", original_article.get("relevance_score")),
            "summary": "\n".join(summary_lines),
            "summary_lines": summary_lines,
            "personalized_insight": personalized_insight,
        }

    def _call_single(self, interest_profile: Dict[str, Any], article: Dict[str, Any]) -> Dict[str, Any]:
        content = self._invoke_groq(self._build_single_messages(interest_profile, article), max_tokens=320)
        parsed = json.loads(content)
        payload = self._extract_article_list(parsed)
        if len(payload) != 1:
            raise ValueError("Groq single response did not return a one-item JSON list")
        return self._normalize_single_response(article, payload[0], interest_profile)

    def _call_batch(
        self,
        interest_profile: Dict[str, Any],
        articles: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        content = self._invoke_groq(self._build_batch_messages(interest_profile, articles), max_tokens=900)
        parsed = json.loads(content)
        payload = self._extract_article_list(parsed)
        if not isinstance(payload, list) or len(payload) != len(articles):
            raise ValueError("Groq batch response did not return a JSON list with one entry per article")

        normalized_articles: List[Dict[str, Any]] = []
        for original_article, parsed_article in zip(articles, payload):
            if not isinstance(parsed_article, dict):
                raise ValueError("Groq batch response contained a non-object article entry")
            normalized_articles.append(
                self._normalize_single_response(original_article, parsed_article, interest_profile)
            )

        return normalized_articles

    def generate_analysis(
        self,
        interest_profile: Dict[str, Any],
        article_list: List[Dict[str, Any]],
        batch_size: int | None = None,
    ) -> List[Dict[str, Any]]:
        """Generate executive-level analysis for one or more articles."""
        if not article_list:
            return []

        effective_batch_size = max(1, int(batch_size or self.batch_size))
        total_description_chars = sum(len(str(article.get("description", ""))) for article in article_list)
        enriched_articles: List[Dict[str, Any]] = []
        total_batches = (len(article_list) + effective_batch_size - 1) // effective_batch_size

        if len(article_list) <= effective_batch_size and total_description_chars <= self.single_prompt_char_limit:
            try:
                enriched_articles = self._call_batch(interest_profile=interest_profile, articles=article_list)
                logger.info(
                    f"[writer] Summarized {len(article_list)} article(s) in a single prompt."
                )
                return enriched_articles
            except Exception as exc:
                logger.warning(
                    f"[writer] Single-prompt analysis failed: {type(exc).__name__}: {str(exc)}. "
                    "Falling back to per-article summarization."
                )

        for batch_index, start in enumerate(range(0, len(article_list), effective_batch_size), start=1):
            batch = article_list[start : start + effective_batch_size]

            try:
                enriched_batch = self._call_batch(interest_profile=interest_profile, articles=batch)
                enriched_articles.extend(enriched_batch)
                logger.info(
                    f"[writer] Summarized batch {batch_index}/{total_batches} with {len(batch)} article(s)."
                )
            except Exception as exc:
                logger.warning(
                    f"[writer] Batch summarization failed for batch {batch_index}/{total_batches}: "
                    f"{type(exc).__name__}: {str(exc)}. Falling back to per-article summarization."
                )

                for article_index, article in enumerate(batch, start=start + 1):
                    try:
                        enriched_article = self._call_single(interest_profile=interest_profile, article=article)
                        enriched_articles.append(enriched_article)
                        logger.info(
                            f"[writer] Summarized article {article_index}/{len(article_list)}: "
                            f"{article.get('title', '')[:80]}"
                        )
                    except Exception as single_exc:
                        logger.error(
                            f"[writer] LLM summarization failed for article {article_index}: "
                            f"{type(single_exc).__name__}: {str(single_exc)}"
                        )
                        enriched_articles.append(self._fallback_enrichment(article, interest_profile))

        return enriched_articles

    def generate_summaries(
        self,
        interest_profile: Dict[str, Any],
        filtered_articles: List[Dict[str, Any]],
        batch_size: int | None = None,
    ) -> List[Dict[str, Any]]:
        """Backward-compatible alias for existing callers."""
        return self.generate_analysis(
            article_list=filtered_articles,
            interest_profile=interest_profile,
            batch_size=batch_size,
        )