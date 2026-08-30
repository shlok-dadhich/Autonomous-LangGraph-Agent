"""Document identity — canonical URL, fingerprints, stable IDs."""

from __future__ import annotations

import hashlib
import re
import uuid
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

TRACKING_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_content",
    "utm_term",
    "utm_id",
    "utm_source_platform",
    "utm_creative_format",
    "utm_marketing_tactic",
    "gclid",
    "fbclid",
    "igshid",
    "mc_cid",
    "mc_eid",
    "dclid",
    "yclid",
    "fb_action_ids",
    "fb_action_types",
}


def canonicalize_url(url: str) -> str:
    """Normalize URL: lowercase host, strip www, remove tracking params, sort query, strip trailing slash."""
    url = url.strip()
    if not url:
        return ""
    parsed = urlparse(url)
    scheme = (parsed.scheme or "https").lower()
    netloc = parsed.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    # also handle bare domain without scheme
    if not netloc and parsed.path:
        # urlparse without scheme treats host as path
        return url
    q = [(k, v) for k, v in parse_qsl(parsed.query, keep_blank_values=True) if k.lower() not in TRACKING_PARAMS]
    q.sort()
    path = parsed.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    return urlunparse((scheme, netloc, path, "", urlencode(q), ""))


def content_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.strip())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def title_hash(title: str) -> str:
    normalized = re.sub(r"\s+", " ", title.strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def document_id(canonical_url: str, content_hash_val: str | None = None) -> uuid.UUID:
    """Deterministic UUID5 from canonical URL + content hash for stable identity."""
    base = canonical_url
    if content_hash_val:
        base = f"{canonical_url}#{content_hash_val[:16]}"
    return uuid.uuid5(uuid.NAMESPACE_URL, base)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())
