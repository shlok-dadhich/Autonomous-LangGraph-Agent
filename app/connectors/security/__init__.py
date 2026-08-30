"""Security adapters — NVD/CISA/OWASP (JSON/RSS)."""

from __future__ import annotations

import requests
from app.connectors import register
from app.connectors.base import RawDocument, SourceQuery

@register("security")
@register("nvd")
@register("cisa-kev")
class SecurityConnector:
    async def fetch(self, query: SourceQuery) -> list[RawDocument]:
        return []  # enable via config/security.sources; needs no key for NVD/CISA
