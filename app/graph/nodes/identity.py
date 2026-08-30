"""Identity — canonical URL + fingerprint."""

from __future__ import annotations

from loguru import logger
from app.domain.documents.identity import canonicalize_url, content_hash, title_hash, document_id

def resolve_identity_node(state: dict) -> dict:
    docs = state.get("documents", [])
    enriched = []
    for d in docs:
        url = d.get("url","")
        canon = canonicalize_url(url)
        c_hash = content_hash(f"{d.get('title','')} {d.get('description','')}")
        t_hash = title_hash(d.get("title",""))
        did = str(document_id(canon, c_hash))
        nd = dict(d)
        nd.update({"canonical_url": canon, "original_url": url, "content_hash": c_hash, "title_hash": t_hash, "document_id": did})
        enriched.append(nd)
    logger.info(f"[identity] resolved {len(enriched)} identities")
    return {"documents": enriched, "logs": [{"level": "info", "message": f"[identity] resolved {len(enriched)} identities"}]}
