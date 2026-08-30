"""Entity extraction node."""

from __future__ import annotations

from loguru import logger
from app.intelligence.entities import extract_entities

def extract_entities_node(state: dict) -> dict:
    docs = state.get("documents", [])
    entities = []
    for d in docs:
        for e in extract_entities(d.get("title",""), d.get("description","")):
            entities.append({"document_id": d.get("document_id"), "canonical_name": e.canonical_name, "kind": e.kind, "aliases": e.aliases})
    logger.info(f"[entities] {len(entities)} mentions")
    return {"entities": entities, "logs": [{"level": "info", "message": f"[entities] {len(entities)} mentions"}]}
