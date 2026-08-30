"""Story clustering node."""

from __future__ import annotations

from loguru import logger
from app.intelligence.clustering import ClusterInput, cluster_documents
from app.intelligence.entities import extract_entities

def cluster_documents_node(state: dict) -> dict:
    docs = state.get("documents", [])
    inputs = []
    for d in docs:
        ents = [e.canonical_name for e in extract_entities(d.get("title",""), d.get("description",""))]
        inputs.append(ClusterInput(id=d.get("document_id", d.get("url","")), title=d.get("title",""), text=d.get("description",""), source=d.get("source",""), entities=ents))
    clusters = cluster_documents(inputs)
    out = [{"cluster_id": c.cluster_id, "cluster_confidence": c.cluster_confidence, "cluster_reason": c.cluster_reason, "document_ids": c.document_ids, "title": c.title} for c in clusters]
    logger.info(f"[cluster] {len(docs)} docs -> {len(out)} clusters")
    return {"clusters": out, "logs": [{"level": "info", "message": f"[cluster] formed {len(out)} story clusters from {len(docs)} docs"}]}
