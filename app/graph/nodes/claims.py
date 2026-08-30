"""Claim/evidence node."""

from __future__ import annotations

from loguru import logger
from app.intelligence.claims import extract_claims
from app.intelligence.evidence import evaluate_evidence, detect_contradiction

def extract_claims_node(state: dict) -> dict:
    docs = state.get("documents", [])
    source_scores = state.get("source_scores", [])
    # map doc_id -> tier
    tier_map = {s.get("document_id"): s.get("tier","D") for s in source_scores} if source_scores else {}
    claims = []
    for d in docs:
        for c in extract_claims(d.get("title",""), d.get("description",""), d.get("document_id",""), d.get("url","")):
            claims.append({"document_id": d.get("document_id"), "text": c.text, "claim_type": c.claim_type, "confidence": c.confidence, "evidence_refs": c.evidence_refs, "tier": tier_map.get(d.get("document_id"),"D")})
    # evidence bundle per cluster
    evidence = []
    claim_texts = [c["text"] for c in claims]
    contradiction = detect_contradiction(claim_texts)
    if contradiction:
        evidence.append({"contradiction": contradiction})
    logger.info(f"[claims] {len(claims)} claims, contradiction={bool(contradiction)}")
    return {"claims": claims, "evidence": evidence, "logs": [{"level": "info", "message": f"[claims] {len(claims)} claims"}]}
