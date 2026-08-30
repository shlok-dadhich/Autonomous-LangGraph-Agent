"""Event extraction node."""

from __future__ import annotations

from loguru import logger
from app.intelligence.events import extract_events

def detect_events_node(state: dict) -> dict:
    docs = state.get("documents", [])
    events = []
    for d in docs:
        for ev in extract_events(d.get("title",""), d.get("description",""), d.get("source","")):
            events.append({"document_id": d.get("document_id"), "event_type": ev.event_type, "confidence": ev.confidence, "evidence": ev.evidence})
    logger.info(f"[events] extracted {len(events)} events from {len(docs)} docs")
    return {"events": events, "logs": [{"level": "info", "message": f"[events] {len(events)} events"}]}
