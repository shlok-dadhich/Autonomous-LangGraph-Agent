"""Tests for Phase 4 personalization."""

import time

from app.intelligence.personalization import Interaction, aggregate_profile, user_affinity_score
from app.providers.email.resend import ingest_resend_webhook


def test_aggregate_decay():
    now = time.time()
    inters = [
        Interaction(action="SAVE", target_type="story", target_id="1", timestamp=now - 86400, meta={"topic": "RAG"}),
        Interaction(action="DISLIKE", target_type="story", target_id="2", timestamp=now - 86400 * 10, meta={"topic": "RAG"}),
    ]
    slices = aggregate_profile(inters, now=now)
    # SAVE more recent should dominate DISLIKE older
    assert slices["recent"].topic_affinity["RAG"] > 0
    assert slices["long"].topic_affinity["RAG"] > 0
    # session empty (older than 2h)
    assert slices["session"].topic_affinity == {}


def test_user_affinity():
    now = time.time()
    inters = [Interaction(action="CLICK", target_type="document", target_id="d1", timestamp=now, meta={"topic": "LLM", "source": "arxiv"})]
    slices = aggregate_profile(inters, now=now)
    aff = user_affinity_score(slices, ["LLM"], ["OpenAI"], "arxiv")
    assert 0 <= aff <= 1
    assert aff > 0.5  # CLICK on LLM+arxiv should boost


def test_affinity_neutral_without_interactions():
    aff = user_affinity_score({}, [], [], "unknown")
    assert aff == 0.5


def test_resend_webhook_ingest():
    payload = {"type": "email.delivered", "data": {"email_id": "123"}}
    out = ingest_resend_webhook(payload)
    assert out["event_type"] == "delivered"
    assert out["message_id"] == "123"


def test_feedback_service_record(tmp_path=False):
    # simple import check — DB integration tested via SessionLocal
    from app.services.feedback_service import VALID_ACTIONS

    assert "SAVE" in VALID_ACTIONS
    assert "MUTE_TOPIC" in VALID_ACTIONS
