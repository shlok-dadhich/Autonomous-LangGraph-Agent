"""Integration tests for Phase 5 API."""

from fastapi.testclient import TestClient

from app.api.app import create_app
from app.storage.db import SessionLocal
from app.storage.models import Document, StoryCluster


def test_health():
    app = create_app()
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_dashboard():
    app = create_app()
    client = TestClient(app)
    r = client.get("/")
    assert r.status_code == 200
    assert "Your Brief" in r.text


def test_ask_grounded_empty():
    app = create_app()
    client = TestClient(app)
    r = client.post("/ask", json={"question": "What changed in AI agents this week?"})
    assert r.status_code == 200
    data = r.json()
    assert "answer" in data
    assert "citations" in data


def test_search_and_stories_empty():
    app = create_app()
    client = TestClient(app)
    assert client.get("/search?q=AI").status_code == 200
    assert client.get("/stories").status_code == 200
    assert client.get("/digests").status_code == 200


def test_feedback_post():
    app = create_app()
    client = TestClient(app)
    # need a user id; use demo user from DB
    from app.storage.db import SessionLocal
    from app.storage.models import User

    db = SessionLocal()
    user = db.query(User).first()
    db.close()
    if not user:
        return  # skip if no demo user
    r = client.post("/feedback", json={"user_id": str(user.id), "target_type": "story", "target_id": "test-id", "action": "SAVE"})
    assert r.status_code == 200
    assert r.json()["action"] == "SAVE"
