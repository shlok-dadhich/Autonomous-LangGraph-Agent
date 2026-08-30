"""Tests for typed settings (app/core/config.py)."""

from __future__ import annotations


def test_settings_defaults(monkeypatch):
    from app.core.config import Settings

    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("NEWSLETTER_TIMEZONE", raising=False)
    s = Settings(_env_file=None)
    assert s.timezone == "UTC"
    assert str(s.data_dir) == "data"
    assert s.groq_model == "llama-3.3-70b-versatile"
    assert s.database_url is None
    assert s.is_postgres() is False
    assert "sqlite" in s.resolved_database_url


def test_settings_respects_env_overrides(monkeypatch):
    from pathlib import Path

    from app.core.config import Settings

    monkeypatch.setenv("NEWSLETTER_TIMEZONE", "Asia/Kolkata")
    monkeypatch.setenv("NEWSLETTER_DATA_DIR", "/tmp/test-data")
    monkeypatch.setenv("GROQ_MODEL", "llama-3.1-8b")
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://user:pass@localhost/db")
    s = Settings(_env_file=None)
    assert s.timezone == "Asia/Kolkata"
    assert Path(s.data_dir) == Path("/tmp/test-data")
    assert s.groq_model == "llama-3.1-8b"
    assert s.is_postgres() is True


def test_settings_recipient_fallback(monkeypatch):
    from app.core.config import Settings

    monkeypatch.delenv("SMTP_TO", raising=False)
    monkeypatch.setenv("SMTP_USER", "me@example.com")
    s = Settings(_env_file=None, smtp_user="me@example.com")
    assert s.recipient_email == "me@example.com"

    s2 = Settings(_env_file=None, smtp_user="a@b.com", smtp_to="to@c.com")
    assert s2.recipient_email == "to@c.com"


def test_get_settings_is_cached():
    from app.core.config import get_settings, _reset_settings_cache

    _reset_settings_cache()
    a = get_settings()
    b = get_settings()
    assert a is b
    _reset_settings_cache()
    c = get_settings()
    # after reset, still same values but cache was cleared
    assert c.timezone == a.timezone
