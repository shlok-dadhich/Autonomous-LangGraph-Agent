"""Typed settings for the Personal Intelligence Platform.

Centralizes all env / .env handling via pydantic-settings. Keeps
legacy NEWSLETTER_* vars working while introducing a clean typed
interface for Phase 1+ (providers, DB, API).

Usage:
    from app.core.config import get_settings
    settings = get_settings()
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from env / .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Core ---
    app_env: Literal["development", "staging", "production"] = Field(default="development", alias="APP_ENV")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    timezone: str = Field(default="UTC", alias="NEWSLETTER_TIMEZONE")
    data_dir: Path = Field(default=Path("data"), alias="NEWSLETTER_DATA_DIR")

    # --- Database ---
    # If unset, app falls back to SQLite under data_dir (dev/CI).
    # Postgres primary when set: postgresql+psycopg://...
    database_url: str | None = Field(default=None, alias="DATABASE_URL")

    # --- Search ---
    tavily_api_key: SecretStr | None = Field(default=None, alias="TAVILY_API_KEY")

    # --- LLM ---
    groq_api_key: SecretStr | None = Field(default=None, alias="GROQ_API_KEY")
    groq_model: str = Field(default="llama-3.3-70b-versatile", alias="GROQ_MODEL")

    # Extensible provider routing (Phase 1+)
    llm_provider: str = Field(default="groq", alias="LLM_PROVIDER")
    embedding_provider: str = Field(default="local:all-MiniLM-L6-v2", alias="EMBEDDING_PROVIDER")
    reranker_provider: str | None = Field(default=None, alias="RERANKER_PROVIDER")
    search_provider: str = Field(default="tavily", alias="SEARCH_PROVIDER")
    email_provider: str = Field(default="smtp", alias="EMAIL_PROVIDER")

    # --- Email (SMTP) ---
    smtp_host: str = Field(default="smtp.gmail.com", alias="SMTP_HOST")
    smtp_port: int = Field(default=587, alias="SMTP_PORT")
    smtp_user: str | None = Field(default=None, alias="SMTP_USER")
    smtp_app_pass: SecretStr | None = Field(default=None, alias="SMTP_APP_PASS")
    smtp_to: str | None = Field(default=None, alias="SMTP_TO")

    # --- Telegram (ops) ---
    telegram_bot_token: SecretStr | None = Field(default=None, alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str | None = Field(default=None, alias="TELEGRAM_CHAT_ID")

    # --- Derived helpers ---
    @property
    def resolved_data_dir(self) -> Path:
        return Path(self.data_dir)

    @property
    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        # fallback to SQLite file under data_dir
        db_path = self.resolved_data_dir / "history.db"
        return f"sqlite:///{db_path}"

    @property
    def recipient_email(self) -> str | None:
        return self.smtp_to or self.smtp_user

    def require(self, *field_names: str) -> None:
        """Raise if any required field is missing/empty."""
        missing: list[str] = []
        for name in field_names:
            val = getattr(self, name, None)
            if val is None:
                missing.append(name)
                continue
            if isinstance(val, SecretStr):
                if not val.get_secret_value():
                    missing.append(name)
            elif isinstance(val, str) and not val.strip():
                missing.append(name)
        if missing:
            raise ValueError(f"Missing required settings: {', '.join(missing)}")

    def is_postgres(self) -> bool:
        return self.database_url is not None and self.database_url.startswith("postgresql")

    # Convenience: check legacy NEWSLETTER_* aliases still work
    @property
    def newsletter_timezone(self) -> str:
        return self.timezone

    @property
    def newsletter_data_dir(self) -> Path:
        return self.resolved_data_dir


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached singleton — call once at startup; use _reset for tests."""
    return Settings()


def _reset_settings_cache() -> None:
    """Clear cached settings (for tests)."""
    get_settings.cache_clear()
