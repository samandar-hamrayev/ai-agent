from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """All runtime configuration, read from environment variables / .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    anthropic_api_key: SecretStr = SecretStr("")
    llm_model: str = "claude-opus-5"
    llm_effort: Literal["low", "medium", "high", "xhigh", "max"] = "high"
    llm_max_tokens: int = Field(default=16_000, ge=1_024)
    # Server-side refusal fallback (beta). Disable if your account/platform doesn't support it.
    llm_fallbacks: bool = True

    # Agent loop
    max_iterations: int = Field(default=10, ge=1, le=50)
    save_to_long_term: bool = True

    # Storage
    redis_url: str = "redis://localhost:6379/0"
    session_ttl_seconds: int = 3600
    database_url: str | None = None  # e.g. postgresql+asyncpg://user:pass@host:5432/db
    chromadb_path: Path = Path("./data/chromadb")

    # Tools
    workspace_dir: Path = Path("./data/workspace")
    # Off by default: the child runs as the same OS user as the API (see README, "Security model").
    enable_code_executor: bool = False

    # API
    api_key: SecretStr | None = None  # if set, every request must send X-API-Key
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:8000"])
    rate_limit_per_minute: int = Field(default=20, ge=0)  # 0 disables the limiter
    log_level: str = "INFO"

    @field_validator("api_key", mode="before")
    @classmethod
    def _blank_api_key_means_disabled(cls, value: object) -> object:
        return None if value in ("", None) else value


@lru_cache
def get_settings() -> Settings:
    return Settings()
