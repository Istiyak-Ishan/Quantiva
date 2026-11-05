from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    # ── Data Stores ───────────────────────────────────────────────────────────
    DATABASE_URL: str
    REDIS_URL: str
    QDRANT_URL: str
    S3_ENDPOINT_URL: str
    AWS_ACCESS_KEY_ID: str = "test"
    AWS_SECRET_ACCESS_KEY: str = "test"
    AWS_DEFAULT_REGION: str = "us-east-1"

    # ── Secrets ───────────────────────────────────────────────────────────────
    GOOGLE_API_KEY: str
    SECRET_KEY: str

    # ── Langfuse — LLM Observability ──────────────────────────────────────────
    # Optional: gracefully degrades if not set.  When all three vars are
    # present, the Langfuse callback handler is enabled automatically.
    LANGFUSE_SECRET_KEY: Optional[str] = None
    LANGFUSE_PUBLIC_KEY: Optional[str] = None
    LANGFUSE_HOST: str = "http://langfuse:3000"

    # ── App metadata ──────────────────────────────────────────────────────────
    APP_ENV: str = "development"   # development | staging | production
    APP_VERSION: str = "1.0.0"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
