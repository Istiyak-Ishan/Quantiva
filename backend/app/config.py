import os
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_backend_dir = Path(__file__).resolve().parent.parent
_env_files = [
    _backend_dir / ".env",
    _backend_dir.parent / ".env",
    Path(".env")
]


def _default_quantiva_data_dir() -> Path:
    """Local-first default: %LOCALAPPDATA%/Quantiva on Windows, ~/.quantiva elsewhere."""
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "Quantiva"
    return Path.home() / ".quantiva"


class Settings(BaseSettings):
    # ── Data Stores ───────────────────────────────────────────────────────────
    DATABASE_URL: str = "postgresql+asyncpg://quantiva_user:quantiva_password@localhost:5432/quantiva_db"
    REDIS_URL: str = "redis://localhost:6379/0"
    QDRANT_URL: str = "http://localhost:6333"
    S3_ENDPOINT_URL: str = "http://localhost:9090"
    AWS_ACCESS_KEY_ID: str = "test"
    AWS_SECRET_ACCESS_KEY: str = "test"
    AWS_DEFAULT_REGION: str = "us-east-1"

    # ── Secrets ───────────────────────────────────────────────────────────────
    GOOGLE_API_KEY: str = "your-dev-google-api-key"
    SECRET_KEY: str = "dev-secret-key-quantiva-change-in-prod"

    # ── Langfuse — LLM Observability ──────────────────────────────────────────
    # Optional: gracefully degrades if not set.  When all three vars are
    # present, the Langfuse callback handler is enabled automatically.
    LANGFUSE_SECRET_KEY: str | None = None
    LANGFUSE_PUBLIC_KEY: str | None = None
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"

    # ── App metadata ──────────────────────────────────────────────────────────
    APP_ENV: str = "development"   # development | staging | production
    APP_VERSION: str = "1.0.0"
    CORS_ORIGINS: list[str] = ["http://localhost:3000"]
    ACCESS_TOKEN_MINUTES: int = 30
    REFRESH_TOKEN_DAYS: int = 7
    GEMINI_CHAT_MODEL: str = "gemini-3.5-flash-lite"
    EMBEDDING_MODEL: str = "gemini-embedding-001"
    EMBEDDING_DIMENSIONS: int = 768
    QDRANT_COLLECTION: str = "quantiva_v2"
    GOOGLE_TIMEOUT_MS: int = 60000

    # ── Local-first Ollama provider ────────────────────────────────────────────
    OLLAMA_BASE_URL: str = "http://127.0.0.1:11434"
    OLLAMA_CHAT_MODEL: str = "llama3.2"
    OLLAMA_EMBEDDING_MODEL: str = "nomic-embed-text"
    OLLAMA_TIMEOUT_SECONDS: float = 120.0
    OLLAMA_CONNECT_TIMEOUT_SECONDS: float = 5.0

    # ── Cloud API keys for OpenAI-compatible and Anthropic adapters ───────────
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    COHERE_API_KEY: str = ""
    JINA_API_KEY: str = ""

    # ── Local-first desktop data directory (all runtime state lives here) ─────
    QUANTIVA_DATA_DIR: Path = _default_quantiva_data_dir()

    # ── In-process local model providers (sentence-transformers) ─────────────
    LOCAL_EMBEDDING_MODEL: str = "intfloat/multilingual-e5-small"
    LOCAL_RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    MODEL_DEVICE: str = "auto"  # auto | cpu | cuda | cuda:N | mps
    HF_OFFLINE: bool = False

    MAX_UPLOAD_BYTES: int = 25 * 1024 * 1024
    # Ingestion watchdog: a row may not stay "processing" forever when no
    # worker exists or the task dies without writing a terminal status.
    INGESTION_STALE_SECONDS: int = 600
    INGESTION_WORKER_GRACE_SECONDS: int = 15
    WORKER_PING_TIMEOUT_SECONDS: float = 1.5
    # Optional ClamAV INSTREAM scanner. Uploads fail closed whenever a host is set.
    CLAMAV_HOST: str | None = None
    CLAMAV_PORT: int = 3310
    CLAMAV_TIMEOUT_SECONDS: float = 10.0
    PII_ENGINE: str = "patterns"  # patterns | presidio (requires provisioned spaCy model)
    SPACY_MODEL: str = "en_core_web_sm"
    S3_PUBLIC_ENDPOINT_URL: str | None = None
    METRICS_TOKEN: str | None = None
    # Optional operator-configured prices. Unknown prices stay null, never fake zero.
    LLM_INPUT_USD_PER_MILLION: float | None = None
    LLM_OUTPUT_USD_PER_MILLION: float | None = None

    @property
    def embedding_collection(self) -> str:
        return f"{self.QDRANT_COLLECTION}_{self.EMBEDDING_MODEL.replace('-', '_')}_{self.EMBEDDING_DIMENSIONS}"

    @field_validator("GEMINI_CHAT_MODEL", "EMBEDDING_MODEL")
    @classmethod
    def reject_retired_models(cls, value: str) -> str:
        if not value.strip() or value in {"gemini-1.5-flash", "gemini-1.5-pro", "text-embedding-004"}:
            raise ValueError("Configure a supported, non-retired model")
        return value

    @field_validator("EMBEDDING_DIMENSIONS")
    @classmethod
    def embedding_dimensions(cls, value: int) -> int:
        if not 128 <= value <= 3072:
            raise ValueError("Embedding dimensions must be between 128 and 3072")
        return value

    @field_validator("DATABASE_URL")
    @classmethod
    def async_database_url(cls, value: str) -> str:
        for scheme in ("postgres://", "postgresql://"):
            if value.startswith(scheme):
                return value.replace(scheme, "postgresql+asyncpg://", 1)
        return value

    @field_validator("CORS_ORIGINS")
    @classmethod
    def exact_origins(cls, origins: list[str]) -> list[str]:
        for origin in origins:
            parsed = urlsplit(origin)
            if "*" in origin or parsed.scheme not in {"http", "https"} or not parsed.netloc or parsed.path or parsed.query or parsed.fragment:
                raise ValueError("CORS_ORIGINS must contain exact HTTP(S) origins without paths or wildcards")
        return origins

    @model_validator(mode="after")
    def production_security(self):
        if self.APP_ENV in {"production", "staging"}:
            if len(self.SECRET_KEY) < 32 or any(word in self.SECRET_KEY.lower() for word in ("dev-secret", "change-in-prod", "your-dev", "replace-with", "changeme")):
                raise ValueError("A strong, non-default SECRET_KEY is required outside development")
            if not self.CORS_ORIGINS or any(not origin.startswith("https://") for origin in self.CORS_ORIGINS):
                raise ValueError("Explicit HTTPS frontend origins are required outside development")
        return self

    model_config = SettingsConfigDict(
        env_file=[str(p) for p in _env_files if p.exists()] or ".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        hide_input_in_errors=True,
        extra="ignore",
    )


settings = Settings()
