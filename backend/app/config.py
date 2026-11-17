import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from platformdirs import user_data_dir
from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_backend_dir = Path(__file__).resolve().parent.parent
_env_files = [
    _backend_dir / ".env",
    _backend_dir.parent / ".env",
    Path(".env")
]


@dataclass(frozen=True, slots=True)
class LocalDataPaths:
    """Canonical paths rooted under the configured Quantiva data directory."""

    root: Path

    @property
    def database_dir(self) -> Path:
        return self.root / "database"

    @property
    def database_file(self) -> Path:
        return self.database_dir / "quantiva.db"

    @property
    def original_documents(self) -> Path:
        return self.root / "documents" / "original"

    @property
    def processed_documents(self) -> Path:
        return self.root / "documents" / "processed"

    @property
    def attachments(self) -> Path:
        return self.root / "documents" / "attachments"

    @property
    def vector_store(self) -> Path:
        return self.root / "vector_store"

    @property
    def embedding_models(self) -> Path:
        return self.root / "models" / "embeddings"

    @property
    def reranker_models(self) -> Path:
        return self.root / "models" / "rerankers"

    @property
    def exports(self) -> Path:
        return self.root / "exports"

    @property
    def backups(self) -> Path:
        return self.root / "backups"

    @property
    def logs(self) -> Path:
        return self.root / "logs"

    @property
    def sandbox_scratch(self) -> Path:
        return self.root / "sandbox_scratch"

    @property
    def required_directories(self) -> tuple[Path, ...]:
        return (
            self.database_dir,
            self.original_documents,
            self.processed_documents,
            self.attachments,
            self.vector_store,
            self.embedding_models,
            self.reranker_models,
            self.exports,
            self.backups,
            self.logs,
            self.sandbox_scratch,
        )


def bootstrap_config_path() -> Path:
    """Return the fixed bootstrap file used before the main DB is available."""
    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            return Path(local_app_data).expanduser() / "Quantiva" / "bootstrap.json"
    return Path(user_data_dir("Quantiva")).expanduser() / "bootstrap.json"


def _bootstrap_data_dir(path: Path | None = None) -> Path | None:
    config_path = path or bootstrap_config_path()
    if not config_path.is_file():
        return None
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    value = payload.get("data_dir") if isinstance(payload, dict) else None
    if not isinstance(value, str) or not value.strip():
        return None
    configured = Path(value).expanduser()
    if not configured.is_absolute():
        configured = config_path.parent / configured
    return configured.resolve()


def resolve_quantiva_data_dir(
    *,
    bootstrap_path: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Resolve data root with env > bootstrap.json > platform default precedence."""
    env = os.environ if environ is None else environ
    override = env.get("QUANTIVA_DATA_DIR", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    configured = _bootstrap_data_dir(bootstrap_path)
    if configured is not None:
        return configured
    return (bootstrap_path or bootstrap_config_path()).parent.resolve()


def ensure_data_directories(data_dir: Path | str) -> LocalDataPaths:
    """Create and validate Quantiva's complete local directory structure."""
    paths = LocalDataPaths(Path(data_dir).expanduser().resolve())
    paths.root.mkdir(parents=True, exist_ok=True)
    for directory in paths.required_directories:
        directory.mkdir(parents=True, exist_ok=True)
        if not directory.is_dir():
            raise OSError(f"Required Quantiva path is not a directory: {directory}")
    paths.database_file.touch(exist_ok=True)

    # Validate write access without leaving temporary artifacts behind.
    probe = paths.root / ".quantiva-write-test"
    try:
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        raise OSError(f"Quantiva data directory is not writable: {paths.root}") from exc
    return paths


class LocalDataDirectoryManager:
    """Resolve, create, and validate the bootstrap-selected local data root."""

    def __init__(
        self,
        *,
        bootstrap_path: Path | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> None:
        self.bootstrap_path = bootstrap_path or bootstrap_config_path()
        self.environ = dict(os.environ if environ is None else environ)

    def resolve(self) -> Path:
        return resolve_quantiva_data_dir(
            bootstrap_path=self.bootstrap_path,
            environ=self.environ,
        )

    def initialize(self) -> LocalDataPaths:
        return ensure_data_directories(self.resolve())


def _default_quantiva_data_dir() -> Path:
    return resolve_quantiva_data_dir()


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
    # OCR fallback for scanned PDFs. Tesseract is looked up in bundled vendor
    # directories first, then on PATH; absence only adds warning metadata.
    OCR_ENABLED: bool = True
    OCR_TESSERACT_CMD: str | None = None
    OCR_LANGUAGES: str = "eng"
    OCR_DPI: int = 200
    OCR_MIN_TEXT_CHARS: int = 30
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

    @property
    def local_data_paths(self) -> LocalDataPaths:
        return LocalDataPaths(self.QUANTIVA_DATA_DIR)

    @property
    def data_dir(self) -> Path:
        """Lowercase compatibility alias used by desktop bootstrap code."""
        return self.QUANTIVA_DATA_DIR

    @property
    def database_path(self) -> Path:
        return self.local_data_paths.database_file

    @property
    def bootstrap_path(self) -> Path:
        return bootstrap_config_path()

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
        self.QUANTIVA_DATA_DIR = ensure_data_directories(
            self.QUANTIVA_DATA_DIR
        ).root
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
