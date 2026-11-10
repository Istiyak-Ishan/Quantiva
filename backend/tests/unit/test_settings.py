import pytest
from pydantic import ValidationError

from app.config import Settings


def test_production_rejects_default_secret():
    with pytest.raises(ValidationError, match="SECRET_KEY"):
        Settings(_env_file=None, APP_ENV="production", SECRET_KEY="dev-secret-key-quantiva-change-in-prod", CORS_ORIGINS=["https://quantiva.example"])


def test_production_accepts_strong_secret_and_explicit_https_origin():
    config = Settings(_env_file=None, APP_ENV="production", SECRET_KEY="a1b2c3d4" * 8, CORS_ORIGINS=["https://quantiva.example"])
    assert config.APP_ENV == "production"


def test_wildcard_cors_rejected():
    with pytest.raises(ValidationError, match="exact"):
        Settings(_env_file=None, CORS_ORIGINS=["https://*.railway.app"])


def test_railway_postgres_url_uses_asyncpg():
    config = Settings(_env_file=None, DATABASE_URL="postgresql://user:pass@host/db")
    assert config.DATABASE_URL == "postgresql+asyncpg://user:pass@host/db"
