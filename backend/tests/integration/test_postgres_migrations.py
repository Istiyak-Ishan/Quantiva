"""Opt-in real PostgreSQL upgrade + drift check, never the configured user DB."""
import asyncio
import os
from pathlib import Path

from alembic import command
from alembic.config import Config
import pytest
from sqlalchemy.engine import make_url

from app.config import settings


async def test_clean_postgres_upgrade_and_orm_drift(monkeypatch):
    url = os.environ.get("TEST_POSTGRES_URL")
    if not url:
        pytest.skip("TEST_POSTGRES_URL not set; clean PostgreSQL acceptance is not established")
    parsed = make_url(url)
    if not parsed.database or not parsed.database.startswith("quantiva_test_"):
        pytest.fail("Refusing to migrate a database not explicitly named quantiva_test_*")
    monkeypatch.setattr(settings, "DATABASE_URL", url)
    backend = Path(__file__).resolve().parents[2]
    config = Config(str(backend / "alembic.ini"))
    config.set_main_option("script_location", str(backend / "app/db/migrations"))
    await asyncio.to_thread(command.upgrade, config, "head")
    await asyncio.to_thread(command.upgrade, config, "head")  # Idempotent at head.
    await asyncio.to_thread(command.check, config)
