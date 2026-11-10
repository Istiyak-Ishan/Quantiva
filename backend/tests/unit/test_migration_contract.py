"""Offline DDL checks; these do not replace clean PostgreSQL upgrade/drift tests."""
from io import StringIO
from pathlib import Path

from alembic import command
from alembic.config import Config


def test_fresh_postgres_ddl_has_message_cost():
    backend = Path(__file__).resolve().parents[2]
    output = StringIO()
    config = Config(str(backend / "alembic.ini"), output_buffer=output)
    config.set_main_option("script_location", str(backend / "app/db/migrations"))
    command.upgrade(config, "head", sql=True)
    ddl = output.getvalue()
    assert "CREATE TABLE messages" in ddl
    assert "ALTER TABLE messages ADD COLUMN cost_usd FLOAT DEFAULT '0.0'" in ddl


def test_backend_build_context_excludes_secrets():
    backend = Path(__file__).resolve().parents[2]
    ignore = (backend / ".dockerignore").read_text().splitlines()
    assert ".env" in ignore
    assert ".env.*" in ignore
    assert ".venv" in ignore
