import asyncio
import json
import sqlite3
import uuid
import pytest
import pytest_asyncio
from datetime import timedelta
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy import select, event
from unittest.mock import AsyncMock, MagicMock, patch

# Teach SQLite how to render ARRAY and UUID types
sqlite3.register_adapter(list, lambda l: json.dumps([str(x) for x in l]))
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.types import ARRAY as BaseARRAY
from sqlalchemy.dialects.postgresql import ARRAY as PG_ARRAY, UUID as PG_UUID

@compiles(BaseARRAY, "sqlite")
def compile_base_array_sqlite(type_, compiler, **kw):
    return "JSON"

@compiles(PG_ARRAY, "sqlite")
def compile_pg_array_sqlite(type_, compiler, **kw):
    return "JSON"

@compiles(PG_UUID, "sqlite")
def compile_uuid_sqlite(type_, compiler, **kw):
    return "TEXT"

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"

from app.db.database import Base
from app.core.dependencies import get_db
from app.main import app

test_engine = create_async_engine(TEST_DB_URL, echo=False)

@event.listens_for(test_engine.sync_engine, "connect")
def enable_sqlite_foreign_keys(connection, _):
    cursor = connection.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()

TestSessionLocal = async_sessionmaker(test_engine, expire_on_commit=False)

@pytest_asyncio.fixture
async def create_tables():
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)

@pytest_asyncio.fixture
async def db_session(create_tables):
    async with TestSessionLocal() as session:
        yield session
        await session.rollback()

@pytest_asyncio.fixture
async def client(db_session):
    async def _override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = _override_get_db

    with (
        patch("app.services.retrieval.vector_store.AsyncQdrantClient", MagicMock()),
        patch("app.services.memory.semantic_cache.redis", MagicMock()),
        patch("app.core.rate_limiter.rate_limiter.check", new_callable=AsyncMock, return_value={"allowed": True, "remaining": 10, "reset_in": 0}),
    ):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac

    app.dependency_overrides.clear()

@pytest_asyncio.fixture
async def auth_token(client):
    resp = await client.post("/api/v1/auth/register", json={
        "email": "test@quantiva.ai",
        "password": "TestPass123!",
    })
    if resp.status_code == 400:
        resp = await client.post("/api/v1/auth/login", json={
            "email": "test@quantiva.ai",
            "password": "TestPass123!",
        })
    assert resp.status_code in (200, 201), f"Auth failed: {resp.text}"
    return resp.json()["access_token"]

@pytest.fixture
def auth_headers(auth_token):
    return {"Authorization": f"Bearer {auth_token}"}


@pytest_asyncio.fixture
async def tenants(client, db_session):
    from app.models.user import User
    from app.models.document import Document
    from app.models.eval_result import EvalRun
    from app.models.message import Message
    headers = []
    for name in ("alice", "bob"):
        response = await client.post("/api/v1/auth/register", json={"email": f"{name}@example.com", "password": "SecurePass123!"})
        assert response.status_code == 200
        headers.append({"Authorization": f"Bearer {response.json()['access_token']}"})
    client.cookies.clear()
    alice = await db_session.scalar(select(User).where(User.email == "alice@example.com"))
    doc = Document(user_id=alice.id, filename="private.pdf", file_type="application/pdf", file_size=20, storage_path="private.pdf", status="ready")
    db_session.add(doc)
    await db_session.flush()
    run = EvalRun(doc_id=doc.id, user_id=alice.id, status="completed")
    db_session.add(run)
    await db_session.commit()
    response = await client.post("/api/v1/chat/sessions", headers=headers[0], json={"document_ids": [str(doc.id)]})
    assert response.status_code == 200, response.text
    session_id = response.json()["session_id"]
    db_session.add(Message(session_id=uuid.UUID(session_id), role="assistant", content="ALICE PRIVATE HISTORY"))
    await db_session.commit()
    return headers[0], headers[1], doc, run, session_id
