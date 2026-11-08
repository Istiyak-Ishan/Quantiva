import asyncio
import pytest
import pytest_asyncio
from datetime import timedelta
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from unittest.mock import AsyncMock, MagicMock, patch

# Teach SQLite how to render ARRAY and UUID types
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
TestSessionLocal = async_sessionmaker(test_engine, expire_on_commit=False)

@pytest_asyncio.fixture(scope="session")
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
