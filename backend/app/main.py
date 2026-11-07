"""
backend/app/main.py
────────────────────────────────────────────────────────────────────────────────
FastAPI application entry point for Quantiva.

Wires together:
  • CORS middleware
  • Prometheus instrumentation middleware  (records every HTTP request)
  • All API v1 routers
  • /metrics endpoint (Prometheus scrape)
  • /health endpoint (deep service health check)
  • Lifespan hooks for clean startup/shutdown
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.db.database import engine
from app.config import settings
from app.api.v1.auth import router as auth_router
from app.api.v1.documents import router as documents_router
from app.api.v1.chat import router as chat_router
from app.api.v1.eval import router as eval_router
from app.api.v1.metrics_endpoint import router as metrics_router
from app.core.middleware import PrometheusMiddleware

import redis.asyncio as redis
from qdrant_client import AsyncQdrantClient

# ── API router ────────────────────────────────────────────────────────────────

api_v1_router = APIRouter(prefix="/api/v1")
api_v1_router.include_router(auth_router,      tags=["auth"])
api_v1_router.include_router(documents_router, tags=["documents"])
api_v1_router.include_router(chat_router,      tags=["chat"])
api_v1_router.include_router(eval_router,      tags=["eval"])

# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.
    Startup:  nothing to do (connections are lazily opened per-request).
    Shutdown: dispose the SQLAlchemy engine connection pool.
    """
    yield
    # Gracefully drain the SQLAlchemy connection pool
    await engine.dispose()

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Quantiva AI API",
    description=(
        "Enterprise AI document agent platform.\n\n"
        "Upload PDFs, CSVs, Excel sheets, and Word docs, then ask questions "
        "and get cited, streaming answers powered by Gemini 1.5 Flash with "
        "hybrid RAG (vector + BM25 + reranking)."
    ),
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

# ── Middleware ─────────────────────────────────────────────────────────────────

# 1. Prometheus — MUST be added before CORS so it sees all requests
app.add_middleware(PrometheusMiddleware)

# 2. CORS — allow frontend origin; tighten in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://*.railway.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ────────────────────────────────────────────────────────────────────

app.include_router(api_v1_router)
app.include_router(metrics_router)   # GET /metrics — Prometheus scrape endpoint

# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health", tags=["Health"], summary="Deep health check")
async def health_check():
    """
    Deep health check — verifies connectivity to every backing service.

    Returns 200 with `status: ok` if all services are reachable.
    Returns 200 with `status: degraded` if any service is down
    (we still return 200 so Railway / load balancers keep routing traffic
    and the issue is visible in monitoring rather than causing a full restart).
    """
    status: dict = {"status": "ok", "version": settings.APP_VERSION, "services": {}}

    # 1. Postgres
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        status["services"]["postgres"] = "up"
    except Exception as exc:
        status["services"]["postgres"] = f"down: {exc}"
        status["status"] = "degraded"

    # 2. Redis
    try:
        r = redis.from_url(settings.REDIS_URL)
        await r.ping()
        await r.aclose()
        status["services"]["redis"] = "up"
    except Exception as exc:
        status["services"]["redis"] = f"down: {exc}"
        status["status"] = "degraded"

    # 3. Qdrant
    try:
        client = AsyncQdrantClient(url=settings.QDRANT_URL)
        await client.get_collections()
        status["services"]["qdrant"] = "up"
    except Exception as exc:
        status["services"]["qdrant"] = f"down: {exc}"
        status["status"] = "degraded"

    return status
