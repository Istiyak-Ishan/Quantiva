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
import asyncio
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
from app.api.v1.retrieval import router as retrieval_router
from app.api.v1.charts import router as charts_router
from app.core.middleware import PrometheusMiddleware, SecurityHeadersMiddleware
from app.core.application_context import get_application_context
from app.storage.vector_store import LocalVectorStore, VectorStoreManager
from app.services.retrieval.vector_store import vector_store
from app.core.rate_limiter import rate_limiter
from app.core.storage import s3_client, BUCKET_NAME
from app.services.memory.session_memory import session_memory
from app.services.memory.semantic_cache import semantic_cache
from fastapi.responses import JSONResponse


# ── API router ────────────────────────────────────────────────────────────────

api_v1_router = APIRouter(prefix="/api/v1")
api_v1_router.include_router(auth_router,      tags=["auth"])
api_v1_router.include_router(documents_router, tags=["documents"])
api_v1_router.include_router(chat_router,      tags=["chat"])
api_v1_router.include_router(eval_router,      tags=["eval"])
api_v1_router.include_router(retrieval_router)
api_v1_router.include_router(charts_router)

# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.
    Startup: acquire the one embedded-Qdrant directory lock for this process.
    Shutdown: close embedded Qdrant and all legacy migration-era clients.
    """
    application_context = get_application_context()
    vector_manager = VectorStoreManager(data_dir=settings.QUANTIVA_DATA_DIR)
    application_context.startup()
    try:
        vector_manager.start()
        application_context.set_vector_client(vector_manager)
        app.state.vector_store_manager = vector_manager
        app.state.local_vector_store = LocalVectorStore(vector_manager)
        yield
    finally:
        # ApplicationContext owns the manager and invokes its idempotent close.
        application_context.shutdown()
        vector_manager.close()
        # Gracefully drain migration-era cloud service clients.
        await engine.dispose()
        await vector_store.client.close()
        await rate_limiter.client.aclose()
        await session_memory.client.aclose()
        await semantic_cache.client.aclose()
        s3_client.close()

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Quantiva AI API",
    description=(
        "Enterprise AI document agent platform.\n\n"
        "Upload PDFs, CSVs, Excel sheets, and Word docs, then ask questions "
        "and get cited, streaming answers powered by a configurable Gemini model with "
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
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Security headers — outermost so responses from every route are hardened.
app.add_middleware(SecurityHeadersMiddleware)

# ── Routes ────────────────────────────────────────────────────────────────────

app.include_router(api_v1_router)
app.include_router(metrics_router)   # GET /metrics — Prometheus scrape endpoint

# ── Health check ──────────────────────────────────────────────────────────────

@app.get("/health", tags=["Health"], summary="Process liveness")
async def health_check():
    return {"status": "ok", "version": settings.APP_VERSION}


@app.get("/ready", tags=["Health"], summary="Dependency readiness")
async def readiness_check():
    async def postgres():
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))

    async def check(operation):
        try:
            await asyncio.wait_for(operation(), timeout=4)
            return "up"
        except Exception:
            return "down"  # Never disclose driver exceptions, URLs or credentials.

    operations = {
        "postgres": postgres,
        "redis": rate_limiter.client.ping,
        "qdrant": vector_store.client.get_collections,
        "s3": lambda: asyncio.to_thread(s3_client.head_bucket, Bucket=BUCKET_NAME),
    }
    results = await asyncio.gather(*(check(op) for op in operations.values()))
    services = dict(zip(operations, results))
    ready = all(value == "up" for value in results)
    return JSONResponse(status_code=200 if ready else 503, content={"status": "ok" if ready else "degraded", "services": services})
