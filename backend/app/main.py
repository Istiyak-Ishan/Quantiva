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
import os
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
from app.api.routes.models import router as models_router
from app.api.routes.settings import router as settings_router
from app.api.routes.system import router as system_router
from app.api.routes.knowledge_bases import router as desktop_kb_router
from app.api.routes.documents import router as desktop_documents_router
from app.api.routes.chat import router as desktop_chat_router
from app.core.middleware import PrometheusMiddleware, SecurityHeadersMiddleware
from app.core.application_context import get_application_context
from app.jobs.indexing import KnowledgeBaseReindexJobHandler
from app.jobs.ingestion import DocumentIngestionJobHandler
from app.jobs.manager import LocalJobManager, configure_default_job_manager
from app.providers.registry import EmbeddingConfiguration as ProviderEmbeddingConfiguration
from app.storage.database import (
    SessionLocal as LocalSessionLocal,
    engine as local_database_engine,
    initialize_database as initialize_local_database,
)
from app.storage.models import (
    EmbeddingConfiguration as StoredEmbeddingConfiguration,
    KnowledgeBase as StoredKnowledgeBase,
)
from app.storage.sparse_store import SQLiteSparseIndex
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
api_v1_router.include_router(desktop_documents_router)
api_v1_router.include_router(documents_router, tags=["documents"])
api_v1_router.include_router(chat_router,      tags=["chat"])
api_v1_router.include_router(eval_router,      tags=["eval"])
api_v1_router.include_router(retrieval_router)
api_v1_router.include_router(charts_router)
api_v1_router.include_router(desktop_kb_router)
api_v1_router.include_router(desktop_chat_router)
api_v1_router.include_router(models_router, prefix="/models", tags=["Models"])
api_v1_router.include_router(settings_router, prefix="/settings", tags=["Settings"])
api_v1_router.include_router(system_router, prefix="/system", tags=["System"])

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
    local_job_manager = None
    application_context.startup()
    try:
        initialize_local_database(local_database_engine)
        vector_manager.start()
        application_context.set_vector_client(vector_manager)
        local_vector_store = LocalVectorStore(vector_manager)
        local_sparse_store = SQLiteSparseIndex(local_database_engine)

        def provider_from_config(config: StoredEmbeddingConfiguration):
            return application_context.provider_registry.get_knowledge_base_embedding_provider(
                ProviderEmbeddingConfiguration(
                    provider=config.provider_type,
                    model=config.model_name,
                    dimensions=config.dimensions,
                    base_url=config.base_url,
                    batch_size=config.batch_size,
                )
            )

        def provider_for_knowledge_base(knowledge_base_id: str):
            with LocalSessionLocal() as session:
                knowledge_base = session.get(StoredKnowledgeBase, knowledge_base_id)
                if knowledge_base is None or not knowledge_base.embedding_config_id:
                    raise KeyError(
                        f"Knowledge base has no embedding configuration: {knowledge_base_id}"
                    )
                config = session.get(
                    StoredEmbeddingConfiguration,
                    knowledge_base.embedding_config_id,
                )
                if config is None:
                    raise KeyError(knowledge_base.embedding_config_id)
                session.expunge(config)
            return provider_from_config(config)

        local_job_manager = LocalJobManager(
            LocalSessionLocal,
            worker_count=2,
            vector_store=local_vector_store,
            sparse_store=local_sparse_store,
        )
        ingestion_handler = DocumentIngestionJobHandler(
            LocalSessionLocal,
            embedding_provider_resolver=provider_for_knowledge_base,
            vector_store=local_vector_store,
            sparse_store=local_sparse_store,
            data_dir=settings.QUANTIVA_DATA_DIR,
        )
        reindex_handler = KnowledgeBaseReindexJobHandler(
            LocalSessionLocal,
            embedding_provider_resolver=provider_from_config,
            vector_store=local_vector_store,
        )
        local_job_manager.register_handler(ingestion_handler.JOB_TYPE, ingestion_handler)
        local_job_manager.register_handler(reindex_handler.JOB_TYPE, reindex_handler)
        configure_default_job_manager(local_job_manager)
        await local_job_manager.start()

        app.state.vector_store_manager = vector_manager
        app.state.local_vector_store = local_vector_store
        app.state.local_sparse_store = local_sparse_store
        app.state.local_job_manager = local_job_manager
        yield
    finally:
        if local_job_manager is not None:
            await local_job_manager.stop(timeout_seconds=5.0)
        # ApplicationContext owns the manager and invokes its idempotent close.
        application_context.shutdown()
        vector_manager.close()
        local_database_engine.dispose()
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

# 2. Desktop security — when QUANTIVA_IPC_TOKEN is set (desktop mode), add
#    IPCTokenAuth and LoopbackGuard middleware BEFORE CORS so the execution
#    order is: LoopbackGuard → IPCTokenAuth → CORS → route.
#    OPTIONS preflight is explicitly exempt inside IPCTokenAuth so CORS works.
if os.environ.get("QUANTIVA_IPC_TOKEN"):
    from app.security.local_api import (  # noqa: PLC0415
        DESKTOP_CORS_ORIGINS,
        configure_desktop_security,
    )
    # Use desktop origins when running under Tauri.
    _cors_origins = DESKTOP_CORS_ORIGINS
    configure_desktop_security(app)
else:
    _cors_origins = settings.CORS_ORIGINS

# 3. CORS — allow frontend origin; tighten in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. Security headers — outermost so responses from every route are hardened.
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
