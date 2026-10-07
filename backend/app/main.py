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
import redis.asyncio as redis
from qdrant_client import AsyncQdrantClient

# Placeholder router for v1 endpoints
api_v1_router = APIRouter(prefix="/api/v1")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup actions
    # This is a great place to run schema migrations or seed initial data
    yield
    # Shutdown actions: Clean up resource connections gracefully
    await engine.dispose()

app = FastAPI(
    title="Quantiva AI API",
    description="Backend for Quantiva AI document agent platform",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for allowing frontend to communicate with API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Should be restricted in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_v1_router.include_router(auth_router, tags=["auth"])
api_v1_router.include_router(documents_router, tags=["documents"])
api_v1_router.include_router(chat_router, tags=["chat"])
api_v1_router.include_router(eval_router, tags=["eval"])
app.include_router(api_v1_router)

@app.get("/health", tags=["Health"])
async def health_check():
    """
    Health check endpoint to verify API and backing service statuses.
    """
    status = {"status": "ok", "services": {}}
    
    # 1. Check Postgres
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        status["services"]["postgres"] = "up"
    except Exception as e:
        status["services"]["postgres"] = "down"
        status["status"] = "degraded"
        
    # 2. Check Redis
    try:
        r = redis.from_url(settings.REDIS_URL)
        await r.ping()
        await r.close()
        status["services"]["redis"] = "up"
    except Exception as e:
        status["services"]["redis"] = "down"
        status["status"] = "degraded"
        
    # 3. Check Qdrant
    try:
        client = AsyncQdrantClient(url=settings.QDRANT_URL)
        # Verify connection by attempting to list collections
        await client.get_collections()
        status["services"]["qdrant"] = "up"
    except Exception as e:
        status["services"]["qdrant"] = "down"
        status["status"] = "degraded"
        
    return status
