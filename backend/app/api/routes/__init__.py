"""
backend/app/api/routes/__init__.py
────────────────────────────────────────────────────────────────────────────────
Quantiva desktop FastAPI routers.
"""
from app.api.routes.chat import router as chat_router
from app.api.routes.documents import router as documents_router
from app.api.routes.evaluation import router as evaluation_router
from app.api.routes.knowledge_bases import router as knowledge_bases_router
from app.api.routes.models import router as models_router
from app.api.routes.settings import router as settings_router
from app.api.routes.system import router as system_router

__all__ = [
    "chat_router",
    "documents_router",
    "evaluation_router",
    "knowledge_bases_router",
    "models_router",
    "settings_router",
    "system_router",
]
