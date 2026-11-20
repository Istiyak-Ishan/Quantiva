"""
backend/app/api/routes/__init__.py
────────────────────────────────────────────────────────────────────────────────
Quantiva desktop FastAPI routers.
"""
from app.api.routes.models import router as models_router
from app.api.routes.settings import router as settings_router
from app.api.routes.system import router as system_router

__all__ = ["models_router", "settings_router", "system_router"]
