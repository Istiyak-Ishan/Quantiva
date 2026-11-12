"""
app/core/application_context.py
──────────────────────────────────────────────────────────────────────────────
Central application context managing runtime singletons and lifecycle for
the Quantiva local-first desktop application.

Key Invariants:
1. Singleton Provider Registry: A single, thread-safe ProviderRegistry instance
   manages LLM, Embedding, and Reranker lifecycles across the application.
2. Singleton Embedded Vector Client: Invariant 5 from AGENTS.md mandates that
   embedded vector stores (QdrantClient or LanceDB) must NEVER be instantiated
   per-request to prevent Windows SQLite / file-lock collisions. The embedded
   client lifecycle is managed exclusively here.
3. Clean Lifecycle Management: Provides thread-safe startup, shutdown, and
   resource cleanup for background executors and model runners.
"""
from __future__ import annotations

import logging
import threading
from typing import Any

from app.config import Settings, settings
from app.providers.interfaces.embedding import EmbeddingProvider
from app.providers.interfaces.llm import LLMProvider
from app.providers.interfaces.reranker import RerankerProvider
from app.providers.registry import ProviderRegistry

logger = logging.getLogger(__name__)

_global_context_lock = threading.RLock()
_global_context: ApplicationContext | None = None


class ApplicationContext:
    """Runtime lifecycle container holding application-wide singletons."""

    def __init__(self, settings_obj: Settings | None = None) -> None:
        self._settings = settings_obj or settings
        self._lock = threading.RLock()
        self._is_started = False
        self._is_closed = False

        # Singletons
        self._provider_registry = ProviderRegistry(self._settings)
        self._vector_client: Any | None = None

    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def provider_registry(self) -> ProviderRegistry:
        return self._provider_registry

    # ── Provider Convenience Accessors ────────────────────────────────────────

    def get_llm_provider(self) -> LLMProvider:
        """Return the active LLMProvider from the registry."""
        return self._provider_registry.get_llm_provider()

    def get_embedding_provider(self) -> EmbeddingProvider:
        """Return the active EmbeddingProvider from the registry."""
        return self._provider_registry.get_embedding_provider()

    def get_reranker_provider(self) -> RerankerProvider:
        """Return the active RerankerProvider from the registry."""
        return self._provider_registry.get_reranker_provider()

    # ── Vector Store Lifecycle (Invariant 5) ──────────────────────────────────

    def get_vector_client(self) -> Any | None:
        """Return the singleton embedded vector store client."""
        with self._lock:
            return self._vector_client

    def set_vector_client(self, client: Any) -> None:
        """Register the singleton embedded vector store client."""
        with self._lock:
            self._vector_client = client
            logger.info("Embedded vector client registered with ApplicationContext.")

    def close_vector_client(self) -> None:
        """Close the vector client if it exposes a close() method."""
        with self._lock:
            client = self._vector_client
            self._vector_client = None
            if client is not None:
                close_fn = getattr(client, "close", None)
                if callable(close_fn):
                    try:
                        close_fn()
                        logger.info("Embedded vector client closed safely.")
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Error closing embedded vector client: %s", exc)

    # ── Lifecycle Hooks ───────────────────────────────────────────────────────

    def startup(self) -> None:
        """Execute application startup hooks."""
        with self._lock:
            if self._is_started:
                return
            self._is_started = True
            self._is_closed = False
            logger.info("ApplicationContext started.")

    def shutdown(self) -> None:
        """Execute application shutdown hooks and release locked resources."""
        with self._lock:
            if self._is_closed:
                return
            self.close_vector_client()
            self._provider_registry.clear_cache()
            self._is_closed = True
            self._is_started = False
            logger.info("ApplicationContext shut down.")


# ── Global Accessors ──────────────────────────────────────────────────────────

def get_application_context() -> ApplicationContext:
    """Return the global ApplicationContext singleton."""
    global _global_context
    if _global_context is None:
        with _global_context_lock:
            if _global_context is None:
                _global_context = ApplicationContext()
    return _global_context


def get_provider_registry() -> ProviderRegistry:
    """Convenience helper returning the global ProviderRegistry."""
    return get_application_context().provider_registry


def reset_application_context() -> None:
    """Reset the global ApplicationContext singleton (primarily for testing)."""
    global _global_context
    with _global_context_lock:
        if _global_context is not None:
            _global_context.shutdown()
        _global_context = None

