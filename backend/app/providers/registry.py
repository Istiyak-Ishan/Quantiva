"""
app/providers/registry.py
──────────────────────────────────────────────────────────────────────────────
Central Provider Registry and Factory for Quantiva.

Instantiates, caches, and coordinates active LLM, Embedding, and Reranker
providers from normalized configuration dataclasses.

Key Architectural Invariants:
1. Strict Provider Decoupling: Calling code interacts only with LLMProvider,
   EmbeddingProvider, and RerankerProvider abstract base classes.
2. Independent Model Spaces: LLM, Embedding, and Reranker configurations are
   strictly independent. Changing an LLM configuration never alters or
   invalidates active Embedding or Reranker providers.
3. Knowledge Base Dimension Locking: Provides per-KnowledgeBase embedding
   provider lookup based on the knowledge base's stored EmbeddingConfiguration
   or IndexMetadata, ensuring dimension mismatch errors are caught early.
"""
from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from app.config import settings
from app.providers.interfaces.embedding import EmbeddingProvider
from app.providers.interfaces.llm import LLMProvider
from app.providers.interfaces.reranker import RerankerProvider
from app.providers.interfaces.types import ProviderType

logger = logging.getLogger(__name__)


# ── Configuration Dataclasses ─────────────────────────────────────────────────

@dataclass
class LLMConfig:
    """Normalized configuration for an LLM provider."""

    provider: ProviderType | str
    model: str
    api_key: str | None = None
    base_url: str | None = None
    timeout_seconds: float = 120.0
    temperature: float = 0.2
    max_tokens: int = 4096
    extra_headers: dict[str, str] = field(default_factory=dict)
    extra_body: dict[str, Any] = field(default_factory=dict)

    def normalized_provider(self) -> str:
        if isinstance(self.provider, ProviderType):
            return self.provider.value
        return str(self.provider).strip().lower()


@dataclass
class EmbeddingConfig:
    """Normalized configuration for an embedding provider."""

    provider: ProviderType | str
    model: str
    dimensions: int | None = None
    api_key: str | None = None
    base_url: str | None = None
    batch_size: int = 32
    device: str | None = None
    normalize_embeddings: bool = True
    use_e5_prefixes: bool | None = None
    cache_dir: str | Path | None = None

    def normalized_provider(self) -> str:
        if isinstance(self.provider, ProviderType):
            return self.provider.value
        return str(self.provider).strip().lower()


@dataclass
class RerankerConfig:
    """Normalized configuration for a reranker provider."""

    provider: ProviderType | str = ProviderType.CROSS_ENCODER
    model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    enabled: bool = True
    api_key: str | None = None
    base_url: str | None = None
    top_k: int = 5
    device: str | None = None
    batch_size: int = 32
    max_length: int = 512
    cache_dir: str | Path | None = None

    def normalized_provider(self) -> str:
        if isinstance(self.provider, ProviderType):
            return self.provider.value
        return str(self.provider).strip().lower()


@dataclass
class EmbeddingConfiguration:
    """Per-KnowledgeBase embedding configuration stored alongside indexed documents."""

    provider: ProviderType | str
    model: str
    dimensions: int
    base_url: str | None = None
    api_key: str | None = None
    device: str | None = None
    batch_size: int = 32
    cache_dir: str | Path | None = None

    def to_embedding_config(self) -> EmbeddingConfig:
        return EmbeddingConfig(
            provider=self.provider,
            model=self.model,
            dimensions=self.dimensions,
            base_url=self.base_url,
            api_key=self.api_key,
            device=self.device,
            batch_size=self.batch_size,
            cache_dir=self.cache_dir,
        )


@dataclass
class IndexMetadata:
    """KnowledgeBase index metadata locking model space and dimensions."""

    knowledge_base_id: str
    embedding_provider: str
    embedding_model_name: str
    embedding_dimensions: int
    collection_name: str | None = None
    created_at: str | None = None
    distance_metric: str = "cosine"

    def to_embedding_config(self) -> EmbeddingConfig:
        return EmbeddingConfig(
            provider=self.embedding_provider,
            model=self.embedding_model_name,
            dimensions=self.embedding_dimensions,
        )


# ── Provider Registry ─────────────────────────────────────────────────────────

class ProviderRegistry:
    """Central factory and cache for LLM, Embedding, and Reranker providers."""

    def __init__(self, settings_obj: Any = None) -> None:
        self._settings = settings_obj or settings
        self._lock = threading.RLock()

        # Active configurations
        self._llm_config: LLMConfig | None = None
        self._embedding_config: EmbeddingConfig | None = None
        self._reranker_config: RerankerConfig | None = None

        # Cached active singleton instances
        self._active_llm: LLMProvider | None = None
        self._active_embedding: EmbeddingProvider | None = None
        self._active_reranker: RerankerProvider | None = None

        # Per-knowledge-base embedding provider cache
        # Key: (normalized_provider, model_name, dimensions, base_url)
        self._kb_embedding_cache: dict[tuple[str, str, int, str | None], EmbeddingProvider] = {}

        # Custom factory overrides (useful for plugins or test doubles)
        self._custom_llm_factories: dict[str, Callable[[LLMConfig], LLMProvider]] = {}
        self._custom_embedding_factories: dict[str, Callable[[EmbeddingConfig], EmbeddingProvider]] = {}
        self._custom_reranker_factories: dict[str, Callable[[RerankerConfig], RerankerProvider]] = {}

    # ── Default Configurations ────────────────────────────────────────────────

    def _build_default_llm_config(self) -> LLMConfig:
        """Construct default LLM config from application settings."""
        # Preference: Gemini if key configured, otherwise Ollama or fallback
        api_key = getattr(self._settings, "GOOGLE_API_KEY", "") or ""
        if api_key and api_key != "mock-key-for-tests":
            return LLMConfig(
                provider=ProviderType.GEMINI,
                model=getattr(self._settings, "GEMINI_CHAT_MODEL", "gemini-3.5-flash-lite"),
                api_key=api_key,
            )
        return LLMConfig(
            provider=ProviderType.OLLAMA,
            model=getattr(self._settings, "OLLAMA_CHAT_MODEL", "llama3.2"),
            base_url=getattr(self._settings, "OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        )

    def _build_default_embedding_config(self) -> EmbeddingConfig:
        """Construct default embedding config from application settings."""
        return EmbeddingConfig(
            provider=ProviderType.SENTENCE_TRANSFORMERS,
            model=getattr(self._settings, "LOCAL_EMBEDDING_MODEL", "intfloat/multilingual-e5-small"),
            dimensions=getattr(self._settings, "EMBEDDING_DIMENSIONS", 768),
            device=getattr(self._settings, "MODEL_DEVICE", "auto"),
        )

    def _build_default_reranker_config(self) -> RerankerConfig:
        """Construct default reranker config from application settings."""
        return RerankerConfig(
            provider=ProviderType.CROSS_ENCODER,
            model=getattr(self._settings, "LOCAL_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            enabled=True,
            device=getattr(self._settings, "MODEL_DEVICE", "auto"),
        )

    # ── Configuration Management ──────────────────────────────────────────────

    def get_llm_config(self) -> LLMConfig:
        with self._lock:
            if self._llm_config is None:
                self._llm_config = self._build_default_llm_config()
            return self._llm_config

    def set_llm_config(self, config: LLMConfig) -> None:
        """Set the active LLM config and invalidate the cached active provider."""
        with self._lock:
            self._llm_config = config
            self._active_llm = None
            logger.info("Active LLM provider configured: %s (%s)", config.normalized_provider(), config.model)

    def get_embedding_config(self) -> EmbeddingConfig:
        with self._lock:
            if self._embedding_config is None:
                self._embedding_config = self._build_default_embedding_config()
            return self._embedding_config

    def set_embedding_config(self, config: EmbeddingConfig) -> None:
        """Set the active embedding config and invalidate the cached active provider."""
        with self._lock:
            self._embedding_config = config
            self._active_embedding = None
            logger.info("Active embedding provider configured: %s (%s)", config.normalized_provider(), config.model)

    def get_reranker_config(self) -> RerankerConfig:
        with self._lock:
            if self._reranker_config is None:
                self._reranker_config = self._build_default_reranker_config()
            return self._reranker_config

    def set_reranker_config(self, config: RerankerConfig) -> None:
        """Set the active reranker config and invalidate the cached active provider."""
        with self._lock:
            self._reranker_config = config
            self._active_reranker = None
            logger.info("Active reranker configured: %s (enabled=%s)", config.normalized_provider(), config.enabled)

    # ── Active Providers Accessors ────────────────────────────────────────────

    def get_llm_provider(self) -> LLMProvider:
        """Get the cached active LLM provider, creating it if needed."""
        with self._lock:
            if self._active_llm is None:
                config = self.get_llm_config()
                self._active_llm = self.create_llm_provider(config)
            return self._active_llm

    def get_embedding_provider(self) -> EmbeddingProvider:
        """Get the cached active Embedding provider, creating it if needed."""
        with self._lock:
            if self._active_embedding is None:
                config = self.get_embedding_config()
                self._active_embedding = self.create_embedding_provider(config)
            return self._active_embedding

    def get_reranker_provider(self) -> RerankerProvider:
        """Get the cached active Reranker provider, creating it if needed."""
        with self._lock:
            if self._active_reranker is None:
                config = self.get_reranker_config()
                self._active_reranker = self.create_reranker_provider(config)
            return self._active_reranker

    # ── Per-KnowledgeBase Embedding Provider ──────────────────────────────────

    def get_knowledge_base_embedding_provider(
        self,
        kb_config: EmbeddingConfiguration | IndexMetadata | EmbeddingConfig,
    ) -> EmbeddingProvider:
        """Get or instantiate a cached embedding provider for a specific knowledge base."""
        with self._lock:
            if isinstance(kb_config, (EmbeddingConfiguration, IndexMetadata)):
                embed_config = kb_config.to_embedding_config()
            else:
                embed_config = kb_config

            provider_name = embed_config.normalized_provider()
            key = (
                provider_name,
                embed_config.model,
                int(embed_config.dimensions or 0),
                embed_config.base_url,
            )
            if key not in self._kb_embedding_cache:
                self._kb_embedding_cache[key] = self.create_embedding_provider(embed_config)
            return self._kb_embedding_cache[key]

    # ── Factories ─────────────────────────────────────────────────────────────

    def register_custom_llm_factory(
        self,
        provider_name: str,
        factory: Callable[[LLMConfig], LLMProvider],
    ) -> None:
        """Register a custom factory for creating an LLM provider."""
        with self._lock:
            self._custom_llm_factories[provider_name.strip().lower()] = factory

    def register_custom_embedding_factory(
        self,
        provider_name: str,
        factory: Callable[[EmbeddingConfig], EmbeddingProvider],
    ) -> None:
        """Register a custom factory for creating an embedding provider."""
        with self._lock:
            self._custom_embedding_factories[provider_name.strip().lower()] = factory

    def register_custom_reranker_factory(
        self,
        provider_name: str,
        factory: Callable[[RerankerConfig], RerankerProvider],
    ) -> None:
        """Register a custom factory for creating a reranker provider."""
        with self._lock:
            self._custom_reranker_factories[provider_name.strip().lower()] = factory

    def create_llm_provider(self, config: LLMConfig) -> LLMProvider:
        """Instantiate an LLMProvider from configuration."""
        name = config.normalized_provider()

        if name in self._custom_llm_factories:
            return self._custom_llm_factories[name](config)

        if name in (ProviderType.GEMINI.value, "gemini"):
            from app.providers.llm.gemini import GeminiLLMProvider
            api_key = config.api_key or getattr(self._settings, "GOOGLE_API_KEY", None)
            return GeminiLLMProvider(api_key=api_key, model=config.model)

        if name in (ProviderType.OLLAMA.value, "ollama"):
            from app.providers.llm.ollama import OllamaLLMProvider
            base_url = config.base_url or getattr(self._settings, "OLLAMA_BASE_URL", "http://127.0.0.1:11434")
            return OllamaLLMProvider(
                base_url=base_url,
                model=config.model,
                timeout_seconds=config.timeout_seconds,
            )

        if name in (ProviderType.OPENAI_COMPATIBLE.value, "openai_compatible"):
            from app.providers.llm.openai_compatible import OpenAICompatibleLLMProvider
            base_url = config.base_url or "https://api.openai.com/v1"
            api_key = config.api_key or getattr(self._settings, "OPENAI_API_KEY", "")
            return OpenAICompatibleLLMProvider(
                base_url=base_url,
                model=config.model,
                api_key=api_key,
                timeout_seconds=config.timeout_seconds,
            )

        if name in (ProviderType.OPENAI.value, "openai"):
            from app.providers.llm.openai_compatible import OpenAICompatibleLLMProvider
            api_key = config.api_key or getattr(self._settings, "OPENAI_API_KEY", "")
            return OpenAICompatibleLLMProvider.for_openai(
                api_key=api_key,
                model=config.model,
                timeout_seconds=config.timeout_seconds,
            )

        if name in (ProviderType.LM_STUDIO.value, "lm_studio"):
            from app.providers.llm.openai_compatible import OpenAICompatibleLLMProvider
            base_url = config.base_url or "http://127.0.0.1:1234/v1"
            return OpenAICompatibleLLMProvider.for_lm_studio(
                base_url=base_url,
                model=config.model,
                timeout_seconds=config.timeout_seconds,
            )

        if name in (ProviderType.LLAMA_CPP.value, "llama_cpp"):
            from app.providers.llm.openai_compatible import OpenAICompatibleLLMProvider
            base_url = config.base_url or "http://127.0.0.1:8080/v1"
            return OpenAICompatibleLLMProvider.for_llama_cpp(
                base_url=base_url,
                model=config.model,
                timeout_seconds=config.timeout_seconds,
            )

        if name in (ProviderType.ANTHROPIC.value, "anthropic"):
            from app.providers.llm.anthropic import AnthropicLLMProvider
            api_key = config.api_key or getattr(self._settings, "ANTHROPIC_API_KEY", "")
            return AnthropicLLMProvider(
                api_key=api_key,
                model=config.model,
                timeout_seconds=config.timeout_seconds,
            )

        raise ValueError(f"Unsupported LLM provider: {config.provider}")

    def create_embedding_provider(self, config: EmbeddingConfig) -> EmbeddingProvider:
        """Instantiate an EmbeddingProvider from configuration."""
        name = config.normalized_provider()

        if name in self._custom_embedding_factories:
            return self._custom_embedding_factories[name](config)

        if name in (ProviderType.SENTENCE_TRANSFORMERS.value, "sentence_transformers", "local"):
            from app.providers.embeddings.sentence_transformers import (
                SentenceTransformersEmbeddingProvider,
            )
            return SentenceTransformersEmbeddingProvider(
                model=config.model,
                dimensions=config.dimensions,
                device=config.device,
                cache_dir=config.cache_dir,
                batch_size=config.batch_size,
                normalize_embeddings=config.normalize_embeddings,
                use_e5_prefixes=config.use_e5_prefixes,
            )

        if name in (ProviderType.GEMINI.value, "gemini"):
            from app.providers.embeddings.gemini_embedding import (
                GeminiEmbeddingProvider,
            )
            api_key = config.api_key or getattr(self._settings, "GOOGLE_API_KEY", None)
            return GeminiEmbeddingProvider(
                api_key=api_key,
                model=config.model,
                dimensions=config.dimensions,
            )

        if name in (ProviderType.OLLAMA.value, "ollama"):
            from app.providers.embeddings.ollama import OllamaEmbeddingProvider
            base_url = config.base_url or getattr(self._settings, "OLLAMA_BASE_URL", "http://127.0.0.1:11434")
            return OllamaEmbeddingProvider(
                base_url=base_url,
                model=config.model,
                dimensions=config.dimensions,
            )

        if name in (ProviderType.OPENAI_COMPATIBLE.value, "openai_compatible"):
            from app.providers.embeddings.api import OpenAICompatibleEmbeddingProvider
            base_url = config.base_url or "https://api.openai.com/v1"
            api_key = config.api_key or getattr(self._settings, "OPENAI_API_KEY", "")
            return OpenAICompatibleEmbeddingProvider(
                base_url=base_url,
                model=config.model,
                dimensions=config.dimensions,
                api_key=api_key,
            )

        if name in (ProviderType.OPENAI.value, "openai"):
            from app.providers.embeddings.api import OpenAICompatibleEmbeddingProvider
            api_key = config.api_key or getattr(self._settings, "OPENAI_API_KEY", "")
            return OpenAICompatibleEmbeddingProvider.for_openai(
                api_key=api_key,
                model=config.model,
                dimensions=config.dimensions,
            )

        raise ValueError(f"Unsupported embedding provider: {config.provider}")

    def create_reranker_provider(self, config: RerankerConfig) -> RerankerProvider:
        """Instantiate a RerankerProvider from configuration."""
        name = config.normalized_provider()

        if name in self._custom_reranker_factories:
            return self._custom_reranker_factories[name](config)

        if not config.enabled or name in ("none", "disabled", "noop"):
            from app.providers.rerankers.cross_encoder import NoOpReranker
            return NoOpReranker()

        if name in (ProviderType.CROSS_ENCODER.value, "cross_encoder", "local", "sentence_transformers"):
            from app.providers.rerankers.cross_encoder import (
                LocalCrossEncoderReranker,
            )
            return LocalCrossEncoderReranker(
                model=config.model,
                device=config.device,
                cache_dir=config.cache_dir,
                max_length=config.max_length,
                batch_size=config.batch_size,
            )

        if name in ("cohere", "jina", "openai_compatible", "api"):
            from app.providers.rerankers.api import APIRerankerProvider
            style = "cohere" if name == "cohere" else ("jina" if name == "jina" else "openai_compatible")
            api_key = config.api_key or (
                getattr(self._settings, "COHERE_API_KEY", "")
                if style == "cohere"
                else getattr(self._settings, "JINA_API_KEY", "")
            )
            base_url = config.base_url or (
                "https://api.cohere.com/v1"
                if style == "cohere"
                else ("https://api.jina.ai/v1" if style == "jina" else "http://127.0.0.1:8000/v1")
            )
            return APIRerankerProvider(
                base_url=base_url,
                model=config.model,
                api_key=api_key,
                style=style,
            )

        raise ValueError(f"Unsupported reranker provider: {config.provider}")

    def clear_cache(self) -> None:
        """Clear cached active providers and KB-specific embedding instances."""
        with self._lock:
            self._active_llm = None
            self._active_embedding = None
            self._active_reranker = None
            self._kb_embedding_cache.clear()
