"""
backend/app/core/model_studio.py
────────────────────────────────────────────────────────────────────────────────
Model Studio Service & Compatibility Validation Engine for Quantiva Desktop.

Responsibilities:
1. Independent Configuration Management:
   - CRUD and default selection for ModelConfiguration (LLM),
     EmbeddingConfiguration, and RetrievalConfiguration (Reranker).
   - Idempotent seeding of standard out-of-the-box local & cloud presets.
   - Zero-plaintext API key storage via CredentialManager (OS keyring / DPAPI).
2. Live Connection & Capability Testing:
   - test_llm_config: Probes connectivity, measures latency, detects capabilities.
   - test_embedding_config: Probes connectivity, detects vector dimensionality.
   - test_reranker_config: Probes reranker relevance on domain-relevant test pairs.
3. Ollama & Local Model Integration:
   - Detects Windows Ollama installation (port 11434, %LOCALAPPDATA%, PATH).
   - Lists local models with parameter size and quantization info.
   - Streams download progress for Ollama and HuggingFace local models.
4. Embedding Compatibility Validation:
   - Analyzes impact of embedding configuration changes on existing KnowledgeBases.
   - Enforces IndexMetadata.needs_reindex = True when models/dimensions change.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy import select, update
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings, settings
from app.core.application_context import get_application_context
from app.providers.interfaces.reranker import RerankDocument
from app.providers.interfaces.types import (
    ConnectionTestResult,
    ProviderCapabilities,
    ProviderType,
)
from app.providers.llm.ollama import OllamaRuntimeManager
from app.providers.local_models import model_cache_dir
from app.providers.registry import (
    EmbeddingConfig,
    LLMConfig,
    ProviderRegistry,
    RerankerConfig,
)
from app.security.credentials import CredentialManager, get_credential_manager
from app.storage.database import SessionLocal, session_scope
from app.storage.models import (
    EmbeddingConfiguration,
    IndexMetadata,
    KnowledgeBase,
    ModelConfiguration,
    RetrievalConfiguration,
)

logger = logging.getLogger(__name__)

# ── Preset Constants ──────────────────────────────────────────────────────────

PRESET_LOCAL_LLM_NAME = "Default Local LLM"
PRESET_CLOUD_LLM_NAME = "Default Cloud LLM"
PRESET_LOCAL_EMBEDDING_NAME = "Default Local Embedding"
PRESET_LOCAL_RERANKER_NAME = "Default Local Reranker"

SAMPLE_RERANK_QUERY = "What was Q4 revenue growth?"
SAMPLE_PASSAGE_RELEVANT = "Revenue grew 18% in Q4"
SAMPLE_PASSAGE_IRRELEVANT = "The office cafeteria menu changed"


# ── Structured Result Dataclasses ─────────────────────────────────────────────

@dataclass
class OllamaDetectionResult:
    """Detection status for the Ollama desktop runtime."""

    is_installed: bool
    is_running: bool
    binary_path: str | None
    base_url: str
    version: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AffectedKnowledgeBase:
    """Summary of a knowledge base impacted by an embedding configuration change."""

    knowledge_base_id: str
    name: str
    document_count: int
    chunk_count: int
    current_provider: str | None
    current_model: str | None
    current_dimensions: int | None
    needs_reindex: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EmbeddingImpactResult:
    """Compatibility report when modifying or switching an embedding configuration."""

    config_id: str
    new_provider: str
    new_model: str
    new_dimensions: int
    requires_reindex: bool
    affected_knowledge_bases: list[AffectedKnowledgeBase] = field(default_factory=list)
    warning_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_id": self.config_id,
            "new_provider": self.new_provider,
            "new_model": self.new_model,
            "new_dimensions": self.new_dimensions,
            "requires_reindex": self.requires_reindex,
            "affected_knowledge_bases": [
                kb.to_dict() for kb in self.affected_knowledge_bases
            ],
            "warning_message": self.warning_message,
        }


# ── Model Studio Service ──────────────────────────────────────────────────────

class ModelStudioService:
    """Central service managing Model Studio operations in Quantiva Desktop."""

    def __init__(
        self,
        session_factory: sessionmaker[Session] | None = None,
        provider_registry: ProviderRegistry | None = None,
        credential_manager: CredentialManager | None = None,
        settings_obj: Settings | None = None,
    ) -> None:
        self._session_factory = session_factory or SessionLocal
        self._registry = provider_registry or get_application_context().provider_registry
        self._credentials = credential_manager or get_credential_manager()
        self._settings = settings_obj or settings

    # ── Preset Seeding ────────────────────────────────────────────────────────

    def seed_presets(
        self,
        session: Session | None = None,
    ) -> dict[str, Any]:
        """Seed default out-of-the-box configurations idempotently on first launch.

        Presets:
        1. Default Local LLM: Ollama (qwen2.5:7b, temp 0.2, max_tokens 4096)
        2. Default Cloud LLM: Gemini (gemini-2.5-flash, temp 0.2, max_tokens 4096)
        3. Default Local Embedding: Sentence Transformers (intfloat/multilingual-e5-base, 768 dims)
        4. Default Local Reranker: Cross Encoder (cross-encoder/ms-marco-MiniLM-L-6-v2)
        """
        def _seed(s: Session) -> dict[str, Any]:
            created: dict[str, Any] = {}

            # 1. Local LLM Preset
            local_llm = s.scalar(
                select(ModelConfiguration).where(
                    ModelConfiguration.name == PRESET_LOCAL_LLM_NAME
                )
            )
            if local_llm is None:
                has_other_default_llm = s.scalar(
                    select(ModelConfiguration).where(ModelConfiguration.is_default.is_(True))
                ) is not None
                local_llm = ModelConfiguration(
                    id=str(uuid.uuid4()),
                    name=PRESET_LOCAL_LLM_NAME,
                    provider_type=ProviderType.OLLAMA.value,
                    model_name="qwen2.5:7b",
                    base_url=getattr(self._settings, "OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
                    temperature=0.2,
                    max_tokens=4096,
                    is_default=not has_other_default_llm,
                )
                s.add(local_llm)
                created["local_llm"] = local_llm
            else:
                created["local_llm"] = local_llm

            # 2. Cloud LLM Preset (Gemini)
            cloud_llm = s.scalar(
                select(ModelConfiguration).where(
                    ModelConfiguration.name == PRESET_CLOUD_LLM_NAME
                )
            )
            if cloud_llm is None:
                cloud_llm = ModelConfiguration(
                    id=str(uuid.uuid4()),
                    name=PRESET_CLOUD_LLM_NAME,
                    provider_type=ProviderType.GEMINI.value,
                    model_name="gemini-2.5-flash",
                    temperature=0.2,
                    max_tokens=4096,
                    is_default=False,
                )
                s.add(cloud_llm)
                created["cloud_llm"] = cloud_llm
            else:
                created["cloud_llm"] = cloud_llm

            # 3. Local Embedding Preset
            local_embed = s.scalar(
                select(EmbeddingConfiguration).where(
                    EmbeddingConfiguration.name == PRESET_LOCAL_EMBEDDING_NAME
                )
            )
            if local_embed is None:
                has_other_default_embed = s.scalar(
                    select(EmbeddingConfiguration).where(
                        EmbeddingConfiguration.is_default.is_(True)
                    )
                ) is not None
                local_embed = EmbeddingConfiguration(
                    id=str(uuid.uuid4()),
                    name=PRESET_LOCAL_EMBEDDING_NAME,
                    provider_type=ProviderType.SENTENCE_TRANSFORMERS.value,
                    model_name="intfloat/multilingual-e5-base",
                    dimensions=768,
                    batch_size=32,
                    is_default=not has_other_default_embed,
                )
                s.add(local_embed)
                created["local_embedding"] = local_embed
            else:
                created["local_embedding"] = local_embed

            # 4. Local Reranker Preset (inside RetrievalConfiguration)
            local_reranker = s.scalar(
                select(RetrievalConfiguration).where(
                    RetrievalConfiguration.name == PRESET_LOCAL_RERANKER_NAME
                )
            )
            if local_reranker is None:
                has_other_default_retrieval = s.scalar(
                    select(RetrievalConfiguration).where(
                        RetrievalConfiguration.is_default.is_(True)
                    )
                ) is not None
                local_reranker = RetrievalConfiguration(
                    id=str(uuid.uuid4()),
                    name=PRESET_LOCAL_RERANKER_NAME,
                    search_mode="hybrid",
                    dense_top_k=20,
                    sparse_top_k=20,
                    rrf_k=60,
                    dense_weight=1.0,
                    sparse_weight=1.0,
                    enable_reranking=True,
                    reranker_provider=ProviderType.CROSS_ENCODER.value,
                    reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
                    final_top_k=8,
                    enable_agentic_mode=True,
                    max_agent_iterations=5,
                    is_default=not has_other_default_retrieval,
                )
                s.add(local_reranker)
                created["local_reranker"] = local_reranker
            else:
                created["local_reranker"] = local_reranker

            s.flush()
            return created

        if session is not None:
            return _seed(session)
        with session_scope(self._session_factory) as s:
            return _seed(s)

    def ensure_presets(self, session: Session | None = None) -> None:
        """Helper to ensure presets exist without returning values."""
        self.seed_presets(session=session)

    # ── LLM Configuration Management ──────────────────────────────────────────

    def list_llm_configs(
        self,
        session: Session | None = None,
    ) -> list[ModelConfiguration]:
        def _list(s: Session) -> list[ModelConfiguration]:
            return list(
                s.scalars(
                    select(ModelConfiguration).order_by(
                        ModelConfiguration.is_default.desc(),
                        ModelConfiguration.name.asc(),
                    )
                ).all()
            )

        if session is not None:
            return _list(session)
        with session_scope(self._session_factory) as s:
            return _list(s)

    def get_llm_config(
        self,
        config_id: str,
        session: Session | None = None,
    ) -> ModelConfiguration | None:
        def _get(s: Session) -> ModelConfiguration | None:
            return s.get(ModelConfiguration, config_id)

        if session is not None:
            return _get(session)
        with session_scope(self._session_factory) as s:
            return _get(s)

    def create_llm_config(
        self,
        payload: dict[str, Any],
        session: Session | None = None,
    ) -> ModelConfiguration:
        def _create(s: Session) -> ModelConfiguration:
            name = payload.get("name")
            if not name:
                raise ValueError("Configuration name is required.")

            existing = s.scalar(
                select(ModelConfiguration).where(ModelConfiguration.name == name)
            )
            if existing is not None:
                raise ValueError(f"An LLM configuration named '{name}' already exists.")

            config_id = payload.get("id") or str(uuid.uuid4())
            is_default = bool(payload.get("is_default", False))

            if is_default:
                s.execute(
                    update(ModelConfiguration).values(is_default=False)
                )

            api_key = payload.get("api_key")
            has_api_key = False
            credential_key_id = None
            if api_key:
                credential_key_id = config_id
                self._credentials.set_api_key(credential_key_id, api_key)
                has_api_key = True

            config = ModelConfiguration(
                id=config_id,
                name=name,
                provider_type=str(payload.get("provider_type", "ollama")).strip().lower(),
                model_name=str(payload.get("model_name", "")),
                base_url=payload.get("base_url"),
                credential_key_id=credential_key_id,
                has_api_key=has_api_key,
                temperature=float(payload.get("temperature", 0.2)),
                max_tokens=int(payload.get("max_tokens", 4096)),
                context_window=payload.get("context_window"),
                capabilities_json=payload.get("capabilities_json") or {},
                is_default=is_default,
            )
            s.add(config)
            s.flush()

            if is_default:
                self._sync_registry_llm(config)

            return config

        if session is not None:
            return _create(session)
        with session_scope(self._session_factory) as s:
            return _create(s)

    def update_llm_config(
        self,
        config_id: str,
        payload: dict[str, Any],
        session: Session | None = None,
    ) -> ModelConfiguration:
        def _update(s: Session) -> ModelConfiguration:
            config = s.get(ModelConfiguration, config_id)
            if config is None:
                raise KeyError(f"LLM configuration not found: {config_id}")

            if "name" in payload and payload["name"] != config.name:
                clash = s.scalar(
                    select(ModelConfiguration).where(
                        ModelConfiguration.name == payload["name"],
                        ModelConfiguration.id != config_id,
                    )
                )
                if clash is not None:
                    raise ValueError(f"An LLM configuration named '{payload['name']}' already exists.")
                config.name = payload["name"]

            if "provider_type" in payload:
                config.provider_type = str(payload["provider_type"]).strip().lower()
            if "model_name" in payload:
                config.model_name = str(payload["model_name"])
            if "base_url" in payload:
                config.base_url = payload["base_url"]
            if "temperature" in payload:
                config.temperature = float(payload["temperature"])
            if "max_tokens" in payload:
                config.max_tokens = int(payload["max_tokens"])
            if "context_window" in payload:
                config.context_window = payload["context_window"]
            if "capabilities_json" in payload:
                config.capabilities_json = payload["capabilities_json"] or {}

            # Handle API Key updates
            if "api_key" in payload:
                new_key = payload["api_key"]
                if new_key:
                    cred_id = config.credential_key_id or config.id
                    self._credentials.set_api_key(cred_id, new_key)
                    config.credential_key_id = cred_id
                    config.has_api_key = True
                else:
                    if config.credential_key_id:
                        self._credentials.delete_api_key(config.credential_key_id)
                    config.credential_key_id = None
                    config.has_api_key = False

            if payload.get("is_default") is True:
                s.execute(
                    update(ModelConfiguration).where(
                        ModelConfiguration.id != config_id
                    ).values(is_default=False)
                )
                config.is_default = True

            s.flush()

            if config.is_default:
                self._sync_registry_llm(config)

            return config

        if session is not None:
            return _update(session)
        with session_scope(self._session_factory) as s:
            return _update(s)

    def delete_llm_config(
        self,
        config_id: str,
        session: Session | None = None,
    ) -> bool:
        def _delete(s: Session) -> bool:
            config = s.get(ModelConfiguration, config_id)
            if config is None:
                return False

            if config.credential_key_id:
                self._credentials.delete_api_key(config.credential_key_id)

            was_default = config.is_default
            s.delete(config)
            s.flush()

            if was_default:
                next_cand = s.scalar(select(ModelConfiguration))
                if next_cand is not None:
                    next_cand.is_default = True
                    s.flush()
                    self._sync_registry_llm(next_cand)

            return True

        if session is not None:
            return _delete(session)
        with session_scope(self._session_factory) as s:
            return _delete(s)

    def set_default_llm_config(
        self,
        config_id: str,
        session: Session | None = None,
    ) -> ModelConfiguration:
        def _set_default(s: Session) -> ModelConfiguration:
            target = s.get(ModelConfiguration, config_id)
            if target is None:
                raise KeyError(f"LLM configuration not found: {config_id}")

            s.execute(
                update(ModelConfiguration).values(is_default=False)
            )
            target.is_default = True
            s.flush()

            self._sync_registry_llm(target)
            return target

        if session is not None:
            return _set_default(session)
        with session_scope(self._session_factory) as s:
            return _set_default(s)

    def get_default_llm_config(
        self,
        session: Session | None = None,
    ) -> ModelConfiguration | None:
        def _get_default(s: Session) -> ModelConfiguration | None:
            return s.scalar(
                select(ModelConfiguration).where(ModelConfiguration.is_default.is_(True))
            )

        if session is not None:
            return _get_default(session)
        with session_scope(self._session_factory) as s:
            return _get_default(s)

    def _sync_registry_llm(self, config: ModelConfiguration) -> None:
        api_key = None
        if config.has_api_key and config.credential_key_id:
            api_key = self._credentials.get_api_key(config.credential_key_id)
        self._registry.set_llm_config(
            LLMConfig(
                provider=config.provider_type,
                model=config.model_name,
                base_url=config.base_url,
                api_key=api_key,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
        )

    # ── Embedding Configuration Management ────────────────────────────────────

    def list_embedding_configs(
        self,
        session: Session | None = None,
    ) -> list[EmbeddingConfiguration]:
        def _list(s: Session) -> list[EmbeddingConfiguration]:
            return list(
                s.scalars(
                    select(EmbeddingConfiguration).order_by(
                        EmbeddingConfiguration.is_default.desc(),
                        EmbeddingConfiguration.name.asc(),
                    )
                ).all()
            )

        if session is not None:
            return _list(session)
        with session_scope(self._session_factory) as s:
            return _list(s)

    def get_embedding_config(
        self,
        config_id: str,
        session: Session | None = None,
    ) -> EmbeddingConfiguration | None:
        def _get(s: Session) -> EmbeddingConfiguration | None:
            return s.get(EmbeddingConfiguration, config_id)

        if session is not None:
            return _get(session)
        with session_scope(self._session_factory) as s:
            return _get(s)

    def create_embedding_config(
        self,
        payload: dict[str, Any],
        session: Session | None = None,
    ) -> EmbeddingConfiguration:
        def _create(s: Session) -> EmbeddingConfiguration:
            name = payload.get("name")
            if not name:
                raise ValueError("Configuration name is required.")

            existing = s.scalar(
                select(EmbeddingConfiguration).where(EmbeddingConfiguration.name == name)
            )
            if existing is not None:
                raise ValueError(f"An embedding configuration named '{name}' already exists.")

            dimensions = payload.get("dimensions")
            if dimensions is None or int(dimensions) <= 0:
                raise ValueError("Positive embedding dimensions are required.")

            config_id = payload.get("id") or str(uuid.uuid4())
            is_default = bool(payload.get("is_default", False))

            if is_default:
                s.execute(
                    update(EmbeddingConfiguration).values(is_default=False)
                )

            api_key = payload.get("api_key")
            has_api_key = False
            credential_key_id = None
            if api_key:
                credential_key_id = config_id
                self._credentials.set_api_key(credential_key_id, api_key)
                has_api_key = True

            config = EmbeddingConfiguration(
                id=config_id,
                name=name,
                provider_type=str(payload.get("provider_type", "sentence_transformers")).strip().lower(),
                model_name=str(payload.get("model_name", "")),
                base_url=payload.get("base_url"),
                credential_key_id=credential_key_id,
                has_api_key=has_api_key,
                dimensions=int(dimensions),
                batch_size=int(payload.get("batch_size", 32)),
                is_default=is_default,
            )
            s.add(config)
            s.flush()

            if is_default:
                self._sync_registry_embedding(config)

            return config

        if session is not None:
            return _create(session)
        with session_scope(self._session_factory) as s:
            return _create(s)

    def update_embedding_config(
        self,
        config_id: str,
        payload: dict[str, Any],
        session: Session | None = None,
    ) -> EmbeddingConfiguration:
        def _update(s: Session) -> EmbeddingConfiguration:
            config = s.get(EmbeddingConfiguration, config_id)
            if config is None:
                raise KeyError(f"Embedding configuration not found: {config_id}")

            if "name" in payload and payload["name"] != config.name:
                clash = s.scalar(
                    select(EmbeddingConfiguration).where(
                        EmbeddingConfiguration.name == payload["name"],
                        EmbeddingConfiguration.id != config_id,
                    )
                )
                if clash is not None:
                    raise ValueError(f"An embedding configuration named '{payload['name']}' already exists.")
                config.name = payload["name"]

            if "provider_type" in payload:
                config.provider_type = str(payload["provider_type"]).strip().lower()
            if "model_name" in payload:
                config.model_name = str(payload["model_name"])
            if "base_url" in payload:
                config.base_url = payload["base_url"]
            if "dimensions" in payload and payload["dimensions"] is not None:
                config.dimensions = int(payload["dimensions"])
            if "batch_size" in payload and payload["batch_size"] is not None:
                config.batch_size = int(payload["batch_size"])

            if "api_key" in payload:
                new_key = payload["api_key"]
                if new_key:
                    cred_id = config.credential_key_id or config.id
                    self._credentials.set_api_key(cred_id, new_key)
                    config.credential_key_id = cred_id
                    config.has_api_key = True
                else:
                    if config.credential_key_id:
                        self._credentials.delete_api_key(config.credential_key_id)
                    config.credential_key_id = None
                    config.has_api_key = False

            if payload.get("is_default") is True:
                s.execute(
                    update(EmbeddingConfiguration).where(
                        EmbeddingConfiguration.id != config_id
                    ).values(is_default=False)
                )
                config.is_default = True

            s.flush()

            if config.is_default:
                self._sync_registry_embedding(config)

            return config

        if session is not None:
            return _update(session)
        with session_scope(self._session_factory) as s:
            return _update(s)

    def delete_embedding_config(
        self,
        config_id: str,
        session: Session | None = None,
    ) -> bool:
        def _delete(s: Session) -> bool:
            config = s.get(EmbeddingConfiguration, config_id)
            if config is None:
                return False

            if config.credential_key_id:
                self._credentials.delete_api_key(config.credential_key_id)

            was_default = config.is_default
            s.delete(config)
            s.flush()

            if was_default:
                next_cand = s.scalar(select(EmbeddingConfiguration))
                if next_cand is not None:
                    next_cand.is_default = True
                    s.flush()
                    self._sync_registry_embedding(next_cand)

            return True

        if session is not None:
            return _delete(session)
        with session_scope(self._session_factory) as s:
            return _delete(s)

    def set_default_embedding_config(
        self,
        config_id: str,
        session: Session | None = None,
    ) -> EmbeddingConfiguration:
        def _set_default(s: Session) -> EmbeddingConfiguration:
            target = s.get(EmbeddingConfiguration, config_id)
            if target is None:
                raise KeyError(f"Embedding configuration not found: {config_id}")

            s.execute(
                update(EmbeddingConfiguration).values(is_default=False)
            )
            target.is_default = True
            s.flush()

            self._sync_registry_embedding(target)
            return target

        if session is not None:
            return _set_default(session)
        with session_scope(self._session_factory) as s:
            return _set_default(s)

    def get_default_embedding_config(
        self,
        session: Session | None = None,
    ) -> EmbeddingConfiguration | None:
        def _get_default(s: Session) -> EmbeddingConfiguration | None:
            return s.scalar(
                select(EmbeddingConfiguration).where(EmbeddingConfiguration.is_default.is_(True))
            )

        if session is not None:
            return _get_default(session)
        with session_scope(self._session_factory) as s:
            return _get_default(s)

    def _sync_registry_embedding(self, config: EmbeddingConfiguration) -> None:
        api_key = None
        if config.has_api_key and config.credential_key_id:
            api_key = self._credentials.get_api_key(config.credential_key_id)
        self._registry.set_embedding_config(
            EmbeddingConfig(
                provider=config.provider_type,
                model=config.model_name,
                dimensions=config.dimensions,
                base_url=config.base_url,
                api_key=api_key,
                batch_size=config.batch_size,
            )
        )

    # ── Retrieval & Reranker Configuration Management ─────────────────────────

    def list_retrieval_configs(
        self,
        session: Session | None = None,
    ) -> list[RetrievalConfiguration]:
        def _list(s: Session) -> list[RetrievalConfiguration]:
            return list(
                s.scalars(
                    select(RetrievalConfiguration).order_by(
                        RetrievalConfiguration.is_default.desc(),
                        RetrievalConfiguration.name.asc(),
                    )
                ).all()
            )

        if session is not None:
            return _list(session)
        with session_scope(self._session_factory) as s:
            return _list(s)

    def get_retrieval_config(
        self,
        config_id: str,
        session: Session | None = None,
    ) -> RetrievalConfiguration | None:
        def _get(s: Session) -> RetrievalConfiguration | None:
            return s.get(RetrievalConfiguration, config_id)

        if session is not None:
            return _get(session)
        with session_scope(self._session_factory) as s:
            return _get(s)

    def create_retrieval_config(
        self,
        payload: dict[str, Any],
        session: Session | None = None,
    ) -> RetrievalConfiguration:
        def _create(s: Session) -> RetrievalConfiguration:
            name = payload.get("name")
            if not name:
                raise ValueError("Configuration name is required.")

            existing = s.scalar(
                select(RetrievalConfiguration).where(RetrievalConfiguration.name == name)
            )
            if existing is not None:
                raise ValueError(f"A retrieval configuration named '{name}' already exists.")

            config_id = payload.get("id") or str(uuid.uuid4())
            is_default = bool(payload.get("is_default", False))

            if is_default:
                s.execute(
                    update(RetrievalConfiguration).values(is_default=False)
                )

            config = RetrievalConfiguration(
                id=config_id,
                name=name,
                search_mode=payload.get("search_mode", "hybrid"),
                dense_top_k=int(payload.get("dense_top_k", 20)),
                sparse_top_k=int(payload.get("sparse_top_k", 20)),
                rrf_k=int(payload.get("rrf_k", 60)),
                dense_weight=float(payload.get("dense_weight", 1.0)),
                sparse_weight=float(payload.get("sparse_weight", 1.0)),
                enable_reranking=bool(payload.get("enable_reranking", True)),
                reranker_provider=payload.get("reranker_provider") or payload.get("provider_type"),
                reranker_model=payload.get("reranker_model") or payload.get("model_name"),
                final_top_k=int(payload.get("final_top_k", 8)),
                enable_agentic_mode=bool(payload.get("enable_agentic_mode", True)),
                max_agent_iterations=int(payload.get("max_agent_iterations", 5)),
                is_default=is_default,
            )
            s.add(config)
            s.flush()

            if is_default:
                self._sync_registry_reranker(config)

            return config

        if session is not None:
            return _create(session)
        with session_scope(self._session_factory) as s:
            return _create(s)

    def update_retrieval_config(
        self,
        config_id: str,
        payload: dict[str, Any],
        session: Session | None = None,
    ) -> RetrievalConfiguration:
        def _update(s: Session) -> RetrievalConfiguration:
            config = s.get(RetrievalConfiguration, config_id)
            if config is None:
                raise KeyError(f"Retrieval configuration not found: {config_id}")

            if "name" in payload and payload["name"] != config.name:
                clash = s.scalar(
                    select(RetrievalConfiguration).where(
                        RetrievalConfiguration.name == payload["name"],
                        RetrievalConfiguration.id != config_id,
                    )
                )
                if clash is not None:
                    raise ValueError(f"A retrieval configuration named '{payload['name']}' already exists.")
                config.name = payload["name"]

            for field_name in (
                "search_mode",
                "dense_top_k",
                "sparse_top_k",
                "rrf_k",
                "dense_weight",
                "sparse_weight",
                "enable_reranking",
                "final_top_k",
                "enable_agentic_mode",
                "max_agent_iterations",
            ):
                if field_name in payload:
                    setattr(config, field_name, payload[field_name])

            if "reranker_provider" in payload or "provider_type" in payload:
                config.reranker_provider = payload.get("reranker_provider") or payload.get("provider_type")
            if "reranker_model" in payload or "model_name" in payload:
                config.reranker_model = payload.get("reranker_model") or payload.get("model_name")

            if payload.get("is_default") is True:
                s.execute(
                    update(RetrievalConfiguration).where(
                        RetrievalConfiguration.id != config_id
                    ).values(is_default=False)
                )
                config.is_default = True

            s.flush()

            if config.is_default:
                self._sync_registry_reranker(config)

            return config

        if session is not None:
            return _update(session)
        with session_scope(self._session_factory) as s:
            return _update(s)

    def delete_retrieval_config(
        self,
        config_id: str,
        session: Session | None = None,
    ) -> bool:
        def _delete(s: Session) -> bool:
            config = s.get(RetrievalConfiguration, config_id)
            if config is None:
                return False

            was_default = config.is_default
            s.delete(config)
            s.flush()

            if was_default:
                next_cand = s.scalar(select(RetrievalConfiguration))
                if next_cand is not None:
                    next_cand.is_default = True
                    s.flush()
                    self._sync_registry_reranker(next_cand)

            return True

        if session is not None:
            return _delete(session)
        with session_scope(self._session_factory) as s:
            return _delete(s)

    def set_default_retrieval_config(
        self,
        config_id: str,
        session: Session | None = None,
    ) -> RetrievalConfiguration:
        def _set_default(s: Session) -> RetrievalConfiguration:
            target = s.get(RetrievalConfiguration, config_id)
            if target is None:
                raise KeyError(f"Retrieval configuration not found: {config_id}")

            s.execute(
                update(RetrievalConfiguration).values(is_default=False)
            )
            target.is_default = True
            s.flush()

            self._sync_registry_reranker(target)
            return target

        if session is not None:
            return _set_default(session)
        with session_scope(self._session_factory) as s:
            return _set_default(s)

    def get_default_retrieval_config(
        self,
        session: Session | None = None,
    ) -> RetrievalConfiguration | None:
        def _get_default(s: Session) -> RetrievalConfiguration | None:
            return s.scalar(
                select(RetrievalConfiguration).where(RetrievalConfiguration.is_default.is_(True))
            )

        if session is not None:
            return _get_default(session)
        with session_scope(self._session_factory) as s:
            return _get_default(s)

    def _sync_registry_reranker(self, config: RetrievalConfiguration) -> None:
        self._registry.set_reranker_config(
            RerankerConfig(
                provider=config.reranker_provider or ProviderType.CROSS_ENCODER.value,
                model=config.reranker_model or "cross-encoder/ms-marco-MiniLM-L-6-v2",
                enabled=config.enable_reranking,
                top_k=config.final_top_k,
            )
        )

    # Reranker-specific convenience aliases
    list_reranker_configs = list_retrieval_configs
    get_reranker_config = get_retrieval_config
    create_reranker_config = create_retrieval_config
    update_reranker_config = update_retrieval_config
    delete_reranker_config = delete_retrieval_config
    set_default_reranker_config = set_default_retrieval_config
    get_default_reranker_config = get_default_retrieval_config

    # ── Live Connection & Capability Testing ──────────────────────────────────

    async def test_llm_config(
        self,
        payload: dict[str, Any],
        session: Session | None = None,
    ) -> ConnectionTestResult:
        """Instantiate a transient LLMProvider from unsaved form inputs or saved config ID,

        execute test_connection(), and return ConnectionTestResult with capabilities.
        """
        config_id = payload.get("config_id") or payload.get("id")
        provider_type = payload.get("provider_type")
        model_name = payload.get("model_name")
        base_url = payload.get("base_url")
        api_key = payload.get("api_key")
        temperature = payload.get("temperature", 0.2)
        max_tokens = payload.get("max_tokens", 4096)

        if config_id and (not provider_type or not model_name):
            saved = self.get_llm_config(config_id, session=session)
            if saved is not None:
                provider_type = provider_type or saved.provider_type
                model_name = model_name or saved.model_name
                base_url = base_url if base_url is not None else saved.base_url
                temperature = temperature if "temperature" in payload else saved.temperature
                max_tokens = max_tokens if "max_tokens" in payload else saved.max_tokens
                if not api_key and saved.has_api_key and saved.credential_key_id:
                    api_key = self._credentials.get_api_key(saved.credential_key_id)

        if not provider_type or not model_name:
            return ConnectionTestResult(
                success=False,
                latency_ms=0.0,
                provider=str(provider_type or ""),
                model=str(model_name or ""),
                error_code="MissingParameters",
                error_message="Both provider_type and model_name must be provided for testing.",
            )

        llm_config = LLMConfig(
            provider=provider_type,
            model=model_name,
            base_url=base_url,
            api_key=api_key,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
        )

        try:
            provider = self._registry.create_llm_provider(llm_config)
            test_res = await provider.test_connection()
            if test_res.success and test_res.capabilities is None:
                # Ensure capabilities descriptor is populated
                caps = provider.get_capabilities()
                test_res = ConnectionTestResult(
                    success=test_res.success,
                    latency_ms=test_res.latency_ms,
                    provider=test_res.provider,
                    model=test_res.model,
                    capabilities=caps,
                    error_code=test_res.error_code,
                    error_message=test_res.error_message,
                )
            return test_res
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error executing LLM test connection probe")
            return ConnectionTestResult(
                success=False,
                latency_ms=0.0,
                provider=str(provider_type),
                model=str(model_name),
                error_code=type(exc).__name__,
                error_message=str(exc),
            )

    async def test_embedding_config(
        self,
        payload: dict[str, Any],
        session: Session | None = None,
    ) -> ConnectionTestResult:
        """Probe embedding provider connectivity and measure vector dimensionality."""
        config_id = payload.get("config_id") or payload.get("id")
        provider_type = payload.get("provider_type")
        model_name = payload.get("model_name")
        base_url = payload.get("base_url")
        api_key = payload.get("api_key")
        dimensions = payload.get("dimensions")

        if config_id and (not provider_type or not model_name):
            saved = self.get_embedding_config(config_id, session=session)
            if saved is not None:
                provider_type = provider_type or saved.provider_type
                model_name = model_name or saved.model_name
                base_url = base_url if base_url is not None else saved.base_url
                dimensions = dimensions if dimensions is not None else saved.dimensions
                if not api_key and saved.has_api_key and saved.credential_key_id:
                    api_key = self._credentials.get_api_key(saved.credential_key_id)

        if not provider_type or not model_name:
            return ConnectionTestResult(
                success=False,
                latency_ms=0.0,
                provider=str(provider_type or ""),
                model=str(model_name or ""),
                error_code="MissingParameters",
                error_message="Both provider_type and model_name must be provided for testing.",
            )

        embed_config = EmbeddingConfig(
            provider=provider_type,
            model=model_name,
            dimensions=int(dimensions) if dimensions else None,
            base_url=base_url,
            api_key=api_key,
        )

        try:
            provider = self._registry.create_embedding_provider(embed_config)
            test_res = await provider.test_embedding()
            return test_res
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error executing embedding test connection probe")
            return ConnectionTestResult(
                success=False,
                latency_ms=0.0,
                provider=str(provider_type),
                model=str(model_name),
                detected_dimensions=None,
                error_code=type(exc).__name__,
                error_message=str(exc),
            )

    async def test_reranker_config(
        self,
        payload: dict[str, Any],
        session: Session | None = None,
    ) -> ConnectionTestResult:
        """Run a test rerank on two sample passages for query 'What was Q4 revenue growth?',

        verifying that the relevant passage ranks higher than the irrelevant one.
        """
        config_id = payload.get("config_id") or payload.get("id")
        provider_type = (
            payload.get("reranker_provider")
            or payload.get("provider_type")
            or ProviderType.CROSS_ENCODER.value
        )
        model_name = (
            payload.get("reranker_model")
            or payload.get("model_name")
            or "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        base_url = payload.get("base_url")
        api_key = payload.get("api_key")

        if config_id and not payload.get("reranker_model") and not payload.get("model_name"):
            saved = self.get_retrieval_config(config_id, session=session)
            if saved is not None:
                provider_type = saved.reranker_provider or provider_type
                model_name = saved.reranker_model or model_name

        reranker_config = RerankerConfig(
            provider=provider_type,
            model=model_name,
            base_url=base_url,
            api_key=api_key,
            enabled=True,
            top_k=2,
        )

        start_time = time.perf_counter()
        try:
            provider = self._registry.create_reranker_provider(reranker_config)

            doc_rel = RerankDocument(
                id="doc_rel",
                content=SAMPLE_PASSAGE_RELEVANT,
                initial_score=0.5,
            )
            doc_irrel = RerankDocument(
                id="doc_irrel",
                content=SAMPLE_PASSAGE_IRRELEVANT,
                initial_score=0.5,
            )

            results = await provider.rerank(
                query=SAMPLE_RERANK_QUERY,
                documents=[doc_rel, doc_irrel],
                top_k=2,
            )
            latency_ms = (time.perf_counter() - start_time) * 1000.0

            if not results:
                return ConnectionTestResult(
                    success=False,
                    latency_ms=round(latency_ms, 2),
                    provider=str(provider_type),
                    model=str(model_name),
                    error_code="EmptyResultsError",
                    error_message="Reranker returned empty results for sample passages.",
                )

            # Passage 1 (doc_rel) must rank higher than Passage 2 (doc_irrel)
            is_valid_ranking = (
                results[0].content == SAMPLE_PASSAGE_RELEVANT
                and (len(results) < 2 or results[0].rerank_score > results[1].rerank_score)
            )

            if not is_valid_ranking:
                return ConnectionTestResult(
                    success=False,
                    latency_ms=round(latency_ms, 2),
                    provider=str(provider_type),
                    model=str(model_name),
                    error_code="RankingValidationError",
                    error_message=(
                        f"Reranker ranking test failed: '{SAMPLE_PASSAGE_RELEVANT}' "
                        f"did not score strictly higher than '{SAMPLE_PASSAGE_IRRELEVANT}'."
                    ),
                )

            return ConnectionTestResult(
                success=True,
                latency_ms=round(latency_ms, 2),
                provider=str(provider_type),
                model=str(model_name),
            )

        except Exception as exc:  # noqa: BLE001
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            logger.exception("Error executing reranker test connection probe")
            return ConnectionTestResult(
                success=False,
                latency_ms=round(latency_ms, 2),
                provider=str(provider_type),
                model=str(model_name),
                error_code=type(exc).__name__,
                error_message=str(exc),
            )

    # ── Ollama & Local Model Integration ──────────────────────────────────────

    async def detect_ollama(
        self,
        base_url: str | None = None,
    ) -> OllamaDetectionResult:
        """Detect whether Ollama is installed and running on Windows.

        Checks:
        1. http://127.0.0.1:11434 (or configured URL)
        2. %LOCALAPPDATA%\\Programs\\Ollama\\ollama.exe
        3. shutil.which("ollama")
        """
        url = (base_url or getattr(self._settings, "OLLAMA_BASE_URL", "http://127.0.0.1:11434")).strip().rstrip("/")
        manager = OllamaRuntimeManager(base_url=url)

        # 1. Check daemon responsiveness
        is_running = await manager.is_daemon_running()

        # 2. Check standard Windows path
        binary_path = None
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            standard_path = Path(local_app_data) / "Programs" / "Ollama" / "ollama.exe"
            if standard_path.is_file():
                binary_path = str(standard_path)

        # 3. Check system PATH
        if not binary_path:
            which_res = shutil.which("ollama") or shutil.which("ollama.exe")
            if which_res:
                binary_path = which_res

        is_installed = is_running or (binary_path is not None)

        version = None
        if is_running:
            try:
                async with httpx.AsyncClient(timeout=3.0) as client:
                    resp = await client.get(f"{url}/api/version")
                    if resp.is_success:
                        version_data = resp.json()
                        version = version_data.get("version")
            except Exception:
                pass

        return OllamaDetectionResult(
            is_installed=is_installed,
            is_running=is_running,
            binary_path=binary_path,
            base_url=url,
            version=version,
        )

    async def list_ollama_models(
        self,
        base_url: str | None = None,
    ) -> list[dict[str, Any]]:
        """List installed local Ollama models with size and quantization metadata."""
        url = (base_url or getattr(self._settings, "OLLAMA_BASE_URL", "http://127.0.0.1:11434")).strip().rstrip("/")
        manager = OllamaRuntimeManager(base_url=url)
        return await manager.list_installed_models()

    async def pull_ollama_model(
        self,
        model_name: str,
        base_url: str | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream model download progress events from Ollama daemon."""
        url = (base_url or getattr(self._settings, "OLLAMA_BASE_URL", "http://127.0.0.1:11434")).strip().rstrip("/")
        manager = OllamaRuntimeManager(base_url=url)
        async for event in manager.pull_model_stream(model_name):
            yield event

    async def download_huggingface_model(
        self,
        model_name: str,
        kind: str = "embeddings",
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream local HuggingFace embedding/reranker weight download progress."""
        if not model_name or not model_name.strip():
            raise ValueError("model_name must be a non-empty string.")

        model_name = model_name.strip()
        kind_clean = "rerankers" if kind in ("reranker", "rerankers") else "embeddings"
        target_dir = model_cache_dir(kind_clean)

        yield {
            "status": "starting",
            "model": model_name,
            "kind": kind_clean,
            "target_dir": str(target_dir),
            "percent": 0.0,
        }

        # Use queue for cross-thread progress streaming
        loop = asyncio.get_running_loop()
        progress_queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()

        def _download_worker() -> None:
            try:
                import tqdm
                from huggingface_hub import snapshot_download

                class _ProgressTqdm(tqdm.tqdm):
                    def __init__(self, *args: Any, **kwargs: Any) -> None:
                        super().__init__(*args, **kwargs)
                        self._report(0)

                    def update(self, n: int = 1) -> bool | None:
                        res = super().update(n)
                        self._report(n)
                        return res

                    def _report(self, increment: int) -> None:
                        total = getattr(self, "total", None) or 1
                        completed = getattr(self, "n", 0)
                        pct = round(max(0.0, min(100.0, completed / total * 100.0)), 2)
                        loop.call_soon_threadsafe(
                            progress_queue.put_nowait,
                            {
                                "status": "downloading",
                                "model": model_name,
                                "completed": completed,
                                "total": total,
                                "percent": pct,
                                "description": getattr(self, "desc", "") or "",
                            },
                        )

                snapshot_download(
                    repo_id=model_name,
                    cache_dir=target_dir,
                    tqdm_class=_ProgressTqdm,
                )

                loop.call_soon_threadsafe(
                    progress_queue.put_nowait,
                    {
                        "status": "completed",
                        "model": model_name,
                        "percent": 100.0,
                    },
                )
            except Exception as exc:  # noqa: BLE001
                loop.call_soon_threadsafe(
                    progress_queue.put_nowait,
                    {
                        "status": "error",
                        "model": model_name,
                        "error": str(exc),
                        "percent": 0.0,
                    },
                )
            finally:
                loop.call_soon_threadsafe(progress_queue.put_nowait, None)

        worker_task = asyncio.to_thread(_download_worker)
        asyncio.create_task(worker_task)

        while True:
            event = await progress_queue.get()
            if event is None:
                break
            yield event
            if event.get("status") in ("completed", "error"):
                break

    # ── Embedding Compatibility Validation ────────────────────────────────────

    def check_embedding_impact(
        self,
        config_id: str,
        new_model_name: str | None = None,
        new_dimensions: int | None = None,
        new_provider_type: str | None = None,
        session: Session | None = None,
    ) -> EmbeddingImpactResult:
        """Check if changing an EmbeddingConfiguration affects any existing KnowledgeBase

        and return a structured warning listing which knowledge bases will require reindexing.
        """
        def _check(s: Session) -> EmbeddingImpactResult:
            config = s.get(EmbeddingConfiguration, config_id)
            if config is None:
                raise KeyError(f"Embedding configuration not found: {config_id}")

            target_provider = (
                str(new_provider_type).strip().lower()
                if new_provider_type is not None
                else config.provider_type
            )
            target_model = (
                str(new_model_name).strip()
                if new_model_name is not None
                else config.model_name
            )
            target_dimensions = (
                int(new_dimensions)
                if new_dimensions is not None
                else config.dimensions
            )

            # Query all knowledge bases pointing to this configuration
            kbs = list(
                s.scalars(
                    select(KnowledgeBase).where(
                        KnowledgeBase.embedding_config_id == config_id
                    )
                ).all()
            )

            affected: list[AffectedKnowledgeBase] = []
            requires_reindex_overall = False

            for kb in kbs:
                metadata = s.scalar(
                    select(IndexMetadata).where(IndexMetadata.knowledge_base_id == kb.id)
                )

                # Has anything been indexed?
                has_content = (
                    (kb.chunk_count > 0)
                    or (kb.document_count > 0)
                    or (metadata is not None and metadata.vector_count > 0)
                )

                if not has_content:
                    affected.append(
                        AffectedKnowledgeBase(
                            knowledge_base_id=kb.id,
                            name=kb.name,
                            document_count=kb.document_count,
                            chunk_count=kb.chunk_count,
                            current_provider=config.provider_type,
                            current_model=config.model_name,
                            current_dimensions=config.dimensions,
                            needs_reindex=False,
                            reason="Knowledge base is empty; no indexed vectors exist.",
                        )
                    )
                    continue

                if metadata is not None:
                    dim_mismatch = metadata.embedding_dimensions != target_dimensions
                    model_mismatch = (
                        metadata.embedding_model_name != target_model
                        or metadata.embedding_provider != target_provider
                    )

                    if dim_mismatch:
                        kb_reindex = True
                        reason = (
                            f"Vector dimension mismatch: index was generated with "
                            f"{metadata.embedding_dimensions}d vectors, but the new configuration "
                            f"produces {target_dimensions}d vectors."
                        )
                    elif model_mismatch:
                        kb_reindex = True
                        reason = (
                            f"Embedding model changed from '{metadata.embedding_provider}/{metadata.embedding_model_name}' "
                            f"to '{target_provider}/{target_model}'. Existing vector embeddings are semantically incompatible."
                        )
                    elif metadata.needs_reindex:
                        kb_reindex = True
                        reason = "Knowledge base was previously flagged as needing reindexing."
                    else:
                        kb_reindex = False
                        reason = "Configuration matches current vector store index metadata."

                    if kb_reindex:
                        requires_reindex_overall = True

                    affected.append(
                        AffectedKnowledgeBase(
                            knowledge_base_id=kb.id,
                            name=kb.name,
                            document_count=kb.document_count,
                            chunk_count=kb.chunk_count,
                            current_provider=metadata.embedding_provider,
                            current_model=metadata.embedding_model_name,
                            current_dimensions=metadata.embedding_dimensions,
                            needs_reindex=kb_reindex,
                            reason=reason,
                        )
                    )
                else:
                    # KB has chunks but no index metadata record
                    requires_reindex_overall = True
                    affected.append(
                        AffectedKnowledgeBase(
                            knowledge_base_id=kb.id,
                            name=kb.name,
                            document_count=kb.document_count,
                            chunk_count=kb.chunk_count,
                            current_provider=config.provider_type,
                            current_model=config.model_name,
                            current_dimensions=config.dimensions,
                            needs_reindex=True,
                            reason="Knowledge base has chunks but missing index metadata.",
                        )
                    )

            warning = None
            if requires_reindex_overall:
                reindex_count = sum(1 for a in affected if a.needs_reindex)
                warning = (
                    f"Updating this embedding configuration will invalidate the index of "
                    f"{reindex_count} knowledge base(s). Reindexing will be required before "
                    f"retrieval can function."
                )

            return EmbeddingImpactResult(
                config_id=config_id,
                new_provider=target_provider,
                new_model=target_model,
                new_dimensions=target_dimensions,
                requires_reindex=requires_reindex_overall,
                affected_knowledge_bases=affected,
                warning_message=warning,
            )

        if session is not None:
            return _check(session)
        with session_scope(self._session_factory) as s:
            return _check(s)

    def apply_embedding_impact(
        self,
        impact_result: EmbeddingImpactResult,
        session: Session | None = None,
    ) -> int:
        """Mark needs_reindex = True on all affected knowledge bases."""
        def _apply(s: Session) -> int:
            marked = 0
            for item in impact_result.affected_knowledge_bases:
                if not item.needs_reindex:
                    continue

                metadata = s.scalar(
                    select(IndexMetadata).where(
                        IndexMetadata.knowledge_base_id == item.knowledge_base_id
                    )
                )
                if metadata is not None:
                    metadata.needs_reindex = True
                    marked += 1

                kb = s.get(KnowledgeBase, item.knowledge_base_id)
                if kb is not None:
                    kb.status = "needs_reindex"

            s.flush()
            return marked

        if session is not None:
            return _apply(session)
        with session_scope(self._session_factory) as s:
            return _apply(s)

    def check_kb_switch_impact(
        self,
        knowledge_base_id: str,
        target_embedding_config_id: str,
        session: Session | None = None,
    ) -> EmbeddingImpactResult:
        """Check compatibility when assigning a different EmbeddingConfiguration to a KnowledgeBase."""
        def _check_kb(s: Session) -> EmbeddingImpactResult:
            kb = s.get(KnowledgeBase, knowledge_base_id)
            if kb is None:
                raise KeyError(f"Knowledge base not found: {knowledge_base_id}")

            target_config = s.get(EmbeddingConfiguration, target_embedding_config_id)
            if target_config is None:
                raise KeyError(f"Target embedding configuration not found: {target_embedding_config_id}")

            metadata = s.scalar(
                select(IndexMetadata).where(IndexMetadata.knowledge_base_id == kb.id)
            )

            has_content = (
                (kb.chunk_count > 0)
                or (kb.document_count > 0)
                or (metadata is not None and metadata.vector_count > 0)
            )

            if not has_content or metadata is None:
                affected = [
                    AffectedKnowledgeBase(
                        knowledge_base_id=kb.id,
                        name=kb.name,
                        document_count=kb.document_count,
                        chunk_count=kb.chunk_count,
                        current_provider=target_config.provider_type,
                        current_model=target_config.model_name,
                        current_dimensions=target_config.dimensions,
                        needs_reindex=False,
                        reason="Knowledge base is empty; switching configuration does not require reindexing.",
                    )
                ]
                return EmbeddingImpactResult(
                    config_id=target_embedding_config_id,
                    new_provider=target_config.provider_type,
                    new_model=target_config.model_name,
                    new_dimensions=target_config.dimensions,
                    requires_reindex=False,
                    affected_knowledge_bases=affected,
                    warning_message=None,
                )

            dim_mismatch = metadata.embedding_dimensions != target_config.dimensions
            model_mismatch = (
                metadata.embedding_model_name != target_config.model_name
                or metadata.embedding_provider != target_config.provider_type
            )

            if dim_mismatch:
                kb_reindex = True
                reason = (
                    f"Vector dimension mismatch: current index has {metadata.embedding_dimensions}d vectors, "
                    f"but target configuration uses {target_config.dimensions}d vectors."
                )
            elif model_mismatch:
                kb_reindex = True
                reason = (
                    f"Embedding model mismatch: current index uses '{metadata.embedding_provider}/{metadata.embedding_model_name}', "
                    f"but target configuration uses '{target_config.provider_type}/{target_config.model_name}'."
                )
            elif metadata.needs_reindex:
                kb_reindex = True
                reason = "Knowledge base was previously flagged as needing reindexing."
            else:
                kb_reindex = False
                reason = "Target configuration matches current index vectors."

            affected = [
                AffectedKnowledgeBase(
                    knowledge_base_id=kb.id,
                    name=kb.name,
                    document_count=kb.document_count,
                    chunk_count=kb.chunk_count,
                    current_provider=metadata.embedding_provider,
                    current_model=metadata.embedding_model_name,
                    current_dimensions=metadata.embedding_dimensions,
                    needs_reindex=kb_reindex,
                    reason=reason,
                )
            ]

            warning = (
                f"Switching knowledge base '{kb.name}' to this embedding configuration will "
                f"require a complete reindex before retrieval can function."
                if kb_reindex
                else None
            )

            return EmbeddingImpactResult(
                config_id=target_embedding_config_id,
                new_provider=target_config.provider_type,
                new_model=target_config.model_name,
                new_dimensions=target_config.dimensions,
                requires_reindex=kb_reindex,
                affected_knowledge_bases=affected,
                warning_message=warning,
            )

        if session is not None:
            return _check_kb(session)
        with session_scope(self._session_factory) as s:
            return _check_kb(s)


# ── Global Singleton & Factory ────────────────────────────────────────────────

_model_studio_service: ModelStudioService | None = None


def get_model_studio_service() -> ModelStudioService:
    """Return the application-wide ModelStudioService singleton."""
    global _model_studio_service
    if _model_studio_service is None:
        _model_studio_service = ModelStudioService()
    return _model_studio_service


def reset_model_studio_service() -> None:
    """Reset the singleton instance (for test isolation)."""
    global _model_studio_service
    _model_studio_service = None


__all__ = [
    "AffectedKnowledgeBase",
    "EmbeddingImpactResult",
    "ModelStudioService",
    "OllamaDetectionResult",
    "PRESET_CLOUD_LLM_NAME",
    "PRESET_LOCAL_EMBEDDING_NAME",
    "PRESET_LOCAL_LLM_NAME",
    "PRESET_LOCAL_RERANKER_NAME",
    "SAMPLE_PASSAGE_IRRELEVANT",
    "SAMPLE_PASSAGE_RELEVANT",
    "SAMPLE_RERANK_QUERY",
    "get_model_studio_service",
    "reset_model_studio_service",
]
