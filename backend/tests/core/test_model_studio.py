"""
backend/tests/core/test_model_studio.py
────────────────────────────────────────────────────────────────────────────────
Tests for Model Studio Service & Compatibility Validation Engine (Prompt P4.2).
"""
from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from sqlalchemy import select

from app.core.model_studio import (
    AffectedKnowledgeBase,
    EmbeddingImpactResult,
    ModelStudioService,
    OllamaDetectionResult,
    PRESET_CLOUD_LLM_NAME,
    PRESET_LOCAL_EMBEDDING_NAME,
    PRESET_LOCAL_LLM_NAME,
    PRESET_LOCAL_RERANKER_NAME,
    SAMPLE_PASSAGE_IRRELEVANT,
    SAMPLE_PASSAGE_RELEVANT,
    SAMPLE_RERANK_QUERY,
)
from app.providers.interfaces.embedding import EmbeddingProvider
from app.providers.interfaces.llm import LLMProvider
from app.providers.interfaces.reranker import (
    RerankDocument,
    RerankerProvider,
    RerankResult,
)
from app.providers.interfaces.types import (
    ChatMessage,
    ConnectionTestResult,
    GenerationConfig,
    LLMResponse,
    ProviderCapabilities,
    ProviderType,
)
from app.providers.registry import (
    EmbeddingConfig,
    LLMConfig,
    ProviderRegistry,
    RerankerConfig,
)
from app.security.credentials import CredentialManager
from app.storage.database import (
    create_session_factory,
    create_sqlite_engine,
    initialize_database,
)
from app.storage.models import (
    Document,
    DocumentChunk,
    EmbeddingConfiguration,
    IndexMetadata,
    KnowledgeBase,
    ModelConfiguration,
    RetrievalConfiguration,
)


# ── Test Fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture
def service_env(tmp_path: Path):
    """Provides an isolated ModelStudioService backed by a clean SQLite database and local credential vault."""
    data_dir = tmp_path / "quantiva-test-data"
    data_dir.mkdir(parents=True, exist_ok=True)
    engine = create_sqlite_engine(data_dir)
    initialize_database(engine)
    factory = create_session_factory(engine)

    creds_dir = data_dir / "security"
    creds_dir.mkdir(parents=True, exist_ok=True)
    cred_manager = CredentialManager(
        service_name="QuantivaTestStudio",
        fallback_dir=creds_dir,
    )

    registry = ProviderRegistry()

    service = ModelStudioService(
        session_factory=factory,
        provider_registry=registry,
        credential_manager=cred_manager,
    )

    try:
        yield service, factory, cred_manager, registry, data_dir
    finally:
        engine.dispose()


# ── Dummy Test Providers ──────────────────────────────────────────────────────

class DummyLLM(LLMProvider):
    def __init__(self, config: LLMConfig):
        self._config = config

    def provider_name(self) -> str:
        return self._config.normalized_provider()

    def model_name(self) -> str:
        return self._config.model

    def get_capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_streaming=True,
            supports_function_calling=True,
            supports_structured_output=True,
            supports_vision=False,
            supports_system_prompt=True,
            max_context_window=8192,
            is_local=True,
        )

    async def generate(self, messages: list[ChatMessage], config=None, tools=None):
        return LLMResponse(content="pong", model=self.model_name(), provider=self.provider_name())

    async def stream(self, messages, config=None, tools=None):
        raise NotImplementedError

    async def test_connection(self) -> ConnectionTestResult:
        if "error" in self._config.model.lower():
            return ConnectionTestResult(
                success=False,
                latency_ms=15.0,
                provider=self.provider_name(),
                model=self.model_name(),
                error_code="ConnectionFailed",
                error_message="Mock connection failed",
            )
        return ConnectionTestResult(
            success=True,
            latency_ms=25.5,
            provider=self.provider_name(),
            model=self.model_name(),
            capabilities=self.get_capabilities(),
        )


class DummyEmbedding(EmbeddingProvider):
    def __init__(self, config: EmbeddingConfig):
        self._config = config
        self._dims = config.dimensions or 768

    def provider_name(self) -> str:
        return self._config.normalized_provider()

    def model_name(self) -> str:
        return self._config.model

    def get_dimensions(self) -> int:
        return self._dims

    async def embed_documents(self, texts, batch_size=32):
        return [[0.1] * self._dims for _ in texts]

    async def embed_query(self, query: str):
        return [0.1] * self._dims

    async def test_embedding(self) -> ConnectionTestResult:
        if "error" in self._config.model.lower():
            return ConnectionTestResult(
                success=False,
                latency_ms=10.0,
                provider=self.provider_name(),
                model=self.model_name(),
                detected_dimensions=None,
                error_code="EmbeddingFailed",
                error_message="Mock embedding failed",
            )
        return ConnectionTestResult(
            success=True,
            latency_ms=30.0,
            provider=self.provider_name(),
            model=self.model_name(),
            detected_dimensions=self._dims,
        )


class DummyReranker(RerankerProvider):
    def __init__(self, config: RerankerConfig, fail_ranking: bool = False):
        self._config = config
        self._fail_ranking = fail_ranking

    def provider_name(self) -> str:
        return self._config.normalized_provider()

    def model_name(self) -> str:
        return self._config.model

    async def rerank(self, query: str, documents: list[RerankDocument], top_k=5):
        if "error" in self._config.model.lower():
            raise RuntimeError("Mock reranker failure")

        if self._fail_ranking:
            # Reverse order so irrelevant ranks higher
            scores = [0.1, 0.9]
            ranked = list(reversed(documents))
        else:
            scores = [0.95, 0.05]
            ranked = documents

        results = []
        for rank, (doc, sc) in enumerate(zip(ranked[:top_k], scores)):
            results.append(
                RerankResult(
                    id=doc.id,
                    content=doc.content,
                    metadata=doc.metadata,
                    rerank_score=sc,
                    initial_score=doc.initial_score,
                    rank=rank,
                )
            )
        return results

    async def test_reranker(self) -> ConnectionTestResult:
        return ConnectionTestResult(
            success=True,
            latency_ms=12.0,
            provider=self.provider_name(),
            model=self.model_name(),
        )


# ── Test Suite ────────────────────────────────────────────────────────────────

class TestPresetSeeding:
    def test_seed_presets_creates_all_four_presets(self, service_env):
        service, factory, _, _, _ = service_env

        presets = service.seed_presets()
        assert "local_llm" in presets
        assert "cloud_llm" in presets
        assert "local_embedding" in presets
        assert "local_reranker" in presets

        with factory() as session:
            # 1. Local LLM (Ollama qwen2.5:7b)
            local_llm = session.scalar(
                select(ModelConfiguration).where(ModelConfiguration.name == PRESET_LOCAL_LLM_NAME)
            )
            assert local_llm is not None
            assert local_llm.provider_type == "ollama"
            assert local_llm.model_name == "qwen2.5:7b"
            assert local_llm.temperature == 0.2
            assert local_llm.max_tokens == 4096
            assert local_llm.is_default is True

            # 2. Cloud LLM (Gemini gemini-2.5-flash)
            cloud_llm = session.scalar(
                select(ModelConfiguration).where(ModelConfiguration.name == PRESET_CLOUD_LLM_NAME)
            )
            assert cloud_llm is not None
            assert cloud_llm.provider_type == "gemini"
            assert cloud_llm.model_name == "gemini-2.5-flash"
            assert cloud_llm.temperature == 0.2
            assert cloud_llm.max_tokens == 4096
            assert cloud_llm.is_default is False

            # 3. Local Embedding (Sentence Transformers multilingual-e5-base)
            local_embed = session.scalar(
                select(EmbeddingConfiguration).where(
                    EmbeddingConfiguration.name == PRESET_LOCAL_EMBEDDING_NAME
                )
            )
            assert local_embed is not None
            assert local_embed.provider_type == "sentence_transformers"
            assert local_embed.model_name == "intfloat/multilingual-e5-base"
            assert local_embed.dimensions == 768
            assert local_embed.is_default is True

            # 4. Local Reranker (Cross Encoder inside RetrievalConfiguration)
            local_reranker = session.scalar(
                select(RetrievalConfiguration).where(
                    RetrievalConfiguration.name == PRESET_LOCAL_RERANKER_NAME
                )
            )
            assert local_reranker is not None
            assert local_reranker.enable_reranking is True
            assert local_reranker.reranker_provider == "cross_encoder"
            assert local_reranker.reranker_model == "cross-encoder/ms-marco-MiniLM-L-6-v2"
            assert local_reranker.is_default is True

    def test_seed_presets_idempotent(self, service_env):
        service, factory, _, _, _ = service_env

        service.seed_presets()
        service.seed_presets()

        with factory() as session:
            llm_count = len(session.scalars(select(ModelConfiguration)).all())
            embed_count = len(session.scalars(select(EmbeddingConfiguration)).all())
            retrieval_count = len(session.scalars(select(RetrievalConfiguration)).all())

            assert llm_count == 2
            assert embed_count == 1
            assert retrieval_count == 1

    def test_seed_presets_preserves_custom_configs(self, service_env):
        service, factory, _, _, _ = service_env

        # Create user custom config first
        service.create_llm_config({
            "name": "Custom Claude",
            "provider_type": "anthropic",
            "model_name": "claude-3-5-sonnet",
            "is_default": True,
        })

        service.seed_presets()

        with factory() as session:
            custom = session.scalar(
                select(ModelConfiguration).where(ModelConfiguration.name == "Custom Claude")
            )
            assert custom is not None
            assert custom.is_default is True

            # Seeded local LLM should not usurp default if one already existed
            local_llm = session.scalar(
                select(ModelConfiguration).where(ModelConfiguration.name == PRESET_LOCAL_LLM_NAME)
            )
            assert local_llm is not None
            assert local_llm.is_default is False


class TestConfigurationCRUD:
    def test_llm_config_crud_and_credential_masking(self, service_env):
        service, factory, creds, _, _ = service_env

        # Create with API key
        created = service.create_llm_config({
            "name": "Production OpenAI",
            "provider_type": "openai",
            "model_name": "gpt-4o",
            "api_key": "sk-test-secret-key-12345",
            "temperature": 0.5,
            "max_tokens": 2048,
        })
        assert created.has_api_key is True
        assert created.credential_key_id is not None

        # Confirm key is in CredentialManager and NOT plaintext in SQLite
        stored_key = creds.get_api_key(created.credential_key_id)
        assert stored_key == "sk-test-secret-key-12345"

        with factory() as session:
            row = session.get(ModelConfiguration, created.id)
            assert row is not None
            # ModelConfiguration has no api_key column
            assert not hasattr(row, "api_key")

        # Update
        updated = service.update_llm_config(created.id, {
            "temperature": 0.7,
            "api_key": "sk-updated-key-99999",
        })
        assert updated.temperature == 0.7
        assert creds.get_api_key(created.credential_key_id) == "sk-updated-key-99999"

        # Delete removes key
        deleted = service.delete_llm_config(created.id)
        assert deleted is True
        assert creds.get_api_key(created.credential_key_id) is None

    def test_llm_set_default_toggles_and_syncs_registry(self, service_env):
        service, factory, _, registry, _ = service_env

        cfg1 = service.create_llm_config({
            "name": "LLM 1",
            "provider_type": "ollama",
            "model_name": "llama3.2",
            "is_default": True,
        })
        cfg2 = service.create_llm_config({
            "name": "LLM 2",
            "provider_type": "gemini",
            "model_name": "gemini-2.5-flash",
            "is_default": False,
        })

        assert cfg1.is_default is True
        assert registry.get_llm_config().model == "llama3.2"

        service.set_default_llm_config(cfg2.id)

        with factory() as session:
            c1 = session.get(ModelConfiguration, cfg1.id)
            c2 = session.get(ModelConfiguration, cfg2.id)
            assert c1.is_default is False
            assert c2.is_default is True

        assert registry.get_llm_config().model == "gemini-2.5-flash"

    def test_embedding_config_crud_and_default(self, service_env):
        service, factory, _, registry, _ = service_env

        embed1 = service.create_embedding_config({
            "name": "E5 Small",
            "provider_type": "sentence_transformers",
            "model_name": "intfloat/multilingual-e5-small",
            "dimensions": 384,
            "is_default": True,
        })
        embed2 = service.create_embedding_config({
            "name": "E5 Base",
            "provider_type": "sentence_transformers",
            "model_name": "intfloat/multilingual-e5-base",
            "dimensions": 768,
            "is_default": False,
        })

        assert embed1.is_default is True
        assert registry.get_embedding_config().dimensions == 384

        service.set_default_embedding_config(embed2.id)

        with factory() as session:
            e1 = session.get(EmbeddingConfiguration, embed1.id)
            e2 = session.get(EmbeddingConfiguration, embed2.id)
            assert e1.is_default is False
            assert e2.is_default is True

        assert registry.get_embedding_config().dimensions == 768

    def test_retrieval_and_reranker_crud(self, service_env):
        service, factory, _, registry, _ = service_env

        rc = service.create_retrieval_config({
            "name": "Hybrid with Reranker",
            "search_mode": "hybrid",
            "enable_reranking": True,
            "reranker_provider": "cross_encoder",
            "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "is_default": True,
        })

        assert rc.enable_reranking is True
        assert registry.get_reranker_config().model == "cross-encoder/ms-marco-MiniLM-L-6-v2"

        service.update_retrieval_config(rc.id, {
            "reranker_model": "cross-encoder/ms-marco-MiniLM-L-12-v2",
        })

        assert registry.get_reranker_config().model == "cross-encoder/ms-marco-MiniLM-L-12-v2"


class TestLiveConnectionTesting:
    @pytest.mark.asyncio
    async def test_llm_connection_testing_unsaved_inputs(self, service_env):
        service, _, _, registry, _ = service_env
        registry.register_custom_llm_factory("dummy_llm", lambda cfg: DummyLLM(cfg))

        payload = {
            "provider_type": "dummy_llm",
            "model_name": "dummy-chat-v1",
            "temperature": 0.2,
        }

        res = await service.test_llm_config(payload)
        assert res.success is True
        assert res.latency_ms > 0
        assert res.provider == "dummy_llm"
        assert res.model == "dummy-chat-v1"
        assert res.capabilities is not None
        assert res.capabilities.supports_function_calling is True

    @pytest.mark.asyncio
    async def test_llm_connection_testing_with_saved_config_id(self, service_env):
        service, _, creds, registry, _ = service_env
        registry.register_custom_llm_factory("dummy_llm", lambda cfg: DummyLLM(cfg))

        saved = service.create_llm_config({
            "name": "Saved Test LLM",
            "provider_type": "dummy_llm",
            "model_name": "dummy-chat-v1",
            "api_key": "secret-123",
        })

        res = await service.test_llm_config({"config_id": saved.id})
        assert res.success is True
        assert res.model == "dummy-chat-v1"

    @pytest.mark.asyncio
    async def test_llm_connection_testing_failure_handling(self, service_env):
        service, _, _, registry, _ = service_env
        registry.register_custom_llm_factory("dummy_llm", lambda cfg: DummyLLM(cfg))

        res = await service.test_llm_config({
            "provider_type": "dummy_llm",
            "model_name": "error-model",
        })
        assert res.success is False
        assert res.error_code == "ConnectionFailed"
        assert "Mock connection failed" in res.error_message

    @pytest.mark.asyncio
    async def test_embedding_connection_testing_dimension_detection(self, service_env):
        service, _, _, registry, _ = service_env
        registry.register_custom_embedding_factory("dummy_embed", lambda cfg: DummyEmbedding(cfg))

        payload = {
            "provider_type": "dummy_embed",
            "model_name": "dummy-embed-base",
            "dimensions": 768,
        }

        res = await service.test_embedding_config(payload)
        assert res.success is True
        assert res.detected_dimensions == 768

    @pytest.mark.asyncio
    async def test_reranker_connection_testing_passage_ranking_success(self, service_env):
        service, _, _, registry, _ = service_env
        registry.register_custom_reranker_factory(
            "dummy_reranker", lambda cfg: DummyReranker(cfg, fail_ranking=False)
        )

        res = await service.test_reranker_config({
            "provider_type": "dummy_reranker",
            "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        })
        assert res.success is True
        assert res.latency_ms > 0

    @pytest.mark.asyncio
    async def test_reranker_connection_testing_ranking_failure(self, service_env):
        service, _, _, registry, _ = service_env
        registry.register_custom_reranker_factory(
            "dummy_reranker", lambda cfg: DummyReranker(cfg, fail_ranking=True)
        )

        res = await service.test_reranker_config({
            "provider_type": "dummy_reranker",
            "model_name": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        })
        assert res.success is False
        assert res.error_code == "RankingValidationError"
        assert "did not score strictly higher" in res.error_message


class TestOllamaIntegration:
    @pytest.mark.asyncio
    async def test_detect_ollama_not_installed(self, service_env):
        service, _, _, _, _ = service_env

        with (
            patch("app.providers.llm.ollama.OllamaRuntimeManager.is_daemon_running", new_callable=AsyncMock) as mock_daemon,
            patch("pathlib.Path.is_file", return_value=False),
            patch("shutil.which", return_value=None),
        ):
            mock_daemon.return_value = False
            res = await service.detect_ollama()
            assert res.is_installed is False
            assert res.is_running is False
            assert res.binary_path is None

    @pytest.mark.asyncio
    async def test_detect_ollama_binary_found_not_running(self, service_env, tmp_path):
        service, _, _, _, _ = service_env
        mock_exe = tmp_path / "ollama.exe"
        mock_exe.touch()

        with (
            patch("app.providers.llm.ollama.OllamaRuntimeManager.is_daemon_running", new_callable=AsyncMock) as mock_daemon,
            patch.dict(os.environ, {"LOCALAPPDATA": str(tmp_path)}),
            patch("pathlib.Path.is_file", side_effect=lambda: True),
            patch("shutil.which", return_value=str(mock_exe)),
        ):
            mock_daemon.return_value = False
            res = await service.detect_ollama()
            assert res.is_installed is True
            assert res.is_running is False
            assert res.binary_path is not None

    @pytest.mark.asyncio
    async def test_list_and_pull_ollama_models(self, service_env):
        service, _, _, _, _ = service_env

        fake_models = [
            {"name": "qwen2.5:7b", "size_bytes": 4700000000, "family": "qwen2", "parameter_size": "7B", "quantization_level": "Q4_K_M"}
        ]

        async def fake_pull_stream(model):
            yield {"status": "downloading", "completed": 50, "total": 100, "percent": 50.0}
            yield {"status": "success", "completed": 100, "total": 100, "percent": 100.0}

        with (
            patch("app.providers.llm.ollama.OllamaRuntimeManager.list_installed_models", new_callable=AsyncMock) as mock_list,
            patch("app.providers.llm.ollama.OllamaRuntimeManager.pull_model_stream", side_effect=fake_pull_stream),
        ):
            mock_list.return_value = fake_models

            models = await service.list_ollama_models()
            assert len(models) == 1
            assert models[0]["name"] == "qwen2.5:7b"

            events = []
            async for ev in service.pull_ollama_model("qwen2.5:7b"):
                events.append(ev)

            assert len(events) == 2
            assert events[0]["percent"] == 50.0
            assert events[1]["percent"] == 100.0

    @pytest.mark.asyncio
    async def test_download_huggingface_model_stream(self, service_env):
        service, _, _, _, _ = service_env

        with patch("huggingface_hub.snapshot_download") as mock_download:
            mock_download.return_value = "/tmp/models"

            events = []
            async for ev in service.download_huggingface_model("intfloat/multilingual-e5-base", kind="embeddings"):
                events.append(ev)

            assert len(events) >= 2
            assert events[0]["status"] == "starting"
            assert events[-1]["status"] == "completed"
            assert events[-1]["percent"] == 100.0


class TestEmbeddingCompatibilityValidation:
    def test_embedding_impact_no_knowledge_bases(self, service_env):
        service, factory, _, _, _ = service_env
        config = service.create_embedding_config({
            "name": "E5 Base",
            "provider_type": "sentence_transformers",
            "model_name": "intfloat/multilingual-e5-base",
            "dimensions": 768,
        })

        impact = service.check_embedding_impact(config.id, new_dimensions=384)
        assert impact.requires_reindex is False
        assert len(impact.affected_knowledge_bases) == 0

    def test_embedding_impact_empty_knowledge_base(self, service_env):
        service, factory, _, _, _ = service_env
        config = service.create_embedding_config({
            "name": "E5 Base",
            "provider_type": "sentence_transformers",
            "model_name": "intfloat/multilingual-e5-base",
            "dimensions": 768,
        })

        with factory() as session:
            kb = KnowledgeBase(
                id=str(uuid.uuid4()),
                name="Empty Workspace",
                embedding_config_id=config.id,
                document_count=0,
                chunk_count=0,
            )
            session.add(kb)
            session.commit()

        impact = service.check_embedding_impact(config.id, new_dimensions=384)
        assert impact.requires_reindex is False
        assert len(impact.affected_knowledge_bases) == 1
        assert impact.affected_knowledge_bases[0].needs_reindex is False

    def test_embedding_impact_dimension_change(self, service_env):
        service, factory, _, _, _ = service_env
        config = service.create_embedding_config({
            "name": "E5 Base",
            "provider_type": "sentence_transformers",
            "model_name": "intfloat/multilingual-e5-base",
            "dimensions": 768,
        })

        with factory() as session:
            kb_id = str(uuid.uuid4())
            kb = KnowledgeBase(
                id=kb_id,
                name="Financial Reports",
                embedding_config_id=config.id,
                document_count=5,
                chunk_count=120,
            )
            meta = IndexMetadata(
                id=str(uuid.uuid4()),
                knowledge_base_id=kb_id,
                vector_collection_name=f"kb_{kb_id}",
                embedding_provider="sentence_transformers",
                embedding_model_name="intfloat/multilingual-e5-base",
                embedding_dimensions=768,
                vector_count=120,
                sparse_count=120,
                needs_reindex=False,
            )
            session.add(kb)
            session.add(meta)
            session.commit()

        # Changing dimensions from 768 to 384
        impact = service.check_embedding_impact(
            config.id,
            new_dimensions=384,
            new_model_name="intfloat/multilingual-e5-small",
        )
        assert impact.requires_reindex is True
        assert len(impact.affected_knowledge_bases) == 1
        aff = impact.affected_knowledge_bases[0]
        assert aff.needs_reindex is True
        assert "dimension mismatch" in aff.reason.lower()
        assert "768d" in aff.reason and "384d" in aff.reason

        # Apply impact
        applied = service.apply_embedding_impact(impact)
        assert applied == 1

        with factory() as session:
            meta_reloaded = session.scalar(
                select(IndexMetadata).where(IndexMetadata.knowledge_base_id == kb_id)
            )
            assert meta_reloaded.needs_reindex is True

    def test_embedding_impact_model_change_same_dimensions(self, service_env):
        service, factory, _, _, _ = service_env
        config = service.create_embedding_config({
            "name": "E5 Base",
            "provider_type": "sentence_transformers",
            "model_name": "intfloat/multilingual-e5-base",
            "dimensions": 768,
        })

        with factory() as session:
            kb_id = str(uuid.uuid4())
            kb = KnowledgeBase(
                id=kb_id,
                name="Research Papers",
                embedding_config_id=config.id,
                document_count=3,
                chunk_count=80,
            )
            meta = IndexMetadata(
                id=str(uuid.uuid4()),
                knowledge_base_id=kb_id,
                vector_collection_name=f"kb_{kb_id}",
                embedding_provider="sentence_transformers",
                embedding_model_name="intfloat/multilingual-e5-base",
                embedding_dimensions=768,
                vector_count=80,
                sparse_count=80,
                needs_reindex=False,
            )
            session.add(kb)
            session.add(meta)
            session.commit()

        # Changing model name to BGE Base (same 768 dimensions)
        impact = service.check_embedding_impact(
            config.id,
            new_model_name="BAAI/bge-base-en-v1.5",
            new_dimensions=768,
        )
        assert impact.requires_reindex is True
        aff = impact.affected_knowledge_bases[0]
        assert aff.needs_reindex is True
        assert "semantically incompatible" in aff.reason.lower()

    def test_embedding_impact_identical_config_no_reindex(self, service_env):
        service, factory, _, _, _ = service_env
        config = service.create_embedding_config({
            "name": "E5 Base",
            "provider_type": "sentence_transformers",
            "model_name": "intfloat/multilingual-e5-base",
            "dimensions": 768,
        })

        with factory() as session:
            kb_id = str(uuid.uuid4())
            kb = KnowledgeBase(
                id=kb_id,
                name="Technical Specs",
                embedding_config_id=config.id,
                document_count=2,
                chunk_count=50,
            )
            meta = IndexMetadata(
                id=str(uuid.uuid4()),
                knowledge_base_id=kb_id,
                vector_collection_name=f"kb_{kb_id}",
                embedding_provider="sentence_transformers",
                embedding_model_name="intfloat/multilingual-e5-base",
                embedding_dimensions=768,
                vector_count=50,
                sparse_count=50,
                needs_reindex=False,
            )
            session.add(kb)
            session.add(meta)
            session.commit()

        impact = service.check_embedding_impact(config.id)
        assert impact.requires_reindex is False
        aff = impact.affected_knowledge_bases[0]
        assert aff.needs_reindex is False

    def test_check_kb_switch_impact(self, service_env):
        service, factory, _, _, _ = service_env
        cfg1 = service.create_embedding_config({
            "name": "E5 Base",
            "provider_type": "sentence_transformers",
            "model_name": "intfloat/multilingual-e5-base",
            "dimensions": 768,
        })
        cfg2 = service.create_embedding_config({
            "name": "E5 Small",
            "provider_type": "sentence_transformers",
            "model_name": "intfloat/multilingual-e5-small",
            "dimensions": 384,
        })

        with factory() as session:
            kb_id = str(uuid.uuid4())
            kb = KnowledgeBase(
                id=kb_id,
                name="Switch Test KB",
                embedding_config_id=cfg1.id,
                document_count=1,
                chunk_count=20,
            )
            meta = IndexMetadata(
                id=str(uuid.uuid4()),
                knowledge_base_id=kb_id,
                vector_collection_name=f"kb_{kb_id}",
                embedding_provider="sentence_transformers",
                embedding_model_name="intfloat/multilingual-e5-base",
                embedding_dimensions=768,
                vector_count=20,
                sparse_count=20,
                needs_reindex=False,
            )
            session.add(kb)
            session.add(meta)
            session.commit()

        impact = service.check_kb_switch_impact(kb_id, cfg2.id)
        assert impact.requires_reindex is True
        assert len(impact.affected_knowledge_bases) == 1
        assert impact.affected_knowledge_bases[0].needs_reindex is True
        assert "dimension mismatch" in impact.affected_knowledge_bases[0].reason.lower()
