"""
tests/providers/test_provider_registry.py
──────────────────────────────────────────────────────────────────────────────
Tests for the ProviderRegistry and ApplicationContext wiring.

Verifies:
1. LLM Provider dynamic swapping (GEMINI -> OLLAMA -> OPENAI_COMPATIBLE -> ANTHROPIC)
   without breaking the unified LLMProvider interface contract.
2. Independent Model Spaces: Changing LLM config does not affect active
   Embedding or Reranker providers.
3. Per-KnowledgeBase EmbeddingProvider instantiation from stored
   EmbeddingConfiguration and IndexMetadata (locking model space & dimensions).
4. Reranker provider selection and NoOp fallbacks.
5. ApplicationContext singleton lifecycle, provider accessors, and shutdown.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from app.core.application_context import (
    ApplicationContext,
    get_application_context,
    get_provider_registry,
    reset_application_context,
)
from app.providers.embeddings.gemini_embedding import GeminiEmbeddingProvider
from app.providers.embeddings.ollama import OllamaEmbeddingProvider
from app.providers.embeddings.sentence_transformers import (
    SentenceTransformersEmbeddingProvider,
)
from app.providers.interfaces.embedding import EmbeddingProvider
from app.providers.interfaces.llm import LLMProvider
from app.providers.interfaces.reranker import RerankerProvider
from app.providers.interfaces.types import ProviderType
from app.providers.llm.anthropic import AnthropicLLMProvider
from app.providers.llm.gemini import GeminiLLMProvider
from app.providers.llm.ollama import OllamaLLMProvider
from app.providers.llm.openai_compatible import OpenAICompatibleLLMProvider
from app.providers.registry import (
    EmbeddingConfig,
    EmbeddingConfiguration,
    IndexMetadata,
    LLMConfig,
    ProviderRegistry,
    RerankerConfig,
)
from app.providers.rerankers.api import APIRerankerProvider
from app.providers.rerankers.cross_encoder import (
    LocalCrossEncoderReranker,
    NoOpReranker,
)


@pytest.fixture(autouse=True)
def clean_application_context():
    """Ensure ApplicationContext is reset before and after every test."""
    reset_application_context()
    yield
    reset_application_context()


class TestLLMProviderSwapping:
    """Verifies seamless swapping of LLM providers while preserving LLMProvider contract."""

    def test_switch_gemini_to_ollama_to_openai_compatible(self):
        """Proof that swapping LLMConfig dynamically updates the active LLMProvider."""
        registry = ProviderRegistry()

        # 1. Configure GEMINI
        gemini_cfg = LLMConfig(
            provider=ProviderType.GEMINI,
            model="gemini-3.5-flash-lite",
            api_key="mock-gemini-key",
        )
        registry.set_llm_config(gemini_cfg)

        provider_1 = registry.get_llm_provider()
        assert isinstance(provider_1, LLMProvider)
        assert isinstance(provider_1, GeminiLLMProvider)
        assert provider_1.provider_name() == "gemini"
        assert provider_1.model_name() == "gemini-3.5-flash-lite"
        # Cached reference check
        assert registry.get_llm_provider() is provider_1

        # 2. Swap to OLLAMA
        ollama_cfg = LLMConfig(
            provider=ProviderType.OLLAMA,
            model="qwen2.5:7b",
            base_url="http://127.0.0.1:11434",
        )
        registry.set_llm_config(ollama_cfg)

        provider_2 = registry.get_llm_provider()
        assert isinstance(provider_2, LLMProvider)
        assert isinstance(provider_2, OllamaLLMProvider)
        assert provider_2.provider_name() == "ollama"
        assert provider_2.model_name() == "qwen2.5:7b"
        assert provider_2 is not provider_1
        assert registry.get_llm_provider() is provider_2

        # 3. Swap to OPENAI_COMPATIBLE (e.g. vLLM or LM Studio)
        openai_cfg = LLMConfig(
            provider=ProviderType.OPENAI_COMPATIBLE,
            model="meta-llama/Llama-3.1-8B-Instruct",
            base_url="http://127.0.0.1:8000/v1",
            api_key="vllm-key",
        )
        registry.set_llm_config(openai_cfg)

        provider_3 = registry.get_llm_provider()
        assert isinstance(provider_3, LLMProvider)
        assert isinstance(provider_3, OpenAICompatibleLLMProvider)
        assert provider_3.provider_name() == "openai_compatible"
        assert provider_3.model_name() == "meta-llama/Llama-3.1-8B-Instruct"
        assert provider_3 is not provider_2
        assert registry.get_llm_provider() is provider_3

        # 4. Swap to ANTHROPIC
        anthropic_cfg = LLMConfig(
            provider=ProviderType.ANTHROPIC,
            model="claude-3-5-sonnet-20241022",
            api_key="sk-ant-test",
        )
        registry.set_llm_config(anthropic_cfg)

        provider_4 = registry.get_llm_provider()
        assert isinstance(provider_4, LLMProvider)
        assert isinstance(provider_4, AnthropicLLMProvider)
        assert provider_4.provider_name() == "anthropic"
        assert provider_4.model_name() == "claude-3-5-sonnet-20241022"
        assert provider_4 is not provider_3

    def test_interface_contract_preserved_across_all_providers(self):
        """Ensure all instantiated providers expose uniform callable signatures."""
        registry = ProviderRegistry()

        configs = [
            LLMConfig(provider=ProviderType.GEMINI, model="gemini-1.5-pro", api_key="test"),
            LLMConfig(provider=ProviderType.OLLAMA, model="llama3.2"),
            LLMConfig(provider=ProviderType.OPENAI_COMPATIBLE, model="gpt-4o-mini", api_key="sk-test"),
            LLMConfig(provider=ProviderType.ANTHROPIC, model="claude-3-haiku-20240307", api_key="sk-ant"),
        ]

        for cfg in configs:
            provider = registry.create_llm_provider(cfg)
            assert isinstance(provider, LLMProvider)
            assert callable(getattr(provider, "generate", None))
            assert callable(getattr(provider, "stream", None))
            assert callable(getattr(provider, "test_connection", None))
            assert callable(getattr(provider, "provider_name", None))
            assert callable(getattr(provider, "model_name", None))
            assert callable(getattr(provider, "get_capabilities", None))


class TestIndependentModelSpaces:
    """Verifies that LLM, Embedding, and Reranker providers operate in isolated spaces."""

    def test_switching_llm_preserves_embedding_and_reranker(self):
        """Changing the active LLM must NOT re-instantiate or mutate Embedding or Reranker."""
        registry = ProviderRegistry()

        # Setup independent trio: Ollama LLM + SentenceTransformers Embedding + CrossEncoder Reranker
        registry.set_llm_config(LLMConfig(provider=ProviderType.OLLAMA, model="llama3.2"))
        registry.set_embedding_config(EmbeddingConfig(
            provider=ProviderType.SENTENCE_TRANSFORMERS,
            model="intfloat/multilingual-e5-small",
            dimensions=384,
        ))
        registry.set_reranker_config(RerankerConfig(
            provider=ProviderType.CROSS_ENCODER,
            model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        ))

        initial_llm = registry.get_llm_provider()
        initial_embedding = registry.get_embedding_provider()
        initial_reranker = registry.get_reranker_provider()

        assert isinstance(initial_llm, OllamaLLMProvider)
        assert isinstance(initial_embedding, SentenceTransformersEmbeddingProvider)
        assert isinstance(initial_reranker, LocalCrossEncoderReranker)

        # Switch LLM to Gemini
        registry.set_llm_config(LLMConfig(
            provider=ProviderType.GEMINI,
            model="gemini-3.5-flash-lite",
            api_key="mock-key",
        ))

        new_llm = registry.get_llm_provider()
        persisted_embedding = registry.get_embedding_provider()
        persisted_reranker = registry.get_reranker_provider()

        # LLM changed
        assert isinstance(new_llm, GeminiLLMProvider)
        assert new_llm is not initial_llm

        # Embedding & Reranker remained identical cached instances
        assert persisted_embedding is initial_embedding
        assert persisted_reranker is initial_reranker


class TestPerKnowledgeBaseEmbedding:
    """Verifies locked per-KnowledgeBase embedding instantiation and dimension consistency."""

    def test_instantiate_from_embedding_configuration(self):
        registry = ProviderRegistry()

        kb_config_1 = EmbeddingConfiguration(
            provider=ProviderType.SENTENCE_TRANSFORMERS,
            model="intfloat/multilingual-e5-base",
            dimensions=768,
        )
        provider_1 = registry.get_knowledge_base_embedding_provider(kb_config_1)
        assert isinstance(provider_1, SentenceTransformersEmbeddingProvider)
        assert provider_1.model_name() == "intfloat/multilingual-e5-base"

        # Caching returns identical instance
        provider_1_again = registry.get_knowledge_base_embedding_provider(kb_config_1)
        assert provider_1_again is provider_1

        # Second KB with different model/dimensions
        kb_config_2 = EmbeddingConfiguration(
            provider=ProviderType.OLLAMA,
            model="nomic-embed-text",
            dimensions=768,
            base_url="http://127.0.0.1:11434",
        )
        provider_2 = registry.get_knowledge_base_embedding_provider(kb_config_2)
        assert isinstance(provider_2, OllamaEmbeddingProvider)
        assert provider_2.model_name() == "nomic-embed-text"
        assert provider_2 is not provider_1

    def test_instantiate_from_index_metadata(self):
        registry = ProviderRegistry()

        meta = IndexMetadata(
            knowledge_base_id="kb-12345",
            embedding_provider="gemini",
            embedding_model_name="gemini-embedding-001",
            embedding_dimensions=768,
        )

        provider = registry.get_knowledge_base_embedding_provider(meta)
        assert isinstance(provider, GeminiEmbeddingProvider)
        assert provider.model_name() == "gemini-embedding-001"
        assert provider.get_dimensions() == 768


class TestRerankerSelection:
    """Verifies reranker factory behavior (local, remote, and noop)."""

    def test_cross_encoder_reranker(self):
        registry = ProviderRegistry()
        cfg = RerankerConfig(
            provider=ProviderType.CROSS_ENCODER,
            model="cross-encoder/ms-marco-MiniLM-L-6-v2",
            enabled=True,
        )
        reranker = registry.create_reranker_provider(cfg)
        assert isinstance(reranker, LocalCrossEncoderReranker)
        assert reranker.model_name() == "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def test_noop_reranker_when_disabled(self):
        registry = ProviderRegistry()
        cfg = RerankerConfig(
            provider=ProviderType.CROSS_ENCODER,
            enabled=False,
        )
        reranker = registry.create_reranker_provider(cfg)
        assert isinstance(reranker, NoOpReranker)
        assert reranker.provider_name() == "none"

    def test_cohere_api_reranker(self):
        registry = ProviderRegistry()
        cfg = RerankerConfig(
            provider="cohere",
            model="rerank-v3.5",
            api_key="mock-cohere-key",
            enabled=True,
        )
        reranker = registry.create_reranker_provider(cfg)
        assert isinstance(reranker, APIRerankerProvider)
        assert reranker.model_name() == "rerank-v3.5"


class TestApplicationContextLifecycle:
    """Verifies ApplicationContext singleton, vector client lifecycle, and shutdown."""

    def test_singleton_application_context_and_registry(self):
        ctx1 = get_application_context()
        ctx2 = get_application_context()
        assert ctx1 is ctx2

        reg1 = get_provider_registry()
        assert reg1 is ctx1.provider_registry

    def test_vector_client_singleton_lifecycle(self):
        """Verifies Invariant 5: singleton vector client lifecycle management."""
        ctx = get_application_context()
        assert ctx.get_vector_client() is None

        mock_vector_client = MagicMock()
        ctx.set_vector_client(mock_vector_client)
        assert ctx.get_vector_client() is mock_vector_client

        # Clean shutdown closes the client
        ctx.shutdown()
        mock_vector_client.close.assert_called_once()
        assert ctx.get_vector_client() is None

    def test_custom_factory_registration(self):
        """Verifies that plugins or tests can register custom provider factories."""
        registry = ProviderRegistry()
        mock_llm = MagicMock(spec=LLMProvider)
        mock_llm.provider_name.return_value = "custom_engine"
        mock_llm.model_name.return_value = "custom-v1"

        registry.register_custom_llm_factory("custom_engine", lambda cfg: mock_llm)

        provider = registry.create_llm_provider(LLMConfig(
            provider="custom_engine",
            model="custom-v1",
        ))
        assert provider is mock_llm
