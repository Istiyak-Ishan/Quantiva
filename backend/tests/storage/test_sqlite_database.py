"""Verification tests for local AppData paths and the SQLite ORM schema."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from sqlalchemy import delete, func, inspect, select, text
from sqlalchemy.exc import IntegrityError

from app.config import ensure_data_directories, resolve_quantiva_data_dir
from app.storage.database import (
    create_session_factory,
    create_sqlite_engine,
    initialize_database,
)
from app.storage.models import (
    ApplicationProfile,
    Base,
    Conversation,
    Document,
    DocumentChunk,
    EmbeddingConfiguration,
    IndexMetadata,
    KnowledgeBase,
    Message,
    ModelConfiguration,
    ProcessingJob,
    RetrievalConfiguration,
    SchemaVersion,
)

EXPECTED_TABLES = {
    "schema_versions",
    "application_profiles",
    "knowledge_bases",
    "documents",
    "document_chunks",
    "conversations",
    "messages",
    "model_configurations",
    "embedding_configurations",
    "retrieval_configurations",
    "processing_jobs",
    "index_metadata",
}


@pytest.fixture
def local_database(tmp_path: Path):
    data_dir = tmp_path / "quantiva-data"
    engine = create_sqlite_engine(data_dir)
    initialize_database(engine)
    factory = create_session_factory(engine)
    try:
        yield data_dir, engine, factory
    finally:
        engine.dispose()


def test_bootstrap_override_and_required_directory_creation(tmp_path: Path):
    bootstrap_dir = tmp_path / "bootstrap-home"
    bootstrap_dir.mkdir()
    bootstrap = bootstrap_dir / "bootstrap.json"
    custom_data = tmp_path / "custom-data"
    bootstrap.write_text(
        json.dumps({"data_dir": str(custom_data)}),
        encoding="utf-8",
    )

    resolved = resolve_quantiva_data_dir(
        bootstrap_path=bootstrap,
        environ={},
    )
    paths = ensure_data_directories(resolved)

    assert resolved == custom_data.resolve()
    assert paths.database_file.is_file()
    assert all(path.is_dir() for path in paths.required_directories)
    assert paths.original_documents == custom_data / "documents" / "original"
    assert paths.processed_documents == custom_data / "documents" / "processed"
    assert paths.attachments == custom_data / "documents" / "attachments"
    assert paths.embedding_models == custom_data / "models" / "embeddings"
    assert paths.reranker_models == custom_data / "models" / "rerankers"


def test_environment_override_wins_over_bootstrap(tmp_path: Path):
    bootstrap = tmp_path / "bootstrap.json"
    bootstrap.write_text(
        json.dumps({"data_dir": str(tmp_path / "bootstrap-data")}),
        encoding="utf-8",
    )
    env_data = tmp_path / "environment-data"

    resolved = resolve_quantiva_data_dir(
        bootstrap_path=bootstrap,
        environ={"QUANTIVA_DATA_DIR": str(env_data)},
    )

    assert resolved == env_data.resolve()


def test_all_twelve_tables_are_created(local_database):
    _data_dir, engine, _factory = local_database

    assert len(Base.metadata.tables) == 12
    assert set(inspect(engine).get_table_names()) == EXPECTED_TABLES


def test_wal_and_connection_pragmas_apply_on_every_connection(local_database):
    _data_dir, engine, _factory = local_database

    with engine.connect() as connection:
        assert connection.execute(text("PRAGMA journal_mode")).scalar_one().lower() == "wal"
        assert connection.execute(text("PRAGMA synchronous")).scalar_one() == 1
        assert connection.execute(text("PRAGMA busy_timeout")).scalar_one() == 10_000
        assert connection.execute(text("PRAGMA foreign_keys")).scalar_one() == 1
        assert connection.execute(text("PRAGMA temp_store")).scalar_one() == 2
        assert connection.execute(text("PRAGMA cache_size")).scalar_one() == -64_000

    # Dispose the pool so this assertion necessarily observes a fresh DBAPI
    # connection and verifies that the Engine connect listener runs again.
    engine.dispose()
    with engine.connect() as connection:
        assert connection.execute(text("PRAGMA journal_mode")).scalar_one().lower() == "wal"
        assert connection.execute(text("PRAGMA foreign_keys")).scalar_one() == 1
        assert connection.execute(text("PRAGMA busy_timeout")).scalar_one() == 10_000


def test_crud_across_all_twelve_tables(local_database):
    _data_dir, _engine, factory = local_database
    with factory() as session:
        schema = SchemaVersion(
            version=1,
            app_version="1.0.0",
            description="Initial local schema",
        )
        profile = ApplicationProfile(
            display_name="Local Analyst",
            workspace_preferences_json={"theme": "dark"},
        )
        model_config = ModelConfiguration(
            name="Local LLM",
            provider_type="ollama",
            model_name="llama3.2",
            has_api_key=False,
            capabilities_json={"function_calling": False},
            is_default=True,
        )
        embedding_config = EmbeddingConfiguration(
            name="Local Embeddings",
            provider_type="sentence_transformers",
            model_name="intfloat/multilingual-e5-small",
            dimensions=384,
            batch_size=32,
            is_default=True,
        )
        retrieval_config = RetrievalConfiguration(
            name="Balanced Hybrid",
            search_mode="hybrid",
            dense_top_k=20,
            sparse_top_k=20,
            final_top_k=8,
            is_default=True,
        )
        session.add_all(
            [
                schema,
                profile,
                model_config,
                embedding_config,
                retrieval_config,
            ]
        )
        session.flush()

        knowledge_base = KnowledgeBase(
            name="Financial Reports",
            description="Local annual reports",
            embedding_config_id=embedding_config.id,
            retrieval_config_id=retrieval_config.id,
            document_count=1,
            chunk_count=1,
        )
        session.add(knowledge_base)
        session.flush()

        document = Document(
            knowledge_base_id=knowledge_base.id,
            filename="annual-report.pdf",
            original_rel_path="documents/original/annual-report.pdf",
            processed_rel_path="documents/processed/annual-report.json",
            file_type="pdf",
            file_size_bytes=1024,
            sha256_hash="a" * 64,
            page_count=10,
            chunk_count=1,
            status="ready",
            metadata_json={"author": "Quantiva"},
        )
        session.add(document)
        session.flush()

        chunk = DocumentChunk(
            document_id=document.id,
            knowledge_base_id=knowledge_base.id,
            chunk_index=0,
            page_number=1,
            section_title="Summary",
            is_table=False,
            content="Revenue increased to 42 million.",
            content_hash="b" * 64,
            char_count=32,
            metadata_json={"language": "en"},
        )
        conversation = Conversation(
            knowledge_base_id=knowledge_base.id,
            title="Revenue review",
            model_config_id=model_config.id,
            is_pinned=True,
        )
        session.add_all([chunk, conversation])
        session.flush()

        message = Message(
            conversation_id=conversation.id,
            role="assistant",
            content="Revenue increased [1].",
            citations_json=[{"chunk_id": chunk.id}],
            retrieved_chunks_json=[{"chunk_id": chunk.id}],
            token_usage_json={"total_tokens": 25},
            latency_ms=12,
            provider_used="ollama",
            model_used="llama3.2",
        )
        job = ProcessingJob(
            job_type="ingestion",
            knowledge_base_id=knowledge_base.id,
            document_id=document.id,
            status="completed",
            stage="index_sparse",
            progress_percent=100.0,
            message="Complete",
        )
        index_metadata = IndexMetadata(
            knowledge_base_id=knowledge_base.id,
            vector_collection_name="kb-financial-reports",
            embedding_provider="sentence_transformers",
            embedding_model_name="intfloat/multilingual-e5-small",
            embedding_dimensions=384,
            vector_count=1,
            sparse_count=1,
        )
        session.add_all([message, job, index_metadata])
        session.commit()

        model_classes = (
            SchemaVersion,
            ApplicationProfile,
            KnowledgeBase,
            Document,
            DocumentChunk,
            Conversation,
            Message,
            ModelConfiguration,
            EmbeddingConfiguration,
            RetrievalConfiguration,
            ProcessingJob,
            IndexMetadata,
        )
        for model_class in model_classes:
            count = session.scalar(select(func.count()).select_from(model_class))
            assert count == 1, model_class.__name__

        loaded = session.get(Document, document.id)
        assert loaded is not None
        assert loaded.metadata_json == {"author": "Quantiva"}
        loaded.status = "archived"
        profile.workspace_preferences_json = {"theme": "light"}
        session.commit()
        assert session.get(Document, document.id).status == "archived"
        assert session.get(ApplicationProfile, profile.profile_id).workspace_preferences_json == {
            "theme": "light"
        }


def test_foreign_keys_reject_orphans_and_cascade_descendants(local_database):
    _data_dir, _engine, factory = local_database
    with factory() as session:
        orphan = DocumentChunk(
            document_id="missing-document",
            knowledge_base_id="missing-kb",
            chunk_index=0,
            content="orphan",
            content_hash="c" * 64,
            char_count=6,
        )
        session.add(orphan)
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()

        kb = KnowledgeBase(name="Cascade KB")
        session.add(kb)
        session.flush()
        document = Document(
            knowledge_base_id=kb.id,
            filename="source.txt",
            original_rel_path="documents/original/source.txt",
            file_type="txt",
            file_size_bytes=10,
            sha256_hash="d" * 64,
        )
        conversation = Conversation(
            knowledge_base_id=kb.id,
            title="Cascade conversation",
        )
        session.add_all([document, conversation])
        session.flush()
        session.add_all(
            [
                DocumentChunk(
                    document_id=document.id,
                    knowledge_base_id=kb.id,
                    chunk_index=0,
                    content="cascade evidence",
                    content_hash="e" * 64,
                    char_count=16,
                ),
                Message(
                    conversation_id=conversation.id,
                    role="user",
                    content="question",
                ),
                ProcessingJob(
                    job_type="ingestion",
                    knowledge_base_id=kb.id,
                    document_id=document.id,
                ),
                IndexMetadata(
                    knowledge_base_id=kb.id,
                    vector_collection_name="cascade-collection",
                    embedding_provider="mock",
                    embedding_model_name="mock-embedding",
                    embedding_dimensions=8,
                ),
            ]
        )
        session.commit()

        session.execute(delete(KnowledgeBase).where(KnowledgeBase.id == kb.id))
        session.commit()

        for model_class in (
            KnowledgeBase,
            Document,
            DocumentChunk,
            Conversation,
            Message,
            ProcessingJob,
            IndexMetadata,
        ):
            assert session.scalar(select(func.count()).select_from(model_class)) == 0
