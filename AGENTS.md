# QUANTIVA DESKTOP MIGRATION — MANDATORY ENGINEERING RULES

You are working on **Quantiva**, a production RAG and Agentic Document Intelligence application being migrated from a cloud-dependent stack to a local-first Windows 10/11 desktop application (Tauri 2 + Next.js Frontend + Packaged FastAPI Backend + SQLite/FTS5 + Local Vector DB).

## Non-Negotiable Architectural Invariants
1. **Do Not Rewrite From Scratch**: Preserve existing working RAG intelligence, document parsing, hybrid search, RRF fusion, ReAct agent, Data Analyst agent, and citation logic. Refactor incrementally behind clean interfaces.
2. **Strict Provider Decoupling**: No file in `app/ingestion/`, `app/retrieval/`, `app/agents/`, or `app/api/` may directly import `google.generativeai`, `google.genai`, `openai`, `anthropic`, or `ollama`. All model calls MUST go through `app/providers/interfaces/` (`LLMProvider`, `EmbeddingProvider`, `RerankerProvider`).
3. **Independent Model Spaces**: LLM, Embedding, and Reranker configurations are strictly independent. Every `KnowledgeBase` locks its `embedding_provider`, `embedding_model_name`, and `embedding_dimensions` in `IndexMetadata`. Querying a knowledge base with a mismatched embedding dimension must raise a typed `EmbeddingDimensionMismatchError` before hitting the vector store.
4. **Zero Mandatory Cloud Infrastructure**: The application must run with zero external servers (no PostgreSQL, no Redis, no Celery, no S3/MinIO, no external Qdrant Docker container). All state lives under the configurable `QUANTIVA_DATA_DIR` (`%LOCALAPPDATA%/Quantiva` by default).
5. **Windows Path & File Locking Safety**:
   - Always use `pathlib.Path` for filesystem paths; never hardcode forward slashes `/` in string concatenation.
   - Never instantiate embedded `QdrantClient` or `LanceDB` connections per-request; always use the singleton lifecycle instance from `app/core/application_context.py`.
   - Always configure SQLite with `WAL` mode, `busy_timeout=10000`, and `foreign_keys=ON`.
6. **Security & Local Isolation**:
   - Backend binds strictly to `127.0.0.1` (never `0.0.0.0`).
   - All API endpoints (except `/api/v1/system/health`) require the `X-Quantiva-Token` header injected by the Tauri shell.
   - Cloud API keys must be stored via `app/security/credentials.py` (Windows Credential Manager via `keyring` with encrypted fallback), never in plaintext SQLite columns or log files.
7. **Verification Before Completion**: Never claim a task is complete without running the relevant `pytest` suite or build verification command and fixing all import, type, and runtime errors.

