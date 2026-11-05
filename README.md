# Quantiva — Enterprise AI Document Agent Platform

<div align="center">

![Quantiva Banner](https://placehold.co/1200x400/0a0a0a/60a5fa?text=Quantiva+%E2%80%94+AI+Document+Agent)

**Upload any document. Ask any question. Get cited, streaming answers.**

[![Python](https://img.shields.io/badge/Python-3.11-3776ab?style=flat&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-16-black?style=flat&logo=next.js&logoColor=white)](https://nextjs.org)
[![TypeScript](https://img.shields.io/badge/TypeScript-5-3178c6?style=flat&logo=typescript&logoColor=white)](https://typescriptlang.org)
[![Gemini](https://img.shields.io/badge/Gemini-1.5_Flash-4285f4?style=flat&logo=google&logoColor=white)](https://ai.google.dev)
[![Qdrant](https://img.shields.io/badge/Qdrant-vector_DB-dc244c?style=flat)](https://qdrant.tech)
[![Docker](https://img.shields.io/badge/Docker-compose-2496ed?style=flat&logo=docker&logoColor=white)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[Live Demo](https://quantiva.railway.app) · [API Docs](https://quantiva.railway.app/docs) · [Langfuse Traces](#observability)

</div>

---

## ✨ What It Does

Quantiva is a **production-grade AI document intelligence platform** built from scratch over 16 days. Upload a PDF, Excel sheet, CSV, or Word document, ask a complex question, and watch a Gemini 1.5 Flash–powered reasoning agent stream a cited answer token by token — complete with source page references and confidence scores.

```
User uploads Annual Report.pdf
  → Celery worker: parse → chunk → embed → index in Qdrant + Postgres

User asks: "What were the Q3 revenue figures and how do they compare to Q2?"
  → ReasoningAgent runs hybrid search (vector + BM25 + cross-encoder rerank)
  → Retrieves top-8 chunks, reasons over evidence
  → Streams answer token by token via SSE
  → Returns JSON: { answer, citations: [{page: 12, filename: "..."}], confidence: 0.94 }
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Next.js 14 Frontend                       │
│  FileUploader (react-dropzone) │ ChatWindow │ MessageBubble      │
│  Zustand stores │ SSE EventSource │ Glassmorphism dark UI        │
└───────────────────────┬─────────────────────────────────────────┘
                        │ REST + SSE (HTTP)
┌───────────────────────▼─────────────────────────────────────────┐
│                    FastAPI Backend (Python 3.11)                  │
│                                                                   │
│  /api/v1/documents  →  Celery Worker  →  Ingestion Pipeline      │
│    parse (PDF/Excel/CSV/DOCX)  →  chunk  →  embed  →  index     │
│                                                                   │
│  /api/v1/chat/message/stream  →  ReasoningAgent                  │
│    HybridRetriever (Qdrant + BM25 + CrossEncoder rerank)         │
│    GeminiClient (1.5 Flash)  →  token stream                     │
│    SemanticCache (Redis + cosine similarity)                      │
│    GuardrailsService (prompt injection + PII detection)          │
└──────┬──────────┬──────────┬──────────┬──────────┬──────────────┘
       │          │          │          │          │
   Postgres    Redis      Qdrant     S3/MinIO  Langfuse
   (metadata) (cache/    (vectors)  (files)   (LLM traces)
              sessions)
                                                    │
                                              Prometheus → Grafana
```

---

## 🚀 Quick Start

### Prerequisites
- Docker Desktop
- Google AI API key ([get one free](https://aistudio.google.com/app/apikey))

### One-command startup

```bash
git clone https://github.com/yourusername/quantiva
cd quantiva

# Add your API key
echo "GOOGLE_API_KEY=your-key-here" > .env
echo "SECRET_KEY=$(openssl rand -hex 32)" >> .env

# Start everything
docker compose up -d

# Run database migrations
docker compose exec backend alembic upgrade head
```

That's it. The full stack starts in ~45 seconds:

| Service | URL | Description |
|---------|-----|-------------|
| **Frontend** | http://localhost:3000 | Next.js chat UI |
| **API Docs** | http://localhost:8000/docs | FastAPI Swagger |
| **Langfuse** | http://localhost:3001 | LLM trace viewer |
| **Grafana** | http://localhost:3002 | Metrics dashboard |
| **Prometheus** | http://localhost:9090 | Metrics scraper |

---

## 🛠️ Tech Stack

### Backend
| Layer | Technology | Why |
|-------|-----------|-----|
| API | **FastAPI** | Async, auto-docs, typed |
| LLM | **Gemini 1.5 Flash** | Fast, cheap, 1M token context |
| Vector DB | **Qdrant** | HNSW index, payload filtering |
| Keyword search | **BM25 (rank_bm25)** | Complements dense retrieval |
| Reranking | **CrossEncoder (ms-marco)** | Precision over recall |
| Semantic cache | **Redis + cosine sim** | Skip LLM for repeated queries |
| Background jobs | **Celery + Redis** | Async document processing |
| Object storage | **S3 / MinIO** | Raw file storage |
| Database | **PostgreSQL + SQLAlchemy** | Document & chunk metadata |
| Auth | **JWT (python-jose)** | Stateless authentication |
| Observability | **Langfuse + Prometheus** | LLM traces + metrics |

### Frontend
| Layer | Technology | Why |
|-------|-----------|-----|
| Framework | **Next.js 14 App Router** | SSR, nested layouts, RSC |
| Language | **TypeScript** | Type safety end-to-end |
| Styling | **Tailwind CSS v4** | Utility-first, fast iteration |
| State | **Zustand** | 1KB, no boilerplate |
| Streaming | **EventSource (SSE)** | Native browser API, no WebSocket overhead |
| File upload | **react-dropzone** | Drag-and-drop + XHR progress |
| Components | **shadcn/ui** | Owns the code, fully customizable |

---

## 📁 Project Structure

```
quantiva/
├── backend/
│   ├── app/
│   │   ├── api/v1/          # REST endpoints (auth, documents, chat, eval)
│   │   ├── core/            # Security, middleware, metrics, dependencies
│   │   ├── db/              # SQLAlchemy models + Alembic migrations
│   │   ├── schemas/         # Pydantic request/response schemas
│   │   ├── services/
│   │   │   ├── agents/      # ReasoningAgent, DataAnalystAgent
│   │   │   ├── ingestion/   # PDF/Excel/CSV/DOCX parsers + chunker
│   │   │   ├── llm/         # GeminiClient (with Langfuse tracing)
│   │   │   ├── memory/      # Session memory + semantic cache
│   │   │   ├── retrieval/   # HybridRetriever + Qdrant + BM25 + reranker
│   │   │   └── tools/       # Python executor, chart generator
│   │   └── workers/         # Celery tasks (document ingestion)
│   ├── mcp_server/          # MCP server (Claude Desktop / Antigravity)
│   └── tests/               # pytest test suite
├── frontend/
│   └── src/
│       ├── app/             # Next.js App Router pages
│       ├── components/      # ChatWindow, MessageBubble, FileUploader
│       ├── lib/             # Axios API client
│       └── store/           # Zustand stores (chat, documents)
├── monitoring/
│   ├── prometheus.yml       # Scrape config
│   └── grafana/             # Auto-provisioned dashboards
├── .github/workflows/       # CI/CD (lint → test → Docker push)
├── docker-compose.yml       # Full stack (10 services)
└── railway.json             # Railway.app deployment config
```

---

## 🔬 Observability

### Langfuse — LLM Traces (http://localhost:3001)

Every LLM call creates a structured trace:

```
trace: "user-question" (session_id, user_id)
├── span: "retrieval"  (query, chunk_count, latency_ms)
├── generation: "llm-call" (model, input_tokens, output_tokens, cost_usd, latency)
└── span: "tool:python_executor" (code, output)
```

Login: `admin@quantiva.ai` / `changeme`

### Grafana — Metrics (http://localhost:3002)

Pre-built dashboard includes:
- Request rate by endpoint
- P95 latency
- LLM token consumption (input vs output, by model)
- Semantic cache hit rate
- Active SSE streaming sessions
- 5xx error rate

Login: `admin` / `admin`

---

## 🤖 MCP Server

Quantiva exposes its document intelligence as an MCP server, making it usable from Claude Desktop, Antigravity IDE, or any MCP-compatible agent.

```bash
# Tools available to any MCP host:
search_documents(query, user_id, doc_ids?, top_k?)  → chunks[]
summarize_document(doc_id, user_id, style?)          → summary
ask_question(question, user_id, doc_ids?, session_id?) → {answer, citations, confidence}
```

**Add to Antigravity IDE (settings.json):**
```json
{
  "mcp.servers": [{
    "name": "quantiva",
    "command": "python -m mcp_server.server",
    "cwd": "/path/to/quantiva/backend",
    "env": {
      "QUANTIVA_API_URL": "http://localhost:8000/api/v1",
      "QUANTIVA_TOKEN": "your-jwt-token"
    }
  }]
}
```

---

## ☁️ Deploy to Railway

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/template)

```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up

# Add environment variables in the Railway dashboard:
# GOOGLE_API_KEY, SECRET_KEY, DATABASE_URL (auto-injected), REDIS_URL (auto-injected)
```

Railway auto-provisions Postgres and Redis. Point your custom domain at the generated Railway URL.

---

## 🧪 Running Tests

```bash
cd backend
pip install -r requirements.txt pytest pytest-asyncio

# Unit + integration tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=app --cov-report=html
```

---

## 📊 Performance Characteristics

| Metric | Value |
|--------|-------|
| Document ingestion (10-page PDF) | ~8 seconds |
| First token latency (cold, no cache) | ~1.2 seconds |
| First token latency (cache hit) | ~40ms |
| Semantic cache hit rate (repeated queries) | ~70% |
| Hybrid retrieval (vector + BM25 + rerank) | ~180ms |
| Concurrent streaming sessions supported | 100+ |

---

## 📖 API Reference

Full OpenAPI docs at `/docs` when running. Key endpoints:

```
POST /api/v1/auth/register        Create account
POST /api/v1/auth/login           Get JWT token

POST /api/v1/documents/upload     Upload document (multipart)
GET  /api/v1/documents/{id}/status  SSE: indexing progress

POST /api/v1/chat/sessions        Create chat session
GET  /api/v1/chat/message/stream  SSE: stream answer tokens
GET  /api/v1/chat/sessions/{id}/history  Conversation history

GET  /health                      Deep health check
GET  /metrics                     Prometheus metrics
```

---

## 🗺️ Day-by-Day Build Log

| Days | What was built |
|------|---------------|
| 1–2 | Project scaffold, Docker, Postgres, SQLAlchemy models, Alembic migrations |
| 3–4 | Document ingestion pipeline: PDF/Excel/CSV/DOCX parsers, chunker |
| 5–6 | Qdrant vector store, embedding service (Google text-embedding-004) |
| 7–8 | Hybrid retriever: BM25 + vector + CrossEncoder reranking |
| 9–10 | GeminiClient, ReasoningAgent, DataAnalystAgent, tool registry |
| 11–12 | Redis session memory, semantic cache, context budget manager |
| 13 | Auth (JWT), rate limiting, guardrails (prompt injection + PII) |
| 14 | Eval framework (RAGAS-style: faithfulness, relevance, groundedness) |
| 15 | Next.js 14 frontend: FileUploader, ChatWindow, streaming MessageBubble |
| 16 | Langfuse tracing, Prometheus metrics, MCP server, CI/CD, Railway deploy |

---

## 📄 License

MIT — see [LICENSE](LICENSE).

---

<div align="center">
Built end-to-end in 16 days as a portfolio project demonstrating production AI engineering.
<br>
<strong>Stack: FastAPI · Gemini · Qdrant · Redis · Celery · Next.js · TypeScript · Docker · Langfuse · MCP</strong>
</div>

