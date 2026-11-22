# Quantiva

Document ingestion and cited question answering with FastAPI, Celery, PostgreSQL,
Redis, Qdrant, S3, and Next.js 16.4.5 / React 19.

## Current status

The application is under audit-driven repair. See [REPAIR_PROGRESS.md](REPAIR_PROGRESS.md)
for verified findings, changes, exact test results, remaining acceptance checks,
and estimated API usage. The original audit is retained unchanged.

**Tested:** backend API/security regressions, real local Qdrant filtering, PDF
parsing, task-owned worker resources, actual Google generation/embedding access,
and a synthetic PDF → real embeddings → local Qdrant → real cited-agent answer.
Frontend lint, TypeScript, production builds and six Chromium production-browser
contract tests have passed during repair; those browser tests intercept API responses.

**Not yet established:** clean PostgreSQL migration/drift acceptance, container
build/startup, live Redis/S3/Celery integration, external Langfuse delivery, and
deployed full-stack/browser acceptance. Do not treat unit tests or the provider
smoke script as proof that deployment works. No published performance benchmarks.

## Local setup

Requires Docker Desktop's Linux engine and a Google API key with access to the
configured models. The current defaults were checked against Google's model catalog:

- Chat: `GEMINI_CHAT_MODEL=gemini-3.5-flash-lite`
- Text embeddings: `EMBEDDING_MODEL=gemini-embedding-001`, 768 normalized dimensions
- Existing retired-model/random-vector collections are **not** reused or deleted.

1. If root `.env` does **not** exist, copy `.env.example` to `.env`. If it exists,
   edit only missing configuration; never overwrite it with an example.
2. Supply `GOOGLE_API_KEY`, a random `SECRET_KEY`, an independent random
   `METRICS_TOKEN`, and `GRAFANA_ADMIN_PASSWORD`. Generate secrets with
   `python -c "import secrets; print(secrets.token_urlsafe(48))"`; do not reuse them.
3. Back up any existing PostgreSQL database before upgrading.
4. Run `docker compose config --quiet`, then `docker compose up --build -d`.
   The one-shot `migrate` service applies additive migrations before API startup.
5. Check `docker compose ps`, `http://localhost:8000/ready`, and worker logs.
6. Open `http://localhost:3000/register`, register, upload a small UTF-8 CSV or
   text PDF, wait for readiness, ask a question, then reload to verify history.

| Service | Local URL |
|---|---|
| Browser | http://localhost:3000 |
| API docs | http://localhost:8000/docs |
| Liveness / readiness | http://localhost:8000/health · /ready |
| Prometheus | http://localhost:9091 |
| Grafana | http://localhost:3002 |

The Compose stack is local-development infrastructure, bound to loopback. It is
not a public production deployment template. No source bind mounts or dev servers
are used in its application images. Langfuse is optional and external; the former
incomplete local Langfuse topology is no longer started.

### Without Docker

Use Python 3.11 and Node 22. Install CPU-only PyTorch and
`backend/requirements.txt`; set actual PostgreSQL/Redis/Qdrant/S3 URLs using
`backend/.env.example` as a reference. Run, from `backend/`:

```sh
alembic upgrade head
uvicorn app.main:app --host 127.0.0.1 --port 8000
# Separate terminal; native Windows Celery may need --pool=solo for development.
celery -A app.workers.celery_app worker --loglevel=info
```

From `frontend/`, run `npm ci` and `npm run dev`. By default the frontend proxies
`/api/v1/*` to `http://localhost:8000`; `BACKEND_URL` changes this destination at
Next.js build time. Same-origin proxying avoids third-party cookie requirements.
Bearer clients may access the API directly. Use explicit `CORS_ORIGINS` JSON
origins if browser requests are cross-origin; wildcard origins are rejected.

## Workflow and security contract

- Login/register issue typed access and rotating refresh tokens. Browser credentials
  use HttpOnly cookies; API/MCP clients use `Authorization: Bearer <access token>`.
  URL credentials and `mock-token` are rejected. Existing legacy tokens require login.
- Access checks validate active users and revocable login sessions. Cookie-authenticated
  mutations require a configured trusted `Origin`. Logout revokes the login session.
- All document/session/history/evaluation routes are owner scoped. Qdrant filters
  require both owner and document IDs. Unknown/foreign resources return 404; malformed
  IDs return 422. No fallback sessions or substitute conversations.
- PDF, DOCX, XLSX, CSV and UTF-8 text upload is bounded to 25 MiB by default,
  with signature/Office structure checks. Set `CLAMAV_HOST` to enable a fail-closed
  ClamAV INSTREAM scan; without it this is **not antivirus scanning**.
- Worker failures remain failures, not ready documents with substitute vectors.
  Full chunk text is stored for BM25 and citations. Chunk sizes are **characters**,
  not tokens; duplicate PDF table/text blocks are collapsed while table chunks are
  preserved. The backend image installs Tesseract and English language data;
  scanned-PDF acceptance on real infrastructure remains outstanding.
- API responses include nosniff, frame-deny, referrer and permissions policies;
  a deny-all CSP is applied outside the OpenAPI UI, and HSTS is production-only.
- SSE carries actual model output, buffered at sentence/newline boundaries for
  PII checks. Final success is sent only after database persistence. Failures use
  structured SSE error events; partial failed output is not finalized as success.
- `confidence` currently means an **embedding-based source-support heuristic**,
  not calibrated correctness probability. With no sources it is zero/flagged.
- Token counts are provider metadata. `cost_usd` is `null` unless current model
  rates are configured with `LLM_INPUT_USD_PER_MILLION` and
  `LLM_OUTPUT_USD_PER_MILLION`; cache hits consume zero generation tokens.
- Semantic cache keys include owner, corpus generation, model/dimensions and
  conversation context. Document invalidation is implemented.
- Deletion removes active-index vectors, cache and raw storage before the row.
  Cleanup failure leaves a non-queryable `deleting` document; retry DELETE.
  Legacy vector collections require separately approved cleanup.

### Guardrails and tools

`PII_ENGINE=patterns` deterministically redacts email, SSN, phone and card-like
identifiers. It does **not** claim person-name/address detection. To enable
Presidio NLP, explicitly provision `SPACY_MODEL` (default `en_core_web_sm`) and
set `PII_ENGINE=presidio`; missing capability fails closed. No model downloads
occur on application import.

RestrictedPython calculations run in a killable child with isolated output,
restricted builtins and bounded input/output. Linux also applies process resource
limits. RestrictedPython is **not a complete security sandbox**; production requires
container/network/resource isolation. Charts use Matplotlib and signed S3 URLs.
Messages persist a stable owner-scoped `/api/v1/charts/{owner}/{name}` path;
loading a historical chart re-signs it, so old conversations survive URL expiry.
Set `S3_PUBLIC_ENDPOINT_URL` to a browser-reachable endpoint, not the internal host.

`/health` is process liveness only; `/ready` checks PostgreSQL, Redis, Qdrant and S3,
returns 503 when degraded, and never exposes driver exception strings. `/metrics`
requires `METRICS_TOKEN` and is disabled without one. HTTP/LLM/cache/stream/retrieval
metrics are instrumented. The ingestion counter is emitted by the worker and needs
Prometheus multiprocess aggregation to appear in the API scrape; that aggregation
is not yet a complete observability solution.

## Existing data and migration safety

Read [docs/OPERATIONS.md](docs/OPERATIONS.md) before updating an existing instance.
Do not drop collections, reset migrations, run `docker compose down -v`, or replace
secrets as a troubleshooting shortcut. No repair command has been applied to your
existing database in this session.

Legacy chunks cannot recover text lost beyond 200 characters. After backing up,
an authenticated owner can explicitly call `POST /api/v1/documents/{id}/reindex`.
This reparses the retained source into the new model/dimension-specific collection;
it does not delete the legacy collection. No bulk reindex runs automatically.

## Tests

```sh
cd backend
pip install -r requirements-test.txt
HF_HUB_OFFLINE=1 python -m pytest tests/ -q
python -m ruff check app/ --select F,E9
python -m mypy app/ --explicit-package-bases --ignore-missing-imports --no-strict-optional

# Explicit opt-in API usage; synthetic text/PDF only, no existing data mutation:
python scripts/check_providers.py --live
HF_HUB_OFFLINE=1 python scripts/check_local_rag.py --live
```

The PostgreSQL migration/drift test runs only when `TEST_POSTGRES_URL` explicitly
targets a disposable database named `quantiva_test_*`. It otherwise skips rather
than pretending migration acceptance passed. Browser contract tests use intercepted
API responses and must not be described as a live infrastructure E2E test.

```sh
cd frontend
npm ci
npm run lint
npm run build
npx playwright install chromium
npm run test:e2e
```

## API and MCP

Authoritative request/response schemas are in running OpenAPI `/docs`.

```text
POST /api/v1/auth/register, /auth/login, /auth/refresh, /auth/logout
GET  /api/v1/auth/me
POST /api/v1/documents/upload
GET  /api/v1/documents, /documents/{id}, /documents/{id}/chunks
GET  /api/v1/documents/{id}/status                 SSE
GET  /api/v1/documents/{id}/download               Owner-scoped original file
POST /api/v1/documents/{id}/reindex                Explicit retry/reindex
DELETE /api/v1/documents/{id}
GET  /api/v1/charts/{owner}/{name}                 Owner-scoped renewable chart
POST /api/v1/chat/sessions, /chat/message
GET  /api/v1/chat/sessions, /chat/sessions/{id}
GET  /api/v1/chat/sessions/{id}/history
GET  /api/v1/chat/message/stream                   Cookie or header auth; no token URL
DELETE /api/v1/chat/sessions/{id}
POST /api/v1/retrieval/search
POST /api/v1/eval/run?doc_id={id}                  Queued Celery evaluation
GET  /api/v1/eval/history/list, /eval/{id}
```

MCP: from `backend/`, run `python -m mcp_server.server` with `QUANTIVA_API_URL`
and a real `QUANTIVA_TOKEN`. Identity is always the authenticated owner, not a tool
argument. Tools: `search_documents(query, doc_ids?, top_k?)`,
`summarize_document(doc_id, style?)`, `ask_question(question, doc_ids?, session_id?)`.
An existing session's document scope cannot be overridden silently.

Railway configuration is **backend-only**; external PostgreSQL/Redis/Qdrant/S3,
worker and frontend services must be configured separately. No automatic full-stack
Railway provisioning or live demo availability is claimed. The learning handbook
and interview walkthrough are historical design notes, not operational instructions.
