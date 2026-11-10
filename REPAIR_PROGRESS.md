# Quantiva repair progress

Updated: 2026-11-09. This is an execution record, not a production-readiness claim.

## Completed increments (latest status overrides the initial verification table)

### Latest verification snapshot

- **Final backend validation: 139 passed, 1 skipped, 1 deprecation warning** (15.47s; 140 collected). The skipped test is the explicitly gated clean-PostgreSQL migration/drift acceptance test, not a pass. Final Ruff F/E9 correctness checks and mypy (**86 source files**) passed; updated Compose syntax also passed with process-local placeholders, without replacing any stored secret.
- **Final frontend checks:** Next.js **16.4.5 production build passed**, lint passed, production browser contract suite **6 passed** (2.8s), production dependency audit **0 vulnerabilities**. Browser tests run the actual standalone production server, but intercept backend API responses. They do not establish live full-stack acceptance.
- Actual synthetic Google provider check: **passed**, 13 generation tokens and a 768-dimensional embedding.
- Actual synthetic local RAG check: initial run exposed raw-dictionary citation assignment and failed. Converted retrieved citations back through Pydantic validation and added a regression. Rerun **passed**: one PDF chunk, one real retrieval tool call, one citation, **2,964 generation tokens**. No existing user data involved.
- Final live **streaming** local-RAG check also **passed** with the typed SDK declarations and task-local retriever: one PDF chunk, one retrieval tool call, one citation, one safety-buffered output event, **2,932 generation tokens**. This tests actual Google streaming through the agent, not a word-splitting simulation. Source-support heuristic was 1.0 for the synthetic fact; this is not calibrated confidence.
- API/worker/cross-tenant/stream/tool regressions are tested with SQLite, local Qdrant and explicit external-service seams. Real PostgreSQL, Redis, S3 and Celery broker acceptance remains outstanding.
- Production browser contract suite: **6 passed**, running the built Next.js 16.4.5 standalone server in Chromium. API responses are intercepted, so this is not full deployed-infrastructure E2E. The prior logout failure was diagnosed as Next's development overlay intercepting pointer events; the production runtime removes that dev-only obstruction without forced clicks. Added a standalone start script that includes static assets.
- Backend Ruff correctness checks **passed**. First mypy invocation failed module discovery; explicit package bases exposed 24 typing issues. Fixed the typing issues, used the SDK's typed content/tool declarations, separated the base agent loop from structured response specialization, and removed the redundant raw-citation assignment. Final result at that point: **no issues in 85 source files** (86 after the latest increment). CI mypy is now blocking.
- Frontend dependency audit found previously unaudited critical Next.js advisories. Updated Next.js and matching ESLint config from 16.3.8 to **16.4.5** and updated compatible follow-redirects/zod transitive dependencies. Latest npm report: **no Critical/High advisories; 8 Moderate development-tooling advisories remain**. No forced major downgrade applied.
- Discovered `.gitignore`'s generic `lib/` pattern hid `frontend/src/lib/`. Added narrow exceptions so the API client and utilities are included in future commits/builds. Existing user files were not discarded.
- Removed unused next-auth after confirming no source imports. Production `npm audit --omit=dev --audit-level=high`: **0 vulnerabilities**. The full development dependency tree still reports the 8 Moderate tooling advisories noted above.
- Final regressions also protect mixed-case legacy login without rewriting user email data, require explicit reindexing after model/dimension/collection changes before any provider request, and allow idempotent deletion when a failed first upload never created a vector collection. All passed in the final suite.

### Remaining Medium/Low closure increment (2026-11-09)

- **MED-03:** `documents_ingested_total` is now incremented by the ingestion worker after a successful commit. It still requires Prometheus multiprocess aggregation to appear in the API `/metrics` scrape; that limitation is documented rather than hidden.
- **MED-09:** Removed the unused `streamingCitations` state. Final SSE events and durable history already carry citations, confidence, chart, cache and cost; history uses backend `created_at`.
- **MED-10:** Added deterministic block deduplication. Exact duplicate text blocks are removed and plain-text copies fully represented by a structured table are dropped while table chunks are preserved. Chunk size remains an explicitly documented character choice.
- **MED-11:** Removed the unused synchronous `validate_and_retry` helper. Long histories now invoke the existing memory pager (LLM summary of omitted turns) behind the deterministic context budget; summary failure falls back to bounded history without fabricating content. `prompt_builder.py` remains a standalone, non-runtime prompt helper; it is not presented as wired into the tool-using agent.
- **MED-12:** Database echo is disabled and no runtime test-detection branches remain; only bounded structured logging is used.
- **LOW-01/LOW-03/LOW-05:** Backend and frontend containers run non-root with health/readiness checks and `npm ci`; added API security headers (nosniff, frame deny, referrer/permissions policies, deny-all CSP outside Swagger/ReDoc, production-only HSTS) with a regression test. `next-auth` remains removed and production dependency audit remains clean.
- **HIGH-14 / security checklist:** Added an optional, fail-closed ClamAV INSTREAM scanner (`CLAMAV_HOST`). Structural upload validation remains the default; without ClamAV configured this is still explicitly **not antivirus scanning**. Scanner protocol behavior is unit-tested; a live clamd service was not available.
- **HIGH-25 / Phase 4 chart persistence:** Charts now persist a stable owner-scoped `/api/v1/charts/{owner}/{name}` path. Reading a historical chart re-signs the S3 object, so stored conversations no longer depend on an hour-old presigned URL. Cross-tenant chart access is rejected. S3 delivery was tested through storage seams, not a live browser/S3 stack.
- **MED-05 / CI:** Remaining unbounded runtime requirements are pinned to installed versions. CI now includes an explicit `alembic upgrade head` + `alembic check`, a gitleaks history scan, and `pip-audit` for Python dependencies. The workflow has not run on a hosted runner in this session; scanners were not executed locally.
- **OCR:** The backend image now installs Tesseract and English data. Image build and scanned-PDF OCR still cannot be verified while the Docker engine is unavailable.
- Tests added: chart generator/endpoint renewal and ownership, security headers, upload malware scanner protocol/fail-closed behavior, table/text deduplication. Final run **139 passed, 1 skipped**; Ruff and mypy passed; frontend lint/TypeScript/build passed and **6 Playwright contracts passed**.

### Infrastructure, integrations, and documentation follow-through

- Added owner-scoped retrieval API and MCP contract tests; removed caller-supplied identity and ignored message fields. Existing session scope is checked rather than silently overridden.
- Real Matplotlib chart rendering tested as PNG; storage-seam tests cover signing. Browser images load the chart URL unoptimized. Chart URL renewal was subsequently implemented (owner-scoped API path); live S3/browser delivery remains unverified.
- Evaluation now queues a Celery task with task-owned engine and Qdrant transport, no answer cache/history contamination, and failed status on judge/provider failure. Real worker execution still requires broker/Postgres acceptance.
- Added explicit owner-initiated `/documents/{id}/reindex`; it does not run automatically. Worker retries clean only that document's derived active-index vectors before replacement. No existing document was reindexed in this session.
- `/health` is liveness; `/ready` returns 503 for backing-service failure without raw exceptions; metrics require a separate bearer secret. Targeted readiness/security tests **5 passed** after fixing two test mocks of a read-only SQLAlchemy method.
- Compose now uses production application images, no source bind mounts, a migration prerequisite, loopback-bound stores, required runtime credentials, an external/optional Langfuse contract, and authenticated Prometheus scraping. Existing named volumes remain declared. Updated Compose syntax **passed** with process-local syntax-check placeholders only. Container execution/build/context inspection remains blocked by the Docker engine.
- Added frontend Dockerfile, same-origin API proxy, Markdown/GFM rendering without raw HTML, real auth UI, metadata/error rendering and browser contracts. Canonical README and `docs/OPERATIONS.md` now distinguish tested behavior, optional capabilities and outstanding acceptance; ignored learning/interview notes marked historical.
- Added CI dependency manifests, frontend lint/build/browser/audit/image checks and a safely gated PostgreSQL migration/drift test. CI workflow itself has not run on a clean hosted runner in this session.
- Modified files are recorded in each increment and visible in `git status`; some were already modified/untracked before this session. No commit/reset/cleanup has been performed.

### 0a — Build context and migration foundation

- Added root/backend Docker ignore rules excluding secrets. Actual image inspection remains blocked by Docker engine availability (CRIT-07 mitigated in configuration, not image-tested).
- Added explicit boto3, removed unused aioboto3, pinned most runtime dependencies to installed baseline versions, declared tabulate, populated test requirements (CRIT-06 dependency hygiene; HIGH-23 partial).
- Added additive migration `d91e3b7a6201` for message costs (CRIT-05 code fix; clean-Postgres application/drift still unverified).
- Files: `.dockerignore`, `backend/.dockerignore`, `backend/requirements{,-test}.txt`, `backend/app/db/migrations/versions/d91e3b7a6201_add_message_cost.py`, `backend/tests/unit/test_migration_contract.py`.
- Tests: full suite **50 passed**, Compose config **passed**, pip dependency-resolution dry run **passed** in existing venv. This is **not** a clean install test.

### 1a — Authentication and tenant boundaries

- Removed mock-token bypass and query-string credentials (CRIT-03). Enforced access/refresh types, active users, revocable DB sessions, refresh compare-and-swap rotation, logout revocation, HttpOnly cookies and trusted-origin CSRF protection (HIGH-22).
- Added exact-origin configuration, production secret validation, Railway async URL normalization (HIGH-17; HIGH-19 partial).
- Session/document ownership now checked before history, chat, streaming, or session creation. Invalid IDs return 422; unknown/foreign sessions 404; no silent substitution. History comes from durable DB, preserving timestamps (CRIT-04 route fixes; HIGH-18 backend).
- Qdrant requires tenant filter as well as document scope. Cache keys include tenant; empty-scope cache disabled. Corpus version/invalidation remain pending (HIGH-06 partial).
- Eval routes moved under `/api/v1/eval`, owner scoped, and migration adds/backfills owner from document without deleting runs. Background execution/provider correctness still pending (HIGH-15 partial).
- Removed agent-side history writes; route is sole turn persistence owner (HIGH-07). No artificial streaming sleeps; currently transparent completed-answer delivery until real provider streaming is implemented (HIGH-01 still open).
- Atomic Redis limiter fails closed; chat/stream share tier-based limits, auth uses per-IP buckets (HIGH-21 partial: upload coverage still pending).
- Tests: auth/security/settings/migration target **32 passed**. First full run **74 passed, 1 failed**: old mocked E2E queried an unprocessed document. Corrected test to explicitly simulate readiness; did not bypass readiness in application. Follow-up tenant/chat/cache/local-Qdrant/limiter target **16 passed**.
- Real local Qdrant engine (in-memory, not Docker) demonstrated isolation with deliberately shared document IDs across two tenants. API tests use real ASGI routing + SQLite, not live Postgres/Redis.
- Files: `backend/app/{config.py,main.py}`, `core/{security,dependencies,ownership,rate_limiter}.py`, `api/v1/{auth,chat,eval}.py`, `models/{auth_session,session,eval_result}.py`, migration env and revisions `e20a497f6312`, `f31b508a7423`, agent/retrieval/cache/eval tenant-context propagation, `backend/eval/run_eval.py`, MCP token default, test fixtures and auth/tenant/rate/settings/Qdrant regressions.

### 1b — Browser authentication

- Added login/register forms and dashboard auth gate; credentials stay in HttpOnly backend cookies, not localStorage or SSE URLs. Serialized refresh flow and logout added. Removed fabricated home/new-session fallback IDs (CRIT-08 implementation; browser E2E not yet run).
- Files: `frontend/src/lib/api.ts`, `components/{AuthForm,ChatWindow}.tsx`, `app/{page,login/page,register/page}.tsx`, `app/(dashboard)/layout.tsx`, `app/(dashboard)/chat/[sessionId]/page.tsx`.
- Tests: frontend TypeScript **passed**; production `npm run build` **passed** including login/register/chat routes. Browser interactions and remaining pre-existing lint failures not yet validated/fixed.

### 2a — Real providers and ingestion correctness

- Configurable `gemini-3.5-flash-lite` and `gemini-embedding-001` selected from Google's current stable-model documentation. Shared SDK adapter uses asynchronous calls for agents, current Langfuse observation API, actual usage metadata (including thinking), and typed failures. All random/mock generation, embedding, judge, dataset, and confidence fallbacks removed (CRIT-01/02, HIGH-02/11 code fixes).
- **Live provider smoke passed** using existing configured credentials: one synthetic generation (**13 total tokens**) and one embedding (**768 dimensions**). No user documents sent. This validates provider access, not full deployment.
- Cost is `null` unless operator provides model prices; never guess input/output proportions or silently report unknown as zero.
- Vector collections are namespaced by model/dimension/version; dimensions/counts/finite vectors validated, deterministic point IDs used. Existing collection/data were not deleted or reindexed.
- Full chunk text stored in additive revision `a42c619b8534`; legacy previews explicitly require reindexing. Worker owns fresh NullPool engine + Qdrant transport per task, moves blocking work off-loop, rolls back and records failure, skips completed retries. Tested two separate Celery-style event loops with real PDF parsing, SQL and local Qdrant.
- Upload limits, signatures/Office structure/UTF-8 validation, visible queue failure, and retryable deletion across vectors/cache/S3 implemented. **No antivirus scanning claim.**
- Cache now scoped by tenant, corpus chunk identity, model, dimensions, and conversation history, with document invalidation indexes. DB is authoritative for prompt history and turn persistence.
- Tests: provider/agent contracts **22 passed**; ingestion/deletion/tenant/cache regression target **20 passed**. External S3, Redis, Celery broker, and PostgreSQL still not tested live.
- Files: shared LLM/embedding clients, `llm/errors.py`, agent classes, eval clients/runner, vector store/reranker, full-text model/migration, worker, upload validation/document routes, cache/history, new provider/agent/worker/document/cache regression tests; `backend/scripts/check_providers.py`.

### 2b — Streaming and guardrails

- Actual SDK async streaming wired through agent to SSE with a bounded queue, cancellation handling, structured errors, and persistence before final success. JSON framing is not shown as answer text. Sentence/newline buffering redacts identifiers split across provider chunks before delivery.
- Final event carries authoritative answer, citations, grounding heuristic, chart, cache status, actual usage and nullable cost. Added metadata persistence revision `b53d720c9645` and history fields.
- Deterministic PII pattern mode protects email/phone/SSN/card-like identifiers. Explicit Presidio mode fails closed if its provisioned spaCy model is absent; **person-name detection in pattern mode is not claimed**. Dependencies installed successfully; no automatic model download at import.
- Removed broad keyword blocks; no-source grounding now zero/flagged. Groundedness is a heuristic, not calibrated confidence.
- Replaced thread sandbox with killable child process, restricted builtins, bounded output, clean environment, concurrency limit, and Unix resource limits. RestrictedPython is **not** a complete OS sandbox.
- Targeted run: **21 passed, 1 failed** exposed same-timestamp ordering of user/assistant rows. Corrected persistence to assign explicit ordered timestamps; full regression rerun pending below.
- Frontend `npm run lint` now **passes with no errors/warnings**; TypeScript and updated production build **passed**. Browser E2E still pending.
- Files: base/reasoning/orchestrator streaming paths, output parser, SSE schema/routes, message model/migration; PII/content/grounding services; Python executor/child entry point; frontend API/store/chat/message/history handling; streaming/guardrail/tool tests.

## Safety and baseline

- Read the entire 2,752-line technical audit, including all phases and acceptance criteria.
- Existing working tree has 29 modified tracked files and additional untracked project files. Preserve that work; no reset, cleanup, secret replacement, or database deletion is authorized.
- No existing database has been migrated or modified by this repair session.
- Docker CLI exists, but the Docker Desktop Linux engine is unavailable. Live container and clean PostgreSQL acceptance checks are blocked until it is running.
- Baseline backend: `HF_HUB_OFFLINE=1 python -m pytest tests/ -q --tb=short` — **48 passed, 6 warnings** (55.23s; root-directory invocation also passed).
- Baseline frontend: `npm run lint` — **6 errors, 8 warnings**.

## Independent Critical/High verification

All entries below were checked against source before repairs. This is the **historical initial verification table**; the latest verification snapshot and increments above record the current repair state. “Confirmed” means code/config evidence, not a live exploit or successful external-service test.

| Issue | Source evidence | Initial verification status (before repairs) |
|---|---|---|
| CRIT-01 | `services/agents/base_agent.py::_think`, `services/llm/gemini_client.py`, eval clients hardcode Gemini 1.5 and fabricate success | Confirmed code; replacement pending |
| CRIT-02 | `services/retrieval/embedding_service.py` catches failures and generates random vectors; vector store fixed at 768 | Confirmed; pending |
| CRIT-03 | `core/dependencies.py::get_current_user` accepts literal mock token and provisions a user | Confirmed; pending |
| CRIT-04 | `api/v1/chat.py` reads Redis before ownership; session document IDs unchecked; vector filter lacks tenant; eval unscoped | Confirmed; pending |
| CRIT-05 | `models/message.py` declares cost; all five existing migrations omit it | Confirmed; pending additive revision |
| CRIT-06 | `core/storage.py` directly imports boto3, requirements only declare aioboto3 | **Audit correction:** installed aioboto3 15.5.0 requires `aiobotocore[boto3]`; dependency metadata does install boto3. Direct declaration still needed; alleged missing transitive dependency not confirmed |
| CRIT-07 | Backend Dockerfile uses `COPY . .`; no Docker ignore file; local env file exists (values not displayed) | Confirmed build-context exposure; image extraction not tested |
| CRIT-08 | `frontend/src/lib/api.ts` supplies mock tokens in three places; home fabricates session; no auth pages | Confirmed; pending |
| HIGH-01 | `api/v1/chat.py::event_generator` splits completed answer with sleeps; stream schema incomplete | Confirmed; pending |
| HIGH-02 | Agent uses independent SDK client; `gemini_client.py` invokes obsolete `.trace()` API | Confirmed; pending |
| HIGH-03 | Compose Langfuse latest has only Postgres, lacks modern backing services | Confirmed config; startup not tested |
| HIGH-04 | Worker calls `asyncio.run` per task with module-global pooled async engine | Confirmed unsafe lifecycle; runtime reproduction pending |
| HIGH-05 | `documents.py::delete_document` never calls existing vector deletion method | Confirmed; pending |
| HIGH-06 | Semantic cache keys omit user; empty scope global; invalidate is `pass` | Confirmed; pending |
| HIGH-07 | Reasoning agent and stream route both call `save_turn` | Confirmed; pending |
| HIGH-08 | Tools/PII import undeclared dependencies and silently disable functionality | Confirmed; pending |
| HIGH-09 | Python tool timeout exits a ThreadPoolExecutor context that waits; replaces global stdout | Confirmed; do not execute infinite loop until fixed |
| HIGH-10 | PII silently no-ops; substring filtering; no-source grounding is 1.0; uncaught SSE errors | Confirmed; pending |
| HIGH-11 | Reasoning parse exception generates random confidence and 120 tokens | Confirmed; pending |
| HIGH-12 | Worker stores 200 chars; Chunk.text returns preview; BM25/eval consume previews | Confirmed; pending |
| HIGH-13 | Sync S3/SDK/reranker/judge in async paths; CrossEncoder instantiated at import | Confirmed; pending |
| HIGH-14 | Upload unbounded read and client MIME trust without content validation | Confirmed; pending |
| HIGH-15 | Eval judge random fallback; dataset mock fallback; unscoped routes; API-process execution | Confirmed; pending |
| HIGH-16 | MCP defaults mock token; `/retrieval/search` missing; ignored doc_ids/user_id | Confirmed; pending |
| HIGH-17 | Main CORS literal `https://*.railway.app` cannot match deployed origin | Confirmed; pending |
| HIGH-18 | Home fabricates IDs; chat silently creates/replaces sessions; history substitutes latest | Confirmed; pending |
| HIGH-19 | Compose frontend runs npm install/dev; no frontend Dockerfile; Railway backend only | Confirmed config; deployment not tested |
| HIGH-20 | Health exposes exception details and leaks Qdrant client; metrics public | Confirmed; pending |
| HIGH-21 | Only nonstream chat limited; email-based tier; Redis failures allow requests | Confirmed; pending |
| HIGH-22 | Access/refresh identical type; no rotation/logout; no active-user/default-secret enforcement | Confirmed; pending |
| HIGH-23 | Empty test requirements/document tests; mocked E2E; CI lacks migration/frontend checks | Confirmed; pending incremental regression coverage |
| HIGH-24 | README claims production-grade, retired models, complete tracing/deployment and invented performance | Confirmed; documentation will track tested implementation |
| HIGH-25 | Chart presign uses internal S3 endpoint; Next Image permits only localhost:8000 | Confirmed config; browser test pending |
| HIGH-26 | Raw-text message rendering, dead document button/cache badge; lint failures reproduced | Confirmed; pending after Phases 0–2 |

## Execution plan

1. Phase 0a: safe Docker contexts, explicit/pinned dependencies, additive cost migration, migration regression checks.
2. Phase 1: remove bypass, enforce token types/active users and tenant ownership, secure frontend authentication and SSE, rate limits/CORS.
3. Phase 2: supported configurable models, typed provider errors, ingestion/full-text retrieval, scoped cache, real streaming and persistence, deterministic guardrails.
4. Infrastructure and polish follow in tested increments. Do not rebuild/delete existing vector collections; use a new versioned collection and opt-in reindexing.

## Changes, tests, and remaining work

- Critical/High source verification is complete, with the CRIT-06 correction above. The table records the initial verification; completed increments above supersede its original pending labels.
- CRIT-01/02/03/04 and HIGH-01/06/07/09/10/11/16/17/18/20/22 have implemented fixes with targeted regression coverage; infrastructure-dependent acceptance is still distinct from code-level tests.
- CRIT-08 authentication/browser contract including logout is now tested. CRIT-05/07, HIGH-03/04/05/08/12/13/14/15/19/21/23/24 have implemented or partial repairs whose full acceptance still requires real services, remaining UX work, or deployment validation. HIGH-25 chart renewal/ownership and HIGH-26 Markdown, citation, cache and lint work are implemented with seam/browser-contract coverage; live deployment acceptance is still pending.
- Owner-scoped source download/open action added and tested at the API boundary; BM25 regression now explicitly matches a keyword after character 200. Local Qdrant deletion test proves removal of one tenant's vectors does not delete the other tenant's points. Targeted run: **11 passed**.
- Remaining priorities: clean PostgreSQL upgrade/drift test and container builds; real upload→worker→retrieval→stream→history→delete workflow; live Redis limit/cache tests; provision/test Presidio model, Tesseract OCR and a live ClamAV service; external Langfuse delivery; resumed processing-status observation after browser reload; run CI secret/dependency scans on a hosted runner; deployment/backup/sandbox hardening; and live acceptance of the repaired Medium/Low findings.
- Release/review notes: no commit was made. The pre-existing untracked `c58ae3901b52` migration, DOCX parser and session compatibility module were preserved and must ship with the reviewed project changes; the new migration chain depends on that existing revision. Git ignore exceptions now expose the previously hidden frontend API/utilities source. Do not omit these files from a future commit.
- Ruff's configured gate checks F/E9 correctness; repository-wide style/import normalization is not claimed. CI and Docker image builds are configured but have not been executed on a clean hosted runner. The overall audit acceptance checklist is **not complete**, and the application is **not certified production-ready** by this session.
- Live Google and synthetic local RAG have passed. Full live PostgreSQL/Redis/Qdrant-server/S3/Celery/browser deployment acceptance has **not** been established.
- Existing keys may need operator-managed rotation if images were previously shared; no keys were changed.

## Estimated API usage

- Intentional Google usage: one provider smoke run plus three synthetic RAG runs (one initial failure, one non-streaming pass, one streaming pass). Estimated **7 generation requests and 13 embedding RPCs**, roughly **9,000 generation tokens** including the unreported initial-run usage. Measured successful responses: **13 + 2,964 + 2,932 = 5,909 generation tokens**. Estimated spend **under $0.10**, not invoice-verified; no current per-model prices were fabricated in application accounting.
- Baseline/unit/browser contract tests mock provider boundaries and are not intended to consume Google quota. Additional live checks must be appended here.
- Coding-assistant token/cost billing is not exposed by this harness; cannot report a measured dollar figure. Track intentional provider calls here as implementation proceeds.
