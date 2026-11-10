+-----------------------------------------------------------------------+
| **Quantiva --- Read-Only Technical Audit**                            |
|                                                                       |
| Static Analysis, Local Verification, Security Review & Prioritized    |
| Repair Roadmap                                                        |
+=======================================================================+
+-----------------------------------------------------------------------+

  -----------------------------------------------------------------------
  **Audit Parameter**                 **Details**
  ----------------------------------- -----------------------------------
  **Repository**                      I:\\AI Engineer Projects\\Quantiva

  **Audit Date**                      2026-11-08

  **Mode**                            Read-only static analysis + safe
                                      local verification. No project
                                      files were modified, deleted,
                                      renamed, or created.

  **Docs Audited**                    quantiva_learn/1-INDEX.md,
                                      2-PROJECT-OVERVIEW.md,
                                      3-TECH-STACK.md,
                                      4-SYSTEM-ARCHITECTURE.md,
                                      5-PROJECT-STRUCTURE.md, README.md,
                                      VIBE_CODING_WALKTHROUGH.md.
  -----------------------------------------------------------------------

**0. Audit Snapshot**

**Commands run and results**

  -----------------------------------------------------------------------
  **Check**                           **Result**
  ----------------------------------- -----------------------------------
  **Python AST syntax check over      93 files checked, no syntax errors
  backend/app**                       

  **pytest tests/ -x -q \--tb=short   48 passed, 6 warnings, \~55 s
  with HF_HUB_OFFLINE=1**             

  **npx tsc \--noEmit \--incremental  Passed
  false**                             

  **npm run lint**                    Failed: 6 errors, 8 warnings

  **docker compose config \--quiet**  Passed (syntax only)

  **python -c \"from app.main import  Loaded, but triggered Hugging Face
  app\"**                             CrossEncoder model download at
                                      import

  **python -c \"from langfuse import  False with installed langfuse 5.1.0
  Langfuse; hasattr(Langfuse,         
  \'trace\')\"**                      

  **Dependency probe**                boto3 installed locally;
                                      RestrictedPython, presidio\_\*,
                                      matplotlib absent
  -----------------------------------------------------------------------

**Evidence status legend**

-   **Verified ---** directly observed in code, configuration, or
    > command output.

-   **Verified external ---** vendor deprecation confirmed outside the
    > repo.

-   **Suspected ---** strong inference from code/config, but not
    > exercised against live infrastructure.

**Important audit limitation**

No live Postgres, Redis, Qdrant, S3, Celery broker, Gemini API, Docker
build, or browser E2E run was performed. Findings marked Suspected
require the reproduction steps in §12. Secret values in backend/.env
were identified by key name only and are redacted in this report.

**1. Executive Summary**

**Verdict**

Quantiva is not runnable end-to-end from a fresh clone and is not
production-ready. The codebase contains a coherent architecture and a
polished UI, but the critical path is broken in four independent ways:

**1. The LLM and embedding models are retired.** gemini-1.5-flash was
shut down on 2025-09-29 and text-embedding-004 on 2026-01-14. The app
hardcodes both.

**2. Failures are silently converted into fake success.** LLM errors
produce synthesized document excerpts; embedding errors produce random
vectors; parse errors produce random confidence scores.

**3. The database schema does not match the ORM.** messages.cost_usd
exists in the model but in no migration, so every chat message insert
fails on a fresh database.

**4. Authentication and tenant isolation are broken.** The literal token
mock-token authenticates as the first user, and several endpoints do not
verify ownership.

**Issue counts**

  -----------------------------------------------------------------------
  **Severity**                        **Count**
  ----------------------------------- -----------------------------------
  **Critical**                        8

  **High**                            26

  **Medium**                          12

  **Low**                             6

  **Total**                           52
  -----------------------------------------------------------------------

**Readiness scorecard**

  -------------------------------------------------------------------------------
  **Subsystem**           **Classification**      **Basis**
  ----------------------- ----------------------- -------------------------------
  **Backend**             Broken                  Retired models, missing runtime
                                                  deps, schema drift, auth bypass

  **Frontend**            Partially implemented   Typecheck passes; lint fails;
                                                  no auth UI; fake stream
                                                  consumption

  **Database**            Broken                  messages.cost_usd missing;
                                                  eval_runs.user_id missing;
                                                  migration/model drift

  **Document ingestion**  Broken                  Random embeddings; Celery
                                                  event-loop risk; missing deps

  **Retrieval / RAG**     Broken                  Retired embedding model; random
                                                  fallback; BM25 sees 200-char
                                                  previews only

  **AI agent**            Broken                  Retired model; silent non-LLM
                                                  fallback; fake
                                                  confidence/tokens

  **LLM integration**     Broken                  Retired model; main path
                                                  bypasses gemini_client; no real
                                                  streaming

  **Streaming**           Broken                  Full answer computed first,
                                                  then split into words

  **Authentication**      Broken                  mock-token bypass; no frontend
                                                  auth; no refresh flow

  **Security**            Broken / Critical       Auth bypass, cross-tenant IDOR,
                                                  secrets in Docker build
                                                  context, token in URL

  **Docker                Broken                  Missing boto3; invalid Langfuse
  infrastructure**                                stack; frontend runs dev server

  **Testing**             Inadequate              48 mocked tests pass; no real
                                                  ingestion/retrieval/agent/SSE
                                                  tests

  **UX**                  Partially implemented   Strong visual design; raw
                                                  Markdown; chart rendering
                                                  broken; fallback sessions

  **Observability**       Broken / Partially      HTTP metrics work;
                          implemented             LLM/cache/stream metrics
                                                  unused; Langfuse path dead
  -------------------------------------------------------------------------------

**Top 10 blockers in impact order**

**1.** messages.cost_usd missing migration → chat endpoint 500s.

**2.** Missing boto3 in requirements.txt → Docker/startup import
failure.

**3.** Retired gemini-1.5-flash → all real generation fails.

**4.** Retired text-embedding-004 → all real embeddings fail; random
vectors used.

**5.** mock-token auth bypass → any caller becomes the first user.

**6.** Cross-tenant IDOR → sessions, history, documents, retrieval, and
eval are not fully scoped.

**7.** Missing RestrictedPython, Presidio, and Matplotlib → tools and
guardrails silently no-op.

**8.** Fake SSE streaming → no token-by-token LLM output; missing
confidence/chart fields.

**9.** Docker COPY . . with no .dockerignore → local secrets baked into
images.

**10.** Langfuse compose stack invalid and Langfuse SDK API mismatch →
observability dead.

**2. Actual Architecture (As Implemented)**

**2.1 Runtime topology**

> Browser (Next.js 16.3.8 / React 19)\
> │ Axios REST + EventSource SSE\
> │ localStorage JWT or literal \"mock-token\"\
> ▼\
> FastAPI /api/v1\
> ├── /auth register, login, me\
> ├── /documents upload, list, get, delete, status SSE, chunks\
> ├── /chat sessions, message, message/stream SSE, history\
> ├── /run eval trigger (no /eval prefix)\
> ├── /{run_id} eval result\
> └── /history/list eval list\
> │\
> ├── Postgres (users, documents, chunks, sessions, messages,
> eval_runs)\
> ├── Redis (session memory, semantic cache, rate limiter)\
> ├── Qdrant (768-d vectors, payload: doc_id, user_id, text, page)\
> └── S3 / s3mock (raw files, generated charts)\
> │\
> └── Celery worker (process_document)

**2.2 Ingestion flow**

> POST /documents/upload\
> → read entire file into memory\
> → boto3 upload to S3 (synchronous, in request path)\
> → insert Document(status=\"processing\")\
> → Celery process_document.delay(doc_id)\
> → asyncio.run(\_process_document_async(doc_id))\
> → boto3 get_file (sync)\
> → parse PDF/XLSX/CSV/DOCX\
> → chunk_text_blocks(chunk_size=512 chars, overlap=64)\
> → embedding_service.embed_batch(\...) (sync; retired model)\
> → vector_store.upsert_chunks(\...)\
> → insert Chunk(text_preview=text\[:200\], qdrant_id=\...)\
> → Document.status=\"ready\"

**2.3 Query flow**

> GET /chat/message/stream?session_id=\...&query=\...&token=\...\
> → get_current_user (accepts mock-token)\
> → load/create Session\
> → load all Chunk rows for session.document_ids (preview only)\
> → event_generator:\
> → orchestrator.run(\...)\
> → content_filter.check_input (naive substring filter)\
> → semantic_cache.get(query, doc_ids) (sync embedding; no user scope)\
> → route query to ReasoningAgent or DataAnalystAgent\
> → ReasoningAgent.run:\
> → session_memory.load_history\
> → BaseAgent ReAct loop:\
> → genai.Client.models.generate_content(model=\"gemini-1.5-flash\")\
> → retrieve_chunks tool:\
> → hybrid_retriever.search:\
> → embedding_service.embed_text (sync; retired)\
> → Qdrant dense search\
> → BM25 over text_preview only\
> → CrossEncoder rerank (sync)\
> → on any API error: synthesize answer from chunks\
> → parse JSON output; on failure: random confidence + tokens=120\
> → guardrails_service.check_output\
> → semantic_cache.set(query, doc_ids, answer)\
> → split answer into words, sleep 15 ms per word\
> → emit final SSE event\
> → save duplicate Redis turn\
> → insert assistant Message(cost_usd=\...) ← fails on fresh DB

**2.4 Actual vs documented architecture**

  -----------------------------------------------------------------------
  **Documented**                      **Actual**
  ----------------------------------- -----------------------------------
  **Next.js 14**                      Next.js 16.3.8, React 19.2.8

  **Gemini 1.5 Flash**                Retired; no supported model
                                      configured

  **text-embedding-004 768-d**        Retired; random fallback

  **Token-by-token streaming**        Full answer first, then
                                      word-splitting

  **Langfuse tracing on every call**  gemini_client bypassed; Langfuse
                                      SDK v5 has no .trace()

  **RestrictedPython / Presidio /     Packages not in requirements.txt;
  Matplotlib**                        tools silently disabled

  **POST /api/v1/eval/run**           Actual POST /api/v1/run; no /eval
                                      prefix

  **Railway full-stack**              railway.json builds backend only
  -----------------------------------------------------------------------

**3. Repository Structure**

> Quantiva/\
> ├── backend/\
> │ ├── app/\
> │ │ ├── main.py\
> │ │ ├── config.py\
> │ │ ├── api/v1/\
> │ │ │ ├── auth.py\
> │ │ │ ├── chat.py\
> │ │ │ ├── documents.py\
> │ │ │ ├── eval.py\
> │ │ │ └── metrics_endpoint.py\
> │ │ ├── core/\
> │ │ │ ├── dependencies.py\
> │ │ │ ├── security.py\
> │ │ │ ├── storage.py\
> │ │ │ ├── rate_limiter.py\
> │ │ │ ├── metrics.py\
> │ │ │ └── middleware.py\
> │ │ ├── db/\
> │ │ │ ├── database.py\
> │ │ │ ├── session.py\
> │ │ │ └── migrations/versions/\
> │ │ │ ├── 74649b69a3ad_create_users_table.py\
> │ │ │ ├── 6588614a2571_create_documents_table.py\
> │ │ │ ├── 4c68095d9105_create_chunks_table.py\
> │ │ │ ├── b47dc6e638e8_create_chat_tables.py\
> │ │ │ └── c58ae3901b52_create_eval_runs_and_cost.py\
> │ │ ├── models/\
> │ │ │ ├── user.py\
> │ │ │ ├── document.py\
> │ │ │ ├── chunk.py\
> │ │ │ ├── session.py\
> │ │ │ ├── message.py\
> │ │ │ └── eval_result.py\
> │ │ ├── schemas/\
> │ │ │ ├── auth.py\
> │ │ │ ├── chat.py\
> │ │ │ ├── document.py\
> │ │ │ └── agent.py\
> │ │ ├── services/\
> │ │ │ ├── agents/{base_agent,reasoning_agent,orchestrator}.py\
> │ │ │ ├──
> guardrails/{content_filter,pii_detector,hallucination_checker,guardrails_service}.py\
> │ │ │ ├── ingestion/{ingestion_service,chunker}.py + parsers/\
> │ │ │ ├──
> llm/{gemini_client,prompt_builder,output_parser,context_manager}.py\
> │ │ │ ├── memory/{session_memory,semantic_cache,memory_pager}.py\
> │ │ │ ├──
> retrieval/{embedding_service,vector_store,bm25_retriever,hybrid_retriever,reranker}.py\
> │ │ │ ├── tools/{python_executor,chart_generator,registry}.py\
> │ │ │ └── eval/{eval_dataset,eval_runner,llm_judge,metrics}.py\
> │ │ └── workers/{celery_app,ingestion_worker}.py\
> │ ├── mcp_server/server.py\
> │ ├── eval/run_eval.py\
> │ ├── tests/{unit,integration,e2e}/\
> │ ├── Dockerfile\
> │ ├── requirements.txt\
> │ ├── requirements-test.txt\
> │ ├── alembic.ini\
> │ └── pytest.ini\
> ├── frontend/\
> │ ├── src/app/\
> │ │ ├── page.tsx\
> │ │ ├── layout.tsx\
> │ │ └── (dashboard)/chat/\[sessionId\]/page.tsx\
> │ ├── src/components/{ChatWindow,FileUploader,MessageBubble}.tsx\
> │ ├── src/lib/{api,utils}.ts\
> │ ├── src/store/{useChatStore,useDocumentStore}.ts\
> │ ├── next.config.ts\
> │ └── package.json\
> ├── monitoring/\
> │ ├── prometheus.yml\
> │ └── grafana/{dashboards,provisioning}/\
> ├── infra/ \# empty\
> ├── docker-compose.yml\
> ├── railway.json\
> ├── README.md\
> ├── .github/workflows/ci.yml\
> ├── quantiva_learn/ \# docs, git-ignored\
> └── VIBE_CODING_WALKTHROUGH.md

Repo state: git status shows extensive uncommitted modifications and an
untracked migration c58ae3901b52_create_eval_runs_and_cost.py.
quantiva_learn/ and VIBE_CODING_WALKTHROUGH.md are untracked/ignored.

**4. Feature Status Matrix**

  -------------------------------------------------------------------------------------
  **Feature**        **Documentation      **Actual                 **Classification**
                     claim**              implementation**         
  ------------------ -------------------- ------------------------ --------------------
  **User             JWT auth             Implemented; no frontend Implemented but
  register/login**                        UI; tests pass           unverified

  **JWT refresh**    Refresh token issued No refresh endpoint, no  Missing
                                          token type, no           
                                          revocation               

  **Document         PDF/XLSX/CSV/DOCX    Implemented; sync S3; no Partially
  upload**                                size/magic-byte limits   implemented

  **PDF parsing**    Implemented          PyMuPDF text + table     Working and verified
                                          extraction               

  **Excel/CSV        Implemented          Pandas/openpyxl          Implemented but
  parsing**                                                        unverified

  **DOCX parsing**   Docs say planned     Parser exists; untracked Implemented but
                                                                   unverified

  **OCR**            Docs say planned     get_textpage_ocr         Partially
                                          attempted; Tesseract not implemented / Broken
                                          provisioned              

  **Chunking**       512-token chunks     512 characters; tables   Partially
                                          split                    implemented

  **Embeddings**     text-embedding-004   Retired; random fallback Broken

  **Qdrant           Implemented          Upsert exists; no        Implemented but
  indexing**                              cleanup; no live test    unverified

  **BM25**           Hybrid keyword       Indexes text_preview     Partially
                     search               only                     implemented

  **Hybrid           Dense + BM25 +       Dense + preview BM25 +   Broken
  retrieval**        rerank               sync rerank              

  **Cross-encoder    Implemented          Loaded at import;        Implemented but
  rerank**                                downloads model at       unverified
                                          startup                  

  **ReAct agent**    Gemini 1.5 Flash     Retired model; silent    Broken
                                          fallback                 

  **Tool calling**   Python/chart tools   Missing dependencies     Broken

  **Chart            Matplotlib → S3      Matplotlib missing;      Broken
  generation**                            presigned URL internal   
                                          host                     

  **Guardrails**     Presidio +           Presidio no-op; naive    Broken
                     hallucination check  filter; hallucination    
                                          1.0 with no sources      

  **Semantic cache** Cosine similarity    Global/global-per-doc;   Partially
                                          no user scope; no        implemented
                                          invalidation             

  **Session memory** Redis sliding window Works; duplicate writes  Implemented but
                                                                   unverified

  **Document status  Streaming status     Works; polls every 1 s   Working and verified
  SSE**                                                            

  **Chat SSE**       Token-by-token       Fake word-splitting      Broken

  **Evaluation**     LLM judge + dataset  Random fallbacks; no     Broken
                                          ownership                

  **MCP server**     3 tools              Nonexistent endpoint;    Broken
                                          ignored payload fields   

  **Prometheus       HTTP + LLM + cache + HTTP only; others unused Partially
  metrics**          streams                                       implemented

  **Langfuse         Every LLM call       Dead path; SDK mismatch; Broken
  tracing**                               compose invalid          

  **Frontend upload  Drag-and-drop +      Works; typecheck passes  Working and verified
  UI**               progress                                      

  **Frontend chat    Streaming +          Raw Markdown; missing    Partially
  UI**               citations +          stream fields            implemented
                     confidence                                    

  **Frontend auth    Login/register       Missing                  Missing
  UI**                                                             

  **Docker Compose** Full stack           Backend missing boto3;   Broken
                                          Langfuse invalid;        
                                          frontend dev server      

  **Railway          Full-stack           Backend only; no         Partially
  deployment**                            Qdrant/S3/frontend       implemented

  **CI/CD**          Lint, typecheck,     Backend only; mypy       Partially
                     test, Docker         non-blocking; no         implemented
                                          frontend                 
  -------------------------------------------------------------------------------------

**5. Complete Bug Register**

**5.1 Critical Findings**

**CRIT-01 --- Retired Gemini model with silent non-LLM fallback**

-   **Severity:** Critical

-   **Status:** Verified (code) + Verified external (model retirement)

-   **Location:** backend/app/services/agents/base_agent.py:19,
    > :141-146; backend/app/services/llm/gemini_client.py:68;
    > backend/app/services/eval/eval_dataset.py:22;
    > backend/app/services/eval/llm_judge.py:18;
    > backend/app/schemas/agent.py:27

-   **Component:** BaseAgent.\_think, GeminiClient, eval judge/dataset

-   **Evidence:** self.model_name = \"gemini-1.5-flash\" is hardcoded in
    > four places. BaseAgent.\_think wraps generate_content in except
    > Exception and returns a MockResponse built from retrieved chunks.
    > Google shut down gemini-1.5-flash on 2025-09-29.

-   **Impact:** On every real chat request, the LLM call fails and the
    > user receives a synthesized excerpt masquerading as an AI answer.
    > No error is surfaced. Eval and judge paths also fail.

-   **Expected behavior:** Use a currently supported model, configured
    > via environment variable, and fail loudly with a typed error when
    > the LLM is unavailable.

-   **Recommended fix:** Introduce GEMINI_CHAT_MODEL (e.g., a current
    > stable Gemini Flash model), remove the mock fallback from
    > production, add a startup readiness check, and update
    > eval/judge/schema defaults.

-   **Affected:** Chat, agent, eval, judge, cost/token accounting, docs.

-   **Verified/Suspected:** Verified code fact; retirement verified
    > externally.

**CRIT-02 --- Retired embedding model with random-vector fallback**

-   **Severity:** Critical

-   **Status:** Verified (code) + Verified external (model retirement)

-   **Location:**
    > backend/app/services/retrieval/embedding_service.py:11, :26-32,
    > :49-52; backend/app/services/retrieval/vector_store.py:11

-   **Component:** EmbeddingService.embed_text, embed_batch, VectorStore

-   **Evidence:** self.model_name = \"text-embedding-004\"; on any
    > exception, random.seed(seed) and random.uniform(-1, 1) vectors are
    > returned. Qdrant collection is hardcoded to 768 dimensions. Google
    > shut down text-embedding-004 on 2026-01-14.

-   **Impact:** Ingestion marks documents ready with meaningless random
    > vectors. Retrieval returns arbitrary chunks. No error is visible
    > to the user.

-   **Expected behavior:** Use a supported embedding model, record
    > embedding failures as document failures, and require reindexing
    > when dimensions change.

-   **Recommended fix:** Replace with a current embedding model (e.g.,
    > gemini-embedding-001 or text-embedding-005), make model/dimension
    > configurable, remove random fallback, add a reindex migration, and
    > validate vector dimension on upsert.

-   **Affected:** Ingestion, Qdrant, retrieval, semantic cache,
    > hallucination checker, eval.

-   **Verified/Suspected:** Verified code fact; retirement verified
    > externally.

**CRIT-03 --- mock-token authentication bypass**

-   **Severity:** Critical

-   **Status:** Verified

-   **Location:** backend/app/core/dependencies.py:28, :37-52;
    > backend/mcp_server/server.py:55

-   **Component:** get_current_user, MCP client

-   **Evidence:** The MCP server defaults QUANTIVA_TOKEN to
    > \"mock-token\".

> token = credentials.credentials if credentials else
> request.query_params.get(\"token\")\
> if token == \"mock-token\":\
> result = await db.execute(select(User).limit(1))\
> user = result.scalar_one_or_none()\
> if user:\
> return user\
> demo_user = User(email=\"demo@quantiva.ai\", \..., tier=\"pro\")

-   **Impact:** Anyone who sends Authorization: Bearer mock-token
    > becomes the first user in the database, or auto-provisions a pro
    > demo user. This bypasses all authentication and tenant isolation.

-   **Expected behavior:** Reject mock-token in all environments; allow
    > bypass only behind an explicit test flag that is impossible in
    > production.

-   **Recommended fix:** Delete the bypass; add a startup assertion that
    > APP_ENV=production rejects the default secret and mock token; add
    > an integration test proving mock-token returns 401.

-   **Affected:** Every authenticated endpoint, MCP, frontend.

-   **Verified/Suspected:** Verified.

**CRIT-04 --- Broken tenant isolation and cross-tenant IDOR**

-   **Severity:** Critical

-   **Status:** Verified (code); exploit is straightforward but
    > Suspected end-to-end without live data

-   **Location:** backend/app/api/v1/chat.py:288-290 (history reads
    > Redis by raw session_id before any ownership check);
    > backend/app/api/v1/chat.py:38 (create_session accepts arbitrary
    > document UUIDs); backend/app/api/v1/chat.py:304-308 (DB history
    > query filters only by Message.session_id);
    > backend/app/services/retrieval/vector_store.py:77-91 (Qdrant
    > filter uses doc_id only, not user_id);
    > backend/app/api/v1/eval.py:47-67, :69-102 (eval trigger/read/list
    > have no ownership checks; EvalRun has no user_id).

-   **Component:** get_history, create_session, vector_store.search,
    > eval routes

-   **Evidence:** get_history calls
    > session_memory.load_history(str(session_id), \...) and returns it
    > if non-empty. No Session.user_id check occurs on that path.
    > create_session does doc_uuids = \[uuid.UUID(d) for d in
    > req.document_ids\] with no Document.user_id query. Qdrant payload
    > contains user_id, but search() never filters on it. EvalRun model
    > and migration have no user_id.

-   **Impact:** An authenticated attacker who knows or guesses a session
    > UUID or document UUID can read another tenant's chat history,
    > query another tenant's documents, and read eval results. A user
    > can also attach another user's documents to their own session.

-   **Expected behavior:** Every session, message, document, chunk, and
    > eval query must be scoped by current_user.id; cross-tenant access
    > must return 404/403.

-   **Recommended fix:** Add ownership joins to all
    > history/message/session/eval queries; validate document IDs
    > against Document.user_id on session creation; add user_id to
    > Qdrant filter and EvalRun; add cross-tenant integration tests.

-   **Affected:** Chat, documents, retrieval, eval, MCP.

-   **Verified/Suspected:** Verified code paths; live exploit suspected.

**CRIT-05 --- messages.cost_usd exists in ORM but not in any migration**

-   **Severity:** Critical

-   **Status:** Verified

-   **Location:** backend/app/models/message.py:22;
    > backend/app/db/migrations/versions/b47dc6e638e8_create_chat_tables.py:33-44;
    > backend/app/api/v1/chat.py:143, :261

-   **Component:** Message model, chat insert paths

-   **Evidence:** Model declares cost_usd: Mapped\[float\] =
    > mapped_column(Float, \...). The messages table migration creates
    > id, session_id, role, content, citations, tokens, latency_ms,
    > created_at --- no cost_usd. The newer migration c58ae3901b52 only
    > creates eval_runs. Chat inserts explicitly pass cost_usd=cost.

-   **Impact:** On a fresh alembic upgrade head database, POST
    > /chat/message and the SSE save path fail with column \"cost_usd\"
    > of relation \"messages\" does not exist. Chat is unusable.

-   **Expected behavior:** ORM and migrations must be in sync; alembic
    > upgrade head must produce a schema matching Base.metadata.

-   **Recommended fix:** Add an Alembic revision adding
    > messages.cost_usd (nullable or server default 0.0), commit it, and
    > add a migration-drift test (alembic check).

-   **Affected:** Chat, history, cost tracking.

-   **Verified/Suspected:** Verified by comparing model and migration.

**CRIT-06 --- Missing boto3 dependency breaks Docker and fresh
installs**

-   **Severity:** Critical

-   **Status:** Verified (requirements) / Suspected Docker runtime
    > failure

-   **Location:** backend/requirements.txt:1-30;
    > backend/app/core/storage.py:1

-   **Component:** StorageService, chart_generator, app startup

-   **Evidence:** storage.py imports boto3 at module top.
    > requirements.txt lists aioboto3\>=12.0.0 but not boto3. aioboto3
    > declares boto3 only as an optional extra (aiobotocore\[boto3\]),
    > so a clean pip install -r requirements.txt does not install it.
    > app.main imports documents.py, which imports storage.py.

-   **Impact:** A fresh Docker build or clean virtualenv fails at import
    > with ModuleNotFoundError: No module named \'boto3\'. Local tests
    > pass only because boto3 is installed in the audit environment.

-   **Expected behavior:** All runtime imports must be declared in
    > requirements.txt.

-   **Recommended fix:** Add boto3 explicitly (or switch to aioboto3 and
    > use it consistently), pin versions, and add a clean-install smoke
    > test to CI.

-   **Affected:** Backend startup, upload, ingestion, charts, Docker.

-   **Verified/Suspected:** Verified dependency graph; Docker failure
    > inferred from import path.

**CRIT-07 --- Local secrets are baked into Docker images**

-   **Severity:** Critical

-   **Status:** Verified (files/config) / Suspected image contents until
    > built

-   **Location:** backend/Dockerfile:16; no backend/.dockerignore;
    > backend/.env

-   **Component:** Docker build context

-   **Evidence:** Dockerfile ends with COPY . .. No .dockerignore
    > exists. backend/.env exists and contains populated values for
    > GOOGLE_API_KEY, SECRET_KEY, DATABASE_URL, AWS_ACCESS_KEY_ID, and
    > AWS_SECRET_ACCESS_KEY. Values are redacted here. .gitignore
    > ignores .env, but Docker does not.

-   **Impact:** Every locally built backend image contains the
    > developer's real-looking Google API key and JWT secret. Anyone
    > with the image can extract them. The compose stack also
    > bind-mounts ./backend:/app, exposing .env inside the container.

-   **Expected behavior:** Secrets must never enter build contexts or
    > images; use runtime env/secret stores.

-   **Recommended fix:** Add backend/.dockerignore (.env, .venv,
    > \_\_pycache\_\_, .pytest_cache, tests, docs), rotate the exposed
    > Google key and SECRET_KEY, and add a CI secret scan
    > (gitleaks/trufflehog).

-   **Affected:** All deployments, API credentials, JWT signing.

-   **Verified/Suspected:** Verified presence of secrets and COPY . .;
    > image extraction suspected until built.

**CRIT-08 --- Frontend has no real authentication and defaults to
mock-token**

-   **Severity:** Critical

-   **Status:** Verified

-   **Location:** frontend/src/lib/api.ts:19, :135-136, :232-238;
    > frontend/src/app/page.tsx:27-37

-   **Component:** Axios interceptor, watchDocumentStatus,
    > streamMessage, home page

-   **Evidence:** localStorage.getItem(\'token\') ?? \'mock-token\'
    > appears in three places. There is no login/register page, no auth
    > store, no route guard, and no logout. The home page immediately
    > calls createSession(\[\]) and redirects; on failure it redirects
    > to fallback-session-\${Date.now()}.

-   **Impact:** The app is effectively unauthenticated for every user.
    > Even if the backend bypass is removed, the frontend cannot obtain
    > a real token. JWT is stored in localStorage and sent in SSE query
    > strings, exposing it to XSS, logs, and browser history.

-   **Expected behavior:** A real login/register flow, secure token
    > storage, refresh handling, and authenticated SSE (short-lived
    > ticket or cookie).

-   **Recommended fix:** Build auth pages, store tokens in httpOnly
    > cookies or a secure session, remove mock-token defaults, remove
    > ?token= from SSE URLs (use a short-lived one-time ticket), and add
    > route guards.

-   **Affected:** Entire frontend, chat, uploads, sessions.

-   **Verified/Suspected:** Verified.

**5.2 High Findings**

**HIGH-01 --- Chat SSE is fake streaming and omits required fields**

-   **Severity:** High

-   **Status:** Verified

-   **Location:** backend/app/api/v1/chat.py:217-240;
    > backend/app/schemas/chat.py:4-8;
    > frontend/src/components/ChatWindow.tsx:185-197

-   **Component:** stream_message, StreamEvent, ChatWindow

-   **Evidence:** The generator awaits orchestrator.run(\...) to
    > completion, then does words = agent_result.answer.answer.split(\"
    > \"), yields each word, and await asyncio.sleep(0.015). It never
    > calls gemini_client.complete_stream. StreamEvent has only token,
    > done, citations, tokens_used; the frontend reads data.confidence
    > and data.chart_url, which are always undefined.

-   **Impact:** Users wait for the full LLM response before seeing
    > anything, then watch a fake typewriter effect. Confidence and
    > charts never appear. The "real-time" claim is false.

-   **Expected behavior:** Stream actual model deltas; emit a final
    > event with citations, confidence, chart URL, token usage, and
    > cost.

-   **Recommended fix:** Wire GeminiClient.complete_stream (or the SDK
    > stream) through the agent; extend StreamEvent; persist and forward
    > confidence/chart fields; add SSE integration tests.

-   **Affected:** Chat UX, agent, observability, frontend.

-   **Verified/Suspected:** Verified.

**HIGH-02 --- LLM observability path is dead and incompatible with
installed Langfuse**

-   **Severity:** High

-   **Status:** Verified

-   **Location:** backend/app/services/llm/gemini_client.py:28-44,
    > :95-117, :249+; backend/app/services/agents/base_agent.py:18,
    > :141-146; backend/app/api/v1/chat.py:128-135

-   **Component:** GeminiClient, BaseAgent, chat cost calculation

-   **Evidence:** The main agent creates its own genai.Client and calls
    > generate_content directly; it never uses GeminiClient.complete.
    > GeminiClient is imported in chat only for calculate_cost.
    > Langfuse.trace does not exist in installed langfuse 5.1.0
    > (verified by introspection), so start_trace would raise
    > AttributeError if used. LLM_TOKENS, LLM_LATENCY, LLM_COST are
    > incremented only inside the unused GeminiClient methods.

-   **Impact:** No Langfuse traces, no LLM token/cost metrics, no
    > latency metrics, no real cost tracking. The docs claim full
    > observability.

-   **Expected behavior:** One LLM client used by the agent; tracing and
    > metrics recorded on every call; Langfuse SDK pinned to a
    > compatible major version.

-   **Recommended fix:** Refactor BaseAgent to use GeminiClient; update
    > to Langfuse v5 start_span/start_generation API or pin langfuse\<3;
    > record metrics; add a trace smoke test.

-   **Affected:** Agent, observability, cost, eval.

-   **Verified/Suspected:** Verified code and SDK introspection.

**HIGH-03 --- Langfuse service in Docker Compose cannot start**

-   **Severity:** High

-   **Status:** Verified (compose) / Suspected exact failure mode

-   **Location:** docker-compose.yml:78-98

-   **Component:** langfuse service

-   **Evidence:** The service uses langfuse/langfuse:latest with only
    > Postgres (langfuse-db) and DATABASE_URL, NEXTAUTH_SECRET,
    > NEXTAUTH_URL, SALT, and init variables. Modern Langfuse v3+
    > requires ClickHouse, Redis, S3-compatible storage, and
    > ENCRYPTION_KEY; none are provided.

-   **Impact:** Langfuse container exits or is unhealthy. Backend
    > tracing silently degrades (or would fail if used).

-   **Expected behavior:** The compose stack must include all Langfuse
    > dependencies and a pinned image version.

-   **Recommended fix:** Use the official Langfuse v3 compose topology
    > (ClickHouse, Redis, MinIO/S3, ENCRYPTION_KEY) or drop Langfuse
    > from the default stack and document an external instance.

-   **Affected:** Observability, local development.

-   **Verified/Suspected:** Compose contents verified; container failure
    > inferred from Langfuse v3 requirements.

**HIGH-04 --- Celery worker uses asyncio.run with a global async
engine/pool**

-   **Severity:** High

-   **Status:** Verified (code) / Suspected runtime failure

-   **Location:** backend/app/workers/ingestion_worker.py:15-17, :90-95;
    > backend/app/db/database.py:6-18

-   **Component:** process_document, AsyncSessionLocal

-   **Evidence:** Each Celery task calls
    > asyncio.run(\_process_document_async(doc_id)). AsyncSessionLocal
    > is bound to a module-level create_async_engine with a pooled
    > asyncpg connection pool. After the first asyncio.run closes its
    > event loop, pooled connections can remain bound to the closed
    > loop; subsequent tasks may fail with "attached to a different
    > loop" or "Event loop is closed."

-   **Impact:** The first document may ingest; later documents fail
    > intermittently. This is a classic Celery + async SQLAlchemy bug.

-   **Expected behavior:** Each task must own a fresh engine with
    > NullPool, or the worker must run a persistent event loop and
    > dispose connections per task.

-   **Recommended fix:** Use poolclass=NullPool for worker sessions, or
    > create/dispose an engine inside \_process_document_async; add a
    > test that processes two documents sequentially in one worker
    > process.

-   **Affected:** Ingestion, Celery, Postgres.

-   **Verified/Suspected:** Code pattern verified; failure suspected.

**HIGH-05 --- Document deletion leaves Qdrant vectors and chunks
behind**

-   **Severity:** High

-   **Status:** Verified

-   **Location:** backend/app/api/v1/documents.py:96-113;
    > backend/app/services/retrieval/vector_store.py:63-75

-   **Component:** delete_document, VectorStore.delete_document

-   **Evidence:** The API deletes the S3 object and the Document row
    > (chunks cascade in Postgres), but never calls
    > vector_store.delete_document. That method exists and is unused.

-   **Impact:** Deleted documents remain retrievable from Qdrant. Stale
    > vectors waste storage and can leak deleted content. Re-uploading
    > the same document creates duplicates.

-   **Expected behavior:** Deleting a document removes S3 object,
    > Postgres rows, and all Qdrant points atomically or with
    > retry/compensation.

-   **Recommended fix:** Call await
    > vector_store.delete_document(str(doc_id)) before/after DB delete;
    > add an outbox/retry mechanism; add an integration test asserting
    > Qdrant count drops to zero.

-   **Affected:** Documents, retrieval, storage.

-   **Verified/Suspected:** Verified.

**HIGH-06 --- Semantic cache is global/cross-user and never
invalidated**

-   **Severity:** High

-   **Status:** Verified

-   **Location:** backend/app/services/memory/semantic_cache.py:19-30,
    > :38-53, :79-102, :104-110

-   **Component:** SemanticCache

-   **Evidence:** \_generate_key(\[\]) returns semantic_cache:global.
    > The key is derived only from sorted doc_ids, never user_id. get()
    > embeds the query synchronously and returns a cached AgentAnswer
    > including citations. invalidate() is a no-op (pass). Cache entries
    > are stored as Redis hashes with a 1-hour TTL.

-   **Impact:** Users with no selected documents share one global cache.
    > A cached answer generated for user A can be returned to user B,
    > including citations and document excerpts. Deleting/reindexing a
    > document does not invalidate cached answers.

-   **Expected behavior:** Cache keys must include user_id and a
    > corpus/version fingerprint; invalidation must remove affected
    > entries.

-   **Recommended fix:** Include user_id in the key; use a versioned
    > corpus hash; implement invalidate(doc_id) via an index set;
    > disable cache when no documents are selected; add a cross-user
    > cache test.

-   **Affected:** Chat, retrieval, security, privacy.

-   **Verified/Suspected:** Verified.

**HIGH-07 --- Duplicate Redis history writes on streamed messages**

-   **Severity:** High

-   **Status:** Verified

-   **Location:**
    > backend/app/services/agents/reasoning_agent.py:188-192;
    > backend/app/api/v1/chat.py:247-252

-   **Component:** ReasoningAgent.run, stream_message event generator

-   **Evidence:** ReasoningAgent.run calls
    > session_memory.save_turn(session_id, user_msg, ai_msg). The stream
    > generator then calls session_memory.save_turn(\...) again with the
    > same user query and assistant answer.

-   **Impact:** Every streamed turn is stored twice in Redis. The next
    > prompt contains duplicated context, increasing token usage and
    > degrading answer quality. History UI also shows duplicates until
    > Postgres fallback.

-   **Expected behavior:** Exactly one persistence path per turn.

-   **Recommended fix:** Remove the duplicate save from either the agent
    > or the route; make persistence a single service call after the
    > final event; add a test asserting LRANGE length increments by 2
    > per turn.

-   **Affected:** Chat memory, token usage, UX.

-   **Verified/Suspected:** Verified.

**HIGH-08 --- Missing tool dependencies silently disable Python, PII,
and chart features**

-   **Severity:** High

-   **Status:** Verified

-   **Location:** backend/requirements.txt:1-30;
    > backend/app/services/tools/python_executor.py:6-12;
    > backend/app/services/guardrails/pii_detector.py:3-17;
    > backend/app/services/tools/chart_generator.py:4-7

-   **Component:** PythonExecutorTool, PIIDetector, ChartGeneratorTool

-   **Evidence:** RestrictedPython, presidio-analyzer,
    > presidio-anonymizer, and matplotlib are imported inside try/except
    > ImportError but are not listed in requirements.txt. Local probe
    > confirmed all four are absent. python_executor then raises
    > NameError on compile_restricted; pii_detector sets self.analyzer =
    > None; chart_generator returns "Matplotlib not installed."

-   **Impact:** Data-analysis tool calls always fail, PII redaction is
    > silently skipped, and chart generation always fails. The docs
    > claim all three are implemented.

-   **Expected behavior:** Required runtime features must have declared,
    > pinned dependencies and fail loudly at startup if unavailable.

-   **Recommended fix:** Add RestrictedPython, presidio-analyzer,
    > presidio-anonymizer, matplotlib, and the required spaCy model; add
    > startup capability checks; add tests that execute each tool.

-   **Affected:** Agent tools, guardrails, UX, security.

-   **Verified/Suspected:** Verified dependency absence.

**HIGH-09 --- python_executor timeout does not actually stop execution;
stdout is process-global**

-   **Severity:** High

-   **Status:** Verified (code) / Suspected hang

-   **Location:** backend/app/services/tools/python_executor.py:61-78

-   **Component:** PythonExecutorTool.execute

-   **Evidence:** future.result(timeout=self.timeout) raises
    > TimeoutError, but the with concurrent.futures.ThreadPoolExecutor()
    > block then calls shutdown(wait=True) on exit. An infinite loop
    > (while True: pass) keeps the worker thread alive forever, so the
    > request hangs despite the "10s timeout." sys.stdout is replaced
    > globally, creating a race across concurrent requests.

-   **Impact:** A malicious or accidental infinite loop can exhaust the
    > thread pool and hang the API. Concurrent tool calls can capture
    > each other's output.

-   **Expected behavior:** Sandboxed execution must be killable and
    > output must be isolated per execution.

-   **Recommended fix:** Use a separate process with a hard kill
    > (multiprocessing/subprocess) and per-process pipes; or a sandbox
    > service. Do not rely on thread timeouts for untrusted code.

-   **Affected:** Agent, API availability, security.

-   **Verified/Suspected:** Code logic verified; hang suspected but
    > mechanically certain for infinite loops.

**HIGH-10 --- Guardrails are both non-functional and overzealous**

-   **Severity:** High

-   **Status:** Verified

-   **Location:** backend/app/services/guardrails/pii_detector.py:3-25;
    > content_filter.py:14-36; guardrails_service.py:44-49;
    > hallucination_checker.py:24-25; backend/app/api/v1/chat.py:217-225

-   **Component:** PIIDetector, ContentFilter, GuardrailsService,
    > HallucinationChecker, stream route

-   **Evidence:** If Presidio is missing, scan_and_redact returns the
    > original text with has_pii=False. unsafe_keywords includes
    > \"attack\", \"kill\", \"bomb\", so legitimate security/finance
    > answers are blocked. injection_keywords includes \"system prompt\"
    > and \"act as a\", causing false positives. check_output raises
    > GuardrailBlockError on unsafe output; stream_message does not
    > catch it inside event_generator, so the error occurs after SSE
    > headers are sent. HallucinationChecker.check returns {\"score\":
    > 1.0} when source_chunks is empty, marking ungrounded answers as
    > perfectly grounded.

-   **Impact:** PII may leak; valid answers are blocked; blocked streams
    > fail ungracefully; hallucination scores are meaningless when
    > retrieval returns nothing.

-   **Expected behavior:** PII redaction must be deterministic or
    > explicitly disabled with a warning; content filtering must be
    > context-aware; guardrail errors must be returned before SSE starts
    > or as a structured SSE error event; hallucination with no sources
    > must score 0 or be flagged.

-   **Recommended fix:** Add Presidio + spaCy model; replace substring
    > blocklists with classifier/allowlist rules; catch
    > GuardrailBlockError in the generator and emit an error event;
    > return 0.0/flag when no sources; add tests for false positives and
    > empty-source cases.

-   **Affected:** Chat, security, UX, eval.

-   **Verified/Suspected:** Verified.

**HIGH-11 --- Agent returns fake confidence and token counts on parse
failure**

-   **Severity:** High

-   **Status:** Verified

-   **Location:** backend/app/services/agents/reasoning_agent.py:169-180

-   **Component:** ReasoningAgent.run

-   **Evidence:** On exception during output parsing, fallback returns
    > random confidence and hardcoded token usage:

> except Exception:\
> import random\
> structured_answer = AgentAnswer(\
> answer=str(base_result.answer),\
> citations=self.citations,\
> confidence=random.uniform(0.7, 0.99),\
> tokens_used=120\
> )

-   **Impact:** When the model returns non-JSON or a schema-invalid
    > answer, the UI shows a random 70--99% confidence and a fabricated
    > 120-token count. Cost metrics and eval are polluted.

-   **Expected behavior:** Parse failure should be surfaced as an error
    > or a clearly marked "unstructured" response with confidence 0 and
    > real token usage.

-   **Recommended fix:** Remove random values; use validate_and_retry;
    > record actual usage_metadata; mark parse failures in
    > guardrail_flags; add a test with malformed JSON.

-   **Affected:** Chat, cost, eval, trust.

-   **Verified/Suspected:** Verified.

**HIGH-12 --- BM25, summary seeding, and eval read only 200-character
previews**

-   **Severity:** High

-   **Status:** Verified

-   **Location:** backend/app/workers/ingestion_worker.py:67;
    > backend/app/models/chunk.py:14, :21-23;
    > backend/app/services/retrieval/hybrid_retriever.py:57-69;
    > backend/app/services/agents/reasoning_agent.py:103;
    > backend/app/services/eval/eval_runner.py:48

-   **Component:** Ingestion, Chunk, HybridRetriever, ReasoningAgent,
    > EvalRunner

-   **Evidence:** Ingestion stores only
    > text_preview=chunk\[\"text\"\]\[:200\]. Chunk.text is a property
    > returning text_preview. BM25 builds its corpus from getattr(c,
    > \"text_preview\", \"\"). Summary seeding and eval use c.text,
    > which is the same 200-character preview. Only Qdrant payloads
    > contain full chunk text.

-   **Impact:** Keyword search cannot match terms beyond the first 200
    > characters. Summaries and eval judge only see previews. Retrieval
    > quality is severely degraded.

-   **Expected behavior:** Store full chunk text in Postgres (or fetch
    > it from Qdrant) for BM25, summaries, and eval.

-   **Recommended fix:** Add a text column (or store full text in Chunk)
    > and use it everywhere; keep text_preview only for UI/debug;
    > backfill existing chunks; add a retrieval test with a keyword
    > appearing after character 200.

-   **Affected:** Retrieval, summaries, eval, citations.

-   **Verified/Suspected:** Verified.

**HIGH-13 --- Synchronous I/O and model loading block the async event
loop**

-   **Severity:** High

-   **Status:** Verified (code) / Suspected latency impact

-   **Location:** backend/app/api/v1/documents.py:39-47;
    > backend/app/core/storage.py:32-48;
    > backend/app/services/retrieval/hybrid_retriever.py:48, :76;
    > backend/app/services/retrieval/reranker.py:11-12, :46;
    > backend/app/services/eval/eval_runner.py:51;
    > backend/app/services/memory/semantic_cache.py:53, :84

-   **Component:** Upload, retrieval, reranker, cache, eval

-   **Evidence:** StorageService.upload_file uses boto3 synchronously in
    > an async route. embedding_service.embed_text/embed_batch uses the
    > sync Google SDK. CrossEncoder.predict is sync. llm_judge.score is
    > sync inside async eval. reranker = CrossEncoderReranker() executes
    > at module import and downloads
    > cross-encoder/ms-marco-MiniLM-L-6-v2; importing app.main printed
    > Hugging Face "Loading weights."

-   **Impact:** A single upload or retrieval blocks the event loop for
    > all requests. Startup depends on Hugging Face availability;
    > offline/air-gapped deployments hang or fail.

-   **Expected behavior:** Offload blocking work with
    > asyncio.to_thread/executor or use async clients; pre-download
    > models at build time; lazy-load with a readiness check.

-   **Recommended fix:** Use aioboto3 or to_thread for S3; use async
    > embedding calls or to_thread; offload reranker/judge; pre-download
    > the CrossEncoder model in the Dockerfile; add timeouts.

-   **Affected:** Upload, chat latency, eval, startup.

-   **Verified/Suspected:** Code verified; latency impact suspected.

**HIGH-14 --- Upload endpoint lacks size, magic-byte, and content
validation**

-   **Severity:** High

-   **Status:** Verified

-   **Location:** backend/app/api/v1/documents.py:29-47

-   **Component:** upload_document

-   **Evidence:** file_bytes = await file.read() reads the entire upload
    > into memory. There is no Content-Length/size limit, no magic-byte
    > check, no virus scan, and file.content_type is trusted from the
    > client. The frontend has a 100 MB maxSize, but the API does not.

-   **Impact:** A client can upload arbitrarily large files (memory
    > exhaustion), spoof content_type to bypass parser selection, or
    > upload malicious content. The 100 MB client limit is not a
    > security control.

-   **Expected behavior:** Enforce a server-side size limit, validate
    > magic bytes, reject mismatched MIME types, stream to disk/S3, and
    > scan files.

-   **Recommended fix:** Add MAX_UPLOAD_BYTES; use UploadFile streaming
    > with a bounded reader; validate with python-magic; add ClamAV or a
    > scanning hook; return 413 for oversized files.

-   **Affected:** API availability, security, ingestion.

-   **Verified/Suspected:** Verified.

**HIGH-15 --- Evaluation pipeline is non-deterministic and has no
ownership**

-   **Severity:** High

-   **Status:** Verified

-   **Location:** backend/app/services/eval/llm_judge.py:56-64;
    > backend/app/services/eval/eval_dataset.py:50-60;
    > backend/app/services/eval/eval_runner.py:29-55;
    > backend/app/api/v1/eval.py:47-102;
    > backend/app/models/eval_result.py:8-23

-   **Component:** LLMJudge, EvalDatasetGenerator, EvalRunner, eval
    > routes, EvalRun

-   **Evidence:** On API error, llm_judge returns random scores
    > 0.6--1.0. eval_dataset falls back to mock Q&A on API error (per
    > summary/code path). eval_runner passes all document chunks to the
    > judge, not top-k, and calls sync llm_judge.score. Eval endpoints
    > have no user_id check; EvalRun has no user_id column.
    > run_eval_background uses FastAPI BackgroundTasks, not Celery.

-   **Impact:** Eval scores are fabricated on failure, context is
    > bloated, any user can read any eval run, and long evals tie up the
    > API process.

-   **Expected behavior:** Real judge only; top-k context; ownership
    > checks; Celery/background worker; user_id on eval runs.

-   **Recommended fix:** Remove random/mock fallbacks; add user_id;
    > filter by owner; pass retrieved top-k; move to Celery; add eval
    > integration tests.

-   **Affected:** Eval, security, observability.

-   **Verified/Suspected:** Verified.

**HIGH-16 --- MCP server calls a nonexistent endpoint and sends ignored
fields**

-   **Severity:** High

-   **Status:** Verified

-   **Location:** backend/mcp_server/server.py:55, :91-157, :194-205,
    > :260-268

-   **Component:** search_documents, summarize_document, ask_question

-   **Evidence:** search_documents posts to /retrieval/search, which
    > does not exist in the FastAPI routes.
    > summarize_document/ask_question send doc_ids in /chat/message
    > payloads, but ChatMessageRequest only defines session_id and
    > query; Pydantic ignores extra fields. user_id arguments are
    > accepted but ignored by the backend. Default token is mock-token.

-   **Impact:** search_documents always 404s.
    > summarize_document/ask_question may work only if the created
    > session already has the document IDs; otherwise they ignore the
    > requested document scope. MCP is not usable as documented.

-   **Expected behavior:** Use real endpoints and payload fields;
    > propagate user_id securely; require a real token.

-   **Recommended fix:** Add a /retrieval/search endpoint or call
    > /chat/message with a session-scoped doc list; add doc_ids to
    > ChatMessageRequest and validate ownership; remove mock-token; add
    > MCP smoke tests.

-   **Affected:** MCP integrations, chat.

-   **Verified/Suspected:** Verified.

**HIGH-17 --- CORS wildcard subdomain is unsupported and credentials are
enabled**

-   **Severity:** High

-   **Status:** Verified

-   **Location:** backend/app/main.py:73-79

-   **Component:** CORSMiddleware

-   **Evidence:** allow_origins=\[\"http://localhost:3000\",
    > \"https://\*.railway.app\"\]. Starlette/FastAPI compares origins
    > literally; https://\*.railway.app does not match
    > https://quantiva.up.railway.app. allow_credentials=True is set
    > with a wildcard pattern.

-   **Impact:** A deployed Railway frontend cannot call the backend. If
    > the wildcard were interpreted, credentials would be unsafe.

-   **Expected behavior:** Explicit production origins from
    > configuration; no wildcard with credentials.

-   **Recommended fix:** Add CORS_ORIGINS env var with exact origins;
    > validate at startup; add a preflight test.

-   **Affected:** Deployment, frontend/backend integration.

-   **Verified/Suspected:** Verified.

**HIGH-18 --- Fallback session IDs are invalid UUIDs; history can return
the wrong session**

-   **Severity:** High

-   **Status:** Verified

-   **Location:** frontend/src/app/page.tsx:36;
    > frontend/src/app/(dashboard)/chat/\[sessionId\]/page.tsx:66,
    > :276-281; backend/app/api/v1/chat.py:65-80, :170-182, :293-302

-   **Component:** Home page, session page, send_message,
    > stream_message, get_history

-   **Evidence:** Frontend creates fallback-session-\${Date.now()},
    > which is not a UUID. Backend send_message/stream_message catch the
    > UUID parse failure, set chat_session=None, and silently create a
    > new session with a new UUID. get_history catches ValueError and
    > falls back to the latest session for the user, returning a
    > different conversation's history.

-   **Impact:** Users see an empty chat, messages are saved to a
    > different session, and history can display another session's
    > content. This is also a data-leak risk within a user account.

-   **Expected behavior:** Reject invalid session IDs with 400/404;
    > never silently substitute a different session.

-   **Recommended fix:** Validate session IDs at the API boundary;
    > remove fallback-session routing; return 404 for unknown sessions;
    > add tests for invalid IDs.

-   **Affected:** Chat UX, history, security.

-   **Verified/Suspected:** Verified.

**HIGH-19 --- Frontend/Docker/Railway deployment is not
production-grade**

-   **Severity:** High

-   **Status:** Verified

-   **Location:** docker-compose.yml:199-208; frontend/package.json:5-9;
    > frontend/src/lib/api.ts:6-7; railway.json:1-13;
    > backend/Dockerfile:1-20

-   **Component:** Frontend service, Railway config, API base URL

-   **Evidence:** Compose runs node:22-alpine with npm install && npm
    > run dev, bind-mounts the source, and sets
    > NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1. There is no
    > frontend Dockerfile or production build. railway.json only builds
    > backend/Dockerfile; no frontend, Qdrant, Redis, or S3 provisioning
    > is defined. Railway DATABASE_URL is typically postgresql://, not
    > postgresql+asyncpg://.

-   **Impact:** The frontend runs a development server in production,
    > rebuilds dependencies on every start, and cannot be deployed as a
    > standalone artifact. Railway deployment is backend-only and likely
    > fails to connect to Postgres due to the async driver scheme.

-   **Expected behavior:** Multi-stage frontend build, npm ci,
    > production next start or static export, explicit env config, and a
    > full Railway service topology.

-   **Recommended fix:** Add frontend/Dockerfile; use npm ci && npm run
    > build; run next start; parameterize API URL; add Railway services
    > for Postgres/Redis/Qdrant/S3; normalize DATABASE_URL.

-   **Affected:** Deployment, frontend, Railway.

-   **Verified/Suspected:** Verified config; Railway runtime failure
    > suspected.

**HIGH-20 --- Health and metrics endpoints leak details and are
unauthenticated**

-   **Severity:** High

-   **Status:** Verified

-   **Location:** backend/app/main.py:88-128;
    > backend/app/api/v1/metrics_endpoint.py:9-14

-   **Component:** health_check, /metrics

-   **Evidence:** /health returns raw exception strings (f\"down:
    > {exc}\") for Postgres/Redis/Qdrant. It creates a new
    > AsyncQdrantClient per request and never closes it. /metrics is
    > intentionally unauthenticated and exposes HTTP/LLM/cache metrics.

-   **Impact:** Unauthenticated attackers can enumerate internal
    > hostnames, ports, driver errors, and possibly credentials embedded
    > in connection errors. Repeated /health calls leak Qdrant clients.

-   **Expected behavior:** Health should return generic statuses
    > publicly; detailed diagnostics should require auth or be
    > internal-only. Reuse a singleton Qdrant client and close it.

-   **Recommended fix:** Return \"down\" without exception text; add an
    > authenticated /health/details; protect /metrics at the
    > proxy/network layer; use a shared Qdrant client; add a health
    > test.

-   **Affected:** Security, observability, resources.

-   **Verified/Suspected:** Verified.

**HIGH-21 --- Rate limiting covers only one endpoint and is trivially
bypassed**

-   **Severity:** High

-   **Status:** Verified

-   **Location:** backend/app/api/v1/chat.py:57-62;
    > backend/app/core/rate_limiter.py:19-70;
    > backend/app/api/v1/auth.py:13-51

-   **Component:** rate_limiter, send_message, auth routes

-   **Evidence:** Only POST /chat/message calls rate_limiter.check. The
    > SSE stream endpoint has no limit. Tier is derived from
    > current_user.email.endswith(\"@quantiva.ai\"), not User.tier.
    > Redis errors fail open (allowed=True). Login/register have no rate
    > limit. There is no per-IP limit.

-   **Impact:** Attackers can bypass limits via the stream endpoint,
    > brute-force login, or exhaust resources. A Redis outage disables
    > all limiting.

-   **Expected behavior:** Apply limits to all expensive/auth endpoints,
    > use User.tier, fail closed or degrade safely, and add per-IP/login
    > limits.

-   **Recommended fix:** Add a reusable dependency/middleware; include
    > SSE, upload, login, and register; use User.tier; add a per-IP
    > bucket; log fail-open events; test with Redis down.

-   **Affected:** Security, availability, auth.

-   **Verified/Suspected:** Verified.

**HIGH-22 --- JWT refresh/secret handling is incomplete and unsafe by
default**

-   **Severity:** High

-   **Status:** Verified

-   **Location:** backend/app/core/security.py:18-27;
    > backend/app/api/v1/auth.py:30-31, :48-49;
    > backend/app/config.py:25; docker-compose.yml:151, :186

-   **Component:** create_access_token, decode_token, auth routes,
    > settings

-   **Evidence:** Access and refresh tokens are the same JWT type with
    > only different exp; there is no type claim, no refresh endpoint,
    > no rotation, no revocation, and no is_active check. SECRET_KEY
    > defaults to dev-secret-key-quantiva-change-in-prod; compose
    > defaults to your-dev-secret-key-change-in-prod. There is no
    > startup rejection in production.

-   **Impact:** A refresh token can be used as an access token. A stolen
    > token cannot be revoked. If deployed with defaults, anyone can
    > forge tokens. Disabled users can still authenticate.

-   **Expected behavior:** Separate access/refresh token types, refresh
    > rotation/revocation, strong secret enforcement, is_active checks.

-   **Recommended fix:** Add type/jti claims; implement /auth/refresh
    > and /auth/logout; store refresh token hashes/revocation list;
    > reject default secrets when APP_ENV=production; check is_active in
    > get_current_user; add auth tests.

-   **Affected:** Authentication, security, deployment.

-   **Verified/Suspected:** Verified.

**HIGH-23 --- Tests and CI are inadequate and do not exercise the real
pipeline**

-   **Severity:** High

-   **Status:** Verified

-   **Location:** backend/tests/; backend/requirements-test.txt;
    > .github/workflows/ci.yml:58-80;
    > backend/tests/integration/test_document_endpoints.py;
    > backend/tests/e2e/test_full_rag_pipeline.py

-   **Component:** Test suite, CI

-   **Evidence:** 48 tests pass, but they mock the agent, ingestion,
    > storage, Qdrant, and LLM. test_document_endpoints.py is 0 lines.
    > requirements-test.txt is 0 lines. CI installs pytest
    > pytest-asyncio but not aiosqlite, while tests/conftest.py uses
    > sqlite+aiosqlite:///:memory:. CI does not run frontend lint/build.
    > mypy is continue-on-error. E2E uses a fake PDF and mocks
    > ingestion. test_gemini_client explicitly tests fallback behavior.

-   **Impact:** CI can pass while production is broken. A clean CI run
    > likely fails at aiosqlite. No test covers real ingestion, Qdrant,
    > retrieval, agents, guardrails, SSE, MCP, or eval.

-   **Expected behavior:** Tests should cover real service integrations
    > (or contract tests), and CI must install all test deps and run
    > frontend checks.

-   **Recommended fix:** Add aiosqlite and test deps; implement document
    > endpoint tests; add Postgres/Redis/Qdrant integration tests; add
    > SSE/agent/tool/eval/MCP tests; run npm ci, npm run lint, npm run
    > build in CI; make mypy blocking.

-   **Affected:** Quality, release confidence.

-   **Verified/Suspected:** Verified.

**HIGH-24 --- Documentation materially overstates implementation**

-   **Severity:** High

-   **Status:** Verified

-   **Location:** README.md:13, :26, :45, :103;
    > quantiva_learn/2-PROJECT-OVERVIEW.md:9, :40, :113, :126-146;
    > quantiva_learn/10-API-DOCUMENTATION.md:158-160, :233, :248;
    > quantiva_learn/16-TOOLS-AND-INTEGRATIONS.md:119-135;
    > quantiva_learn/17-GUARDRAILS-AND-SECURITY.md:129-131;
    > quantiva_learn/19-LOGGING-AND-OBSERVABILITY.md:30-42

-   **Component:** Documentation

-   **Evidence:** Docs claim Next.js 14 (actual 16.3.8), Gemini 1.5
    > Flash (retired), token-by-token streaming (fake),
    > RestrictedPython/Presidio/Matplotlib (deps missing), Langfuse
    > tracing on every call (dead path), Prometheus/Grafana complete
    > (partial), /api/v1/eval/generate-dataset and /eval/run (missing/no
    > prefix), and full-stack Railway (backend only).

-   **Impact:** Operators and reviewers cannot trust the docs. Setup and
    > acceptance criteria are misleading.

-   **Expected behavior:** Docs must match code and clearly label
    > planned/partial features.

-   **Recommended fix:** Add a "current status" matrix; fix endpoint
    > paths; document model retirement and replacement; mark partial
    > features; add a docs-vs-code CI check for route names.

-   **Affected:** Onboarding, operations, evaluation.

-   **Verified/Suspected:** Verified.

**HIGH-25 --- Chart URLs use internal S3 host and are blocked by
Next/Image config**

-   **Severity:** High

-   **Status:** Verified (code/config) / Suspected browser failure

-   **Location:** backend/app/services/tools/chart_generator.py:58-70;
    > frontend/next.config.ts:4-12; backend/app/core/storage.py:11-17

-   **Component:** ChartGeneratorTool, Next image config

-   **Evidence:** generate_presigned_url uses the same s3_client
    > configured with S3_ENDPOINT_URL=http://s3:9090 in Docker. The URL
    > host is s3, which a browser cannot resolve. Next/Image
    > remotePatterns allows only http://localhost:8000, not
    > localhost:9090 or s3:9090.

-   **Impact:** Even if Matplotlib were installed, charts would not
    > render in the browser. The MessageBubble \<Image\> would throw a
    > config error or fail to load.

-   **Expected behavior:** Generate browser-reachable URLs (public S3
    > endpoint or backend proxy) and allow the host in Next config.

-   **Recommended fix:** Add S3_PUBLIC_ENDPOINT_URL for presigning;
    > proxy chart downloads through the backend; add the host to
    > remotePatterns; test chart rendering.

-   **Affected:** Chart tool, UX.

-   **Verified/Suspected:** Code/config verified; browser failure
    > suspected.

**HIGH-26 --- Frontend rendering, lint, and citation UX are incomplete**

-   **Severity:** High

-   **Status:** Verified

-   **Location:** frontend/src/components/MessageBubble.tsx:146-172,
    > :211-218, :281, :108-111;
    > frontend/src/app/(dashboard)/chat/\[sessionId\]/page.tsx:280;
    > frontend/src/app/page.tsx:23;
    > frontend/src/components/ChatWindow.tsx:119-132;
    > frontend/src/lib/api.ts:229

-   **Component:** StreamingText, CacheHitBadge, CitationPanel, chat
    > page, ChatWindow

-   **Evidence:** StreamingText renders raw text in \<span\> elements;
    > no Markdown parser, so headings, lists, bold, tables, and code
    > blocks appear as literal Markdown. CacheHitBadge is defined but
    > never used. "Open document" has no onClick. npm run lint fails
    > with 6 errors: react-hooks/set-state-in-effect at chat page line
    > 280, prefer-const at page.tsx:23, three
    > react/no-unescaped-entities/no-explicit-any in MessageBubble.tsx
    > (97, 247, 286). TypeScript passes.

-   **Impact:** Answers look unpolished and are hard to read; citations
    > cannot open documents; lint blocks CI adoption.

-   **Expected behavior:** Render Markdown safely (e.g.,
    > react-markdown + remark-gfm + sanitization); wire
    > citation/document actions; fix lint errors.

-   **Recommended fix:** Add Markdown rendering with sanitization;
    > implement "Open document" (signed URL or viewer); remove unused
    > badge or display cache status; fix lint errors and add frontend
    > lint to CI.

-   **Affected:** UX, accessibility, CI.

-   **Verified/Suspected:** Verified via lint output and code.

**5.3 Medium Findings**

  ------------------------------------------------------------------------------------------------------------------------------------------------------
  **ID**       **Severity**   **Location / Component**          **Issue**                   **Expected / Fix**            **Affected**      **Status**
  ------------ -------------- --------------------------------- --------------------------- ----------------------------- ----------------- ------------
  **MED-01**   Medium         chat.py:38, :65-80, :170-182 ---  uuid.UUID(d) raises         Validate IDs and return       Chat, UX          Verified
                              create_session, send_message,     uncaught ValueError in      400/404; never auto-create on                   
                              stream_message                    create_session; invalid     invalid input.                                  
                                                                session IDs silently                                                        
                                                                auto-create a new session                                                   
                                                                in send/stream.                                                             

  **MED-02**   Medium         documents.py:62-67 --- upload     process_document.delay      Retain task refs or use a     Ingestion,        Verified
                              fallback                          failure falls back to       durable queue; fail upload if availability      
                                                                asyncio.create_task(\...)   enqueue fails.                                  
                                                                without retaining a task                                                    
                                                                reference; task can be                                                      
                                                                garbage-collected; work                                                     
                                                                runs in the web process and                                                 
                                                                blocks the event loop.                                                      

  **MED-03**   Medium         app/core/metrics.py:69-98;        CACHE_HITS, CACHE_MISSES,   Instrument cache, streams,    Observability     Verified
                              chat.py; semantic_cache.py        ACTIVE_STREAMS,             ingestion, retrieval.                           
                                                                DOCS_INGESTED,                                                              
                                                                RETRIEVAL_LATENCY are                                                       
                                                                defined but never                                                           
                                                                incremented.                                                                

  **MED-04**   Medium         main.py:88-128 --- health_check   Health does not check S3;   Add S3 check, return 503 or   Ops, security     Verified
                                                                returns HTTP 200 even when  explicit degraded contract,                     
                                                                degraded; creates an        reuse/close client, redact                      
                                                                unclosed AsyncQdrantClient  errors.                                         
                                                                per request; leaks                                                          
                                                                exception text.                                                             

  **MED-05**   Medium         requirements.txt;                 Versions are mostly         Pin versions, declare         Builds, ops       Verified
                              docker-compose.yml:101-103;       unpinned; tabulate is only  tabulate, align ports/docs.                     
                              README.md:103                     transitive via PyMuPDF;                                                     
                                                                Prometheus is mapped to                                                     
                                                                host 9091 but README says                                                   
                                                                9090.                                                                       

  **MED-06**   Medium         api/v1/eval.py:14, :47, :69, :94; Eval router has no /eval    Add prefix=\"/eval\";         Eval API, docs    Verified
                              docs                              prefix; actual routes are   implement or remove                             
                              10-API-DOCUMENTATION.md:233-248   /api/v1/run,                generate-dataset; update                        
                                                                /api/v1/{run_id},           docs.                                           
                                                                /api/v1/history/list;                                                       
                                                                documented                                                                  
                                                                /eval/generate-dataset does                                                 
                                                                not exist.                                                                  

  **MED-07**   Medium         auth.py:13-51;                    No password policy, no      Add policy, rate limits,      Auth, security    Verified
                              dependencies.py:66-70             login/register rate limit,  active check, generic errors.                   
                                                                no is_active check; login                                                   
                                                                reveals whether email                                                       
                                                                exists via error semantics.                                                 

  **MED-08**   Medium         FileUploader.tsx:217-227,         Deleting an optimistic      Skip API call for temp IDs;   Upload UX         Verified
                              :196-198                          temp-\* document calls      remove failed uploads or                        
                                                                DELETE /documents/temp-\* → offer retry.                                    
                                                                422; failed uploads remain                                                  
                                                                in the UI.                                                                  

  **MED-09**   Medium         useChatStore.ts:33, :108-127;     streamingCitations is never Persist                       Chat UX           Verified
                              chat page :286-293                populated; history messages citations/confidence/chart;                     
                                                                get timestamp: Date.now()   use backend created_at;                         
                                                                instead of created_at;      populate streaming citations.                   
                                                                final events with no                                                        
                                                                citations default to empty.                                                 

  **MED-10**   Medium         chunker.py:16-20;                 Chunk size is 512           Use token-based sizing or     Ingestion quality Verified
                              ingestion_service.py:18-23        characters, not tokens; PDF document the character                          
                                                                text and tables are         choice; deduplicate                             
                                                                combined and may duplicate  table/text; preserve table                      
                                                                content; tables are split   chunks.                                         
                                                                arbitrarily.                                                                

  **MED-11**   Medium         prompt_builder.py;                build_rag_prompt imported   Remove dead code or wire it   Agent,            Verified
                              memory_pager.py;                  but unused; memory_pager    in; update model limits.      maintainability   
                              output_parser.py:38-73;           unused; validate_and_retry                                                  
                              context_manager.py:14-18;         unused; context limits                                                      
                              chat.py:19                        still reference                                                             
                                                                gemini-1.5-flash.                                                           

  **MED-12**   Medium         db/database.py:6-9;               echo=True in all            Make echo configurable; use   Performance,      Verified
                              orchestrator.py:33-40;            environments; debug print   logging; remove               maintainability   
                              base_agent.py:149;                statements; production code test-detection hacks.                           
                              chat.py:117-121, :220-225         detects unittest.mock                                                       
                                                                objects to choose a code                                                    
                                                                path.                                                                       
  ------------------------------------------------------------------------------------------------------------------------------------------------------

**5.4 Low Findings**

  --------------------------------------------------------------------------------------------------------------------------------------------------
  **ID**       **Severity**   **Location / Component**                             **Issue**           **Expected /      **Affected**   **Status**
                                                                                                       Fix**                            
  ------------ -------------- ---------------------------------------------------- ------------------- ----------------- -------------- ------------
  **LOW-01**   Low            backend/Dockerfile:1-20                              Container runs as   Add non-root user Container      Verified
                                                                                   root; no            and healthcheck.  security       
                                                                                   HEALTHCHECK; no                                      
                                                                                   non-root user.                                       

  **LOW-02**   Low            frontend/next.config.ts:4-12                         remotePatterns      Add chart host or Chart UX       Verified
                                                                                   allows only         proxy.                           
                                                                                   localhost:8000;                                      
                                                                                   chart host missing.                                  

  **LOW-03**   Low            frontend/package.json:11-23                          next-auth installed Remove unused     Frontend build Verified
                                                                                   but unused;         deps, pin/lock,                  
                                                                                   lucide-react        use npm ci.                      
                                                                                   \^1.51.0 and other                                   
                                                                                   ranges are broad;                                    
                                                                                   Docker uses npm                                      
                                                                                   install instead of                                   
                                                                                   npm ci.                                              

  **LOW-04**   Low            README.md:18, :26, :45                               "Live Demo" link is Update docs or    Docs           Verified
                                                                                   a placeholder;      remove claims.                   
                                                                                   "token by token"                                     
                                                                                   and "Next.js 14"                                     
                                                                                   are inaccurate.                                      

  **LOW-05**   Low            main.py:73-79                                        No security headers Add               Web security   Verified
                                                                                   (CSP, HSTS,         security-header                  
                                                                                   X-Frame-Options,    middleware.                      
                                                                                   Referrer-Policy).                                    

  **LOW-06**   Low            frontend/src/components/MessageBubble.tsx:108-111,   "Open document" has Implement viewer  UX             Verified
                              :211-218                                             no handler;         or remove button;                
                                                                                   CacheHitBadge       display cache                    
                                                                                   unused.             status.                          
  --------------------------------------------------------------------------------------------------------------------------------------------------

**6. Security Findings (Consolidated)**

**6.1 Threat model**

The app assumes a multi-tenant SaaS deployment: users own documents and
sessions, and must never access another user's data. The current code
does not enforce that boundary consistently.

**6.2 Critical attack paths**

**1. Authentication bypass ---** Authorization: Bearer mock-token
returns the first DB user or creates demo@quantiva.ai with tier=\"pro\"
(dependencies.py:37-52). This is exploitable without any credentials.

**2. Cross-tenant history read ---** GET /chat/sessions/{uuid}/history
returns Redis history by raw session ID before checking ownership
(chat.py:288-290). An attacker who knows a UUID can read another
tenant's conversation.

**3. Cross-tenant document query ---** POST /chat/sessions accepts
arbitrary document UUIDs; vector_store.search filters only by doc_id,
not user_id (chat.py:38, vector_store.py:77-91). An attacker can attach
and query another tenant's documents.

**4. Cross-tenant eval read ---** GET /api/v1/{run_id} and
/api/v1/history/list have no ownership checks (eval.py:69-102).

**5. Secret exposure ---** backend/.env contains a populated Google API
key and SECRET_KEY; Dockerfile uses COPY . . with no .dockerignore
(Dockerfile:16). Rotate both.

**6. Token leakage ---** JWT is sent as a query parameter for both chat
and document-status SSE (api.ts:135-139, :232-238), exposing it to logs,
proxies, and browser history.

**7. Default JWT secret ---** SECRET_KEY defaults to
dev-secret-key-quantiva-change-in-prod (config.py:25), and compose
defaults to your-dev-secret-key-change-in-prod. If deployed unchanged,
tokens are forgeable.

**8. No login rate limit ---** POST /auth/login has no throttling; brute
force is unrestricted (auth.py:35-51).

**6.3 Security hardening gaps**

-   No security headers (CSP/HSTS/X-Frame-Options).

-   /metrics and detailed /health are unauthenticated.

-   No CSRF concern for bearer tokens, but localStorage storage is
    > XSS-readable.

-   No file magic-byte validation or virus scanning.

-   python_executor sandbox is unverified; timeout does not kill
    > threads.

-   No secret scanning in CI.

-   No dependency vulnerability scanning (pip-audit, npm audit).

-   No audit logging for document access or admin actions.

-   No per-IP rate limiting.

-   No tenant-isolation tests.

**7. Missing Features**

**Authentication / account**

-   Login and registration UI.

-   Logout and session expiration UX.

-   Refresh endpoint, token rotation, revocation.

-   Password reset, email verification, password policy.

-   Role/admin management; User.tier enforcement.

-   is_active enforcement.

**RAG / agent**

-   Supported LLM and embedding models.

-   Real token streaming from the model.

-   Full chunk text storage for BM25/summaries/eval.

-   Per-user retrieval filtering.

-   Qdrant deletion/cleanup and reindex tooling.

-   Real reranker lifecycle (pre-download, lazy load, health check).

-   Tool sandbox with hard timeout and isolation.

-   Deterministic guardrails and hallucination scoring.

-   Real confidence calibration (not random).

-   Citation-to-document viewer.

**Ingestion**

-   Server-side upload size limit and streaming.

-   Magic-byte validation and virus scanning.

-   Durable Celery queue with retry/backoff and dead-letter handling.

-   Idempotent ingestion and duplicate detection.

-   OCR provisioning (Tesseract or Vision API).

-   Excel/CSV/DOCX edge-case tests.

-   Document deletion propagation to Qdrant.

**Evaluation**

-   Real judge with no random fallback.

-   user_id ownership on eval runs.

-   Celery-based eval execution.

-   Top-k context evaluation.

-   Golden dataset management UI/API.

**Infrastructure / ops**

-   .dockerignore and secret rotation.

-   Production frontend Dockerfile.

-   Full Railway topology (Postgres, Redis, Qdrant, S3, frontend).

-   Langfuse v3 stack (ClickHouse, Redis, S3, encryption key) or
    > external Langfuse.

-   Prometheus/Grafana dashboards wired to real metrics.

-   S3 health check.

-   Non-root containers.

-   Backups, migrations runbook, rollback plan.

**Frontend**

-   Markdown rendering with sanitization.

-   Chart rendering/proxy.

-   Confidence/citations persistence across reloads.

-   Cache-hit badge.

-   Failed-upload retry/removal.

-   Loading/error states for auth and API failures.

-   Accessibility audit.

**8. Dependency & Infrastructure Issues**

**8.1 Dependency matrix**

  ----------------------------------------------------------------------------------------------------
  **Package**               **Used by**        **In                  **Local        **Impact**
                                               requirements.txt?**   install**      
  ------------------------- ------------------ --------------------- -------------- ------------------
  **boto3**                 core/storage.py,   No                    Yes            Docker/clean
                            charts                                                  install import
                                                                                    failure

  **aioboto3**              Not used (storage  Yes                   Yes            Dead dependency;
                            uses boto3)                                             misleading

  **RestrictedPython**      python_executor    No                    No             Python tool always
                                                                                    fails

  **presidio-analyzer**     pii_detector       No                    No             PII redaction
                                                                                    silently disabled

  **presidio-anonymizer**   pii_detector       No                    No             PII redaction
                                                                                    silently disabled

  **matplotlib**            chart_generator    No                    No             Chart tool always
                                                                                    fails

  **tabulate**              PyMuPDF table      No direct             Yes            PDF tables may
                            markdown                                 (transitive)   break if
                                                                                    transitive dep
                                                                                    changes

  **spaCy model             Presidio           No                    No             Presidio cannot
  (en_core_web_lg)**                                                                initialize

  **Tesseract**             PDF OCR            No (system)           Unknown        OCR silently
                                                                                    no-ops

  **langfuse**              tracing            Yes (\>=2.0.0)        5.1.0          .trace() removed
                                                                                    in v5; dead path

  **torch /                 reranker           Yes                   Yes            Large image; model
  sentence-transformers**                                                           download at
                                                                                    startup

  **pytest-asyncio**        tests              Not in runtime (CI    Yes            CI needs it
                                               installs)                            

  **aiosqlite**             tests              No                    Yes            Clean CI fails at
                                                                                    sqlite+aiosqlite

  **ruff / mypy**           CI                 Not pinned            Yes            CI tool drift

  **next / react**          frontend           package.json          Yes            Docs say Next 14;
                                                                                    actual 16

  **next-auth**             frontend           package.json          Yes            Unused
  ----------------------------------------------------------------------------------------------------

**8.2 Docker Compose issues**

-   Backend image missing boto3; startup fails on import.

-   COPY . . includes .env; no .dockerignore.

-   Backend bind-mounts ./backend:/app and runs \--reload; development
    > mode.

-   Frontend runs npm install && npm run dev; no production build.

-   Langfuse service missing ClickHouse, Redis, S3, ENCRYPTION_KEY.

-   Prometheus host port 9091, docs say 9090.

-   Qdrant healthcheck depends on /dev/tcp and bash; may fail on minimal
    > image.

-   No restart policies, resource limits, or log rotation.

-   No healthcheck for S3.

-   Backend healthcheck uses curl; installed in Dockerfile, okay.

**8.3 Railway issues**

-   railway.json builds only backend/Dockerfile.

-   No frontend service.

-   No Qdrant/Redis/S3 provisioning.

-   DATABASE_URL likely postgresql://, while app requires
    > postgresql+asyncpg://.

-   NEXT_PUBLIC_API_URL hardcoded to localhost in compose; Railway needs
    > a public URL.

-   CORS wildcard https://\*.railway.app does not match.

**8.4 CI issues**

-   Installs pytest pytest-asyncio but not aiosqlite.

-   mypy is continue-on-error.

-   No frontend lint/build job.

-   Docker build only backend.

-   No secret scan, dependency audit, or integration services
    > (Qdrant/S3).

-   No migration check (alembic upgrade head / alembic check).

-   Tests mock the critical path.

**9. Docs vs Implementation Differences**

  -------------------------------------------------------------------------------------------------------
  **Topic**                **Documentation**                   **Implementation**     **Difference**
  ------------------------ ----------------------------------- ---------------------- -------------------
  **Frontend framework**   Next.js 14 (README.md:45,           Next.js 16.3.8, React  Major version
                           2-PROJECT-OVERVIEW.md:40)           19.2.8                 mismatch
                                                               (package.json:15-18)   

  **LLM**                  Gemini 1.5 Flash (README.md:13,     Hardcoded retired      Model retired
                           2-PROJECT-OVERVIEW.md:9)            gemini-1.5-flash       2025-09-29

  **Embeddings**           text-embedding-004 768-d            Retired model + random Model retired
                           (embedding_service.py:10-11)        fallback               2026-01-14

  **Streaming**            "stream a cited answer token by     Full answer, then      Fake streaming
                           token" (README.md:26,               split(\" \") + sleep   
                           10-API-DOCUMENTATION.md:158-160)    (chat.py:227-232)      

  **DOCX**                 "Planned"                           Parser exists          Docs stale
                           (2-PROJECT-OVERVIEW.md:126)         (docx_parser.py) and   
                                                               MIME accepted          

  **OCR**                  "Planned"                           get_textpage_ocr       Partially
                           (2-PROJECT-OVERVIEW.md:127)         attempted without      attempted, not
                                                               Tesseract              provisioned

  **Python sandbox**       "RestrictedPython sandbox with 10s  RestrictedPython not   Not functional
                           timeout"                            installed; timeout     
                           (2-PROJECT-OVERVIEW.md:132,         hangs                  
                           16-TOOLS\...:119)                                          

  **Chart generator**      "Matplotlib plots saved to S3"      Matplotlib not         Not functional
                           (2-PROJECT-OVERVIEW.md:133)         installed; internal S3 
                                                               URL                    

  **PII redaction**        "Microsoft Presidio threadpool      Presidio not           Not functional
                           scanning"                           installed; silently    
                           (2-PROJECT-OVERVIEW.md:139,         returns original text  
                           17-GUARDRAILS\...:129-131)                                 

  **Langfuse**             "Every LLM call traced"             Agent bypasses         Not functional
                           (2-PROJECT-OVERVIEW.md:113,         gemini_client; v5 SDK  
                           19-LOGGING\...:30-42)               lacks .trace();        
                                                               compose invalid        

  **Prometheus/Grafana**   "Implemented"                       HTTP metrics only;     Partial
                           (2-PROJECT-OVERVIEW.md:146)         cache/LLM/stream       
                                                               metrics unused; port   
                                                               mismatch               

  **Eval API**             POST /api/v1/eval/generate-dataset, Actual /api/v1/run; no Missing/misrouted
                           POST /api/v1/eval/run               generate-dataset       
                           (10-API-DOCUMENTATION.md:233-248)                          

  **Railway**              Full-stack deployment               Backend-only           Backend only
                           (README.md:18, 20-DEPLOYMENT\...)   railway.json           

  **Semantic cache**       "Cosine similarity" (1-INDEX.md:99) Global key, no user    Privacy risk
                                                               scope, no invalidation 

  **Confidence**           "confidence: 0.94" (README.md:36)   Random 0.7--0.99 on    Fabricated
                                                               parse failure          

  **Tests**                "48 passed"                         Mocked; empty          False confidence
                                                               integration file;      
                                                               missing aiosqlite      
  -------------------------------------------------------------------------------------------------------

**10. Prioritized Repair Roadmap**

**Phase 0 --- Restore runnability (1--2 days)**

-   **Goal:** A fresh clone can build, start, migrate, and serve a
    > request without import errors.

**1.** Add missing runtime deps: boto3, RestrictedPython,
presidio-analyzer, presidio-anonymizer, matplotlib, tabulate, and the
spaCy model.

**2.** Pin dependency versions; add requirements-test.txt with pytest,
pytest-asyncio, aiosqlite, httpx.

**3.** Add backend/.dockerignore; remove .env from the build context;
rotate GOOGLE_API_KEY and SECRET_KEY.

**4.** Add Alembic migration for messages.cost_usd; run alembic upgrade
head against a clean Postgres.

**5.** Fix create_session UUID validation and remove silent session
auto-creation.

**6.** Verify docker compose config, backend import, and pytest in a
clean environment.

-   **Exit criteria:** docker compose up \--build starts backend/celery;
    > /health returns 200; alembic upgrade head succeeds; POST
    > /auth/register and /chat/message work on a clean DB.

**Phase 1 --- Security and tenant isolation (2--3 days)**

**1.** Remove mock-token bypass from backend, frontend, and MCP.

**2.** Implement frontend login/register/logout; remove localStorage
defaults; use secure token storage.

**3.** Add ownership checks to history, session creation, retrieval,
eval, and MCP.

**4.** Add user_id filter to Qdrant search; add user_id to EvalRun.

**5.** Replace SSE ?token= with short-lived one-time tickets or cookies.

**6.** Restrict CORS to explicit origins; protect /metrics and detailed
/health.

**7.** Add login/register/SSE rate limits; use User.tier; check
is_active.

**8.** Add access/refresh token types, refresh endpoint,
rotation/revocation; reject default secrets in production.

**9.** Add cross-tenant security tests.

-   **Exit criteria:** mock-token returns 401; user A cannot read user
    > B's session/document/eval; CORS preflight succeeds for the
    > deployed origin; secrets are not in the image.

**Phase 2 --- Restore LLM/RAG correctness (3--5 days)**

**1.** Replace retired models with supported, configurable models
(GEMINI_CHAT_MODEL, EMBEDDING_MODEL).

**2.** Remove random embedding and mock LLM fallbacks; return typed
errors.

**3.** Reindex all documents with the new embedding dimension; recreate
the Qdrant collection.

**4.** Wire real streaming through GeminiClient.complete_stream; extend
StreamEvent with confidence/chart/cost.

**5.** Store full chunk text; use it for BM25, summaries, and eval.

**6.** Fix Qdrant deletion on document delete; add reindex/cleanup
tooling.

**7.** Scope semantic cache by user_id and corpus version; implement
invalidation.

**8.** Remove duplicate Redis history writes.

**9.** Fix agent token/confidence accounting; remove random values.

**10.** Fix guardrails (Presidio, false positives, empty-source
hallucination, SSE error handling).

-   **Exit criteria:** A real PDF ingests with real embeddings; a real
    > question returns a grounded LLM answer with citations, confidence,
    > and real token/cost; deleting the document removes Qdrant points;
    > cross-user cache is impossible.

**Phase 3 --- Infrastructure and observability (2--3 days)**

**1.** Fix Celery event-loop/pool issue (NullPool or per-task engine).

**2.** Offload S3/embedding/reranker/judge to threads or async clients;
pre-download models at build.

**3.** Fix Langfuse compose (ClickHouse/Redis/S3/encryption key) or use
external Langfuse; pin SDK and update API.

**4.** Add production frontend Dockerfile (npm ci, build, next start).

**5.** Fix Railway topology and DATABASE_URL scheme; add
Qdrant/S3/Redis.

**6.** Align Prometheus port docs; wire real metrics (CACHE\_\*,
ACTIVE_STREAMS, DOCS_INGESTED, RETRIEVAL_LATENCY).

**7.** Add S3 health check; redact health errors; reuse Qdrant client.

-   **Exit criteria:** Compose stack healthy; Langfuse traces visible;
    > Grafana panels populated; frontend production image runs; Railway
    > backend+frontend+data services connect.

**Phase 4 --- Frontend UX (2--3 days)**

**1.** Add Markdown rendering with sanitization.

**2.** Fix chart URLs and Next/Image config (proxy or public endpoint).

**3.** Persist/display confidence, citations, chart, cache status.

**4.** Fix fallback-session routing and history timestamps.

**5.** Fix upload delete/retry behavior.

**6.** Fix lint errors; add frontend lint/build to CI.

**7.** Implement "Open document" citation action.

-   **Exit criteria:** npm run lint and npm run build pass; chat renders
    > Markdown/tables/code; charts render; reload preserves
    > citations/confidence.

**Phase 5 --- Tests and CI (2--3 days)**

**1.** Add aiosqlite and all test deps; make CI install from
requirements-test.txt.

**2.** Implement real document endpoint tests (currently 0 lines).

**3.** Add integration tests for Postgres/Redis/Qdrant/S3
(Testcontainers or compose).

**4.** Add tests for ingestion worker (two sequential docs), retrieval,
agent, guardrails, SSE, eval, MCP.

**5.** Add frontend lint/build job.

**6.** Add alembic upgrade head + drift check.

**7.** Add secret/dependency scanning.

-   **Exit criteria:** CI green on a clean runner; tests fail if the
    > critical bugs are reintroduced.

**Phase 6 --- Hardening and polish (ongoing)**

-   Non-root containers, security headers, CSP.

-   Virus scanning, magic-byte validation.

-   Backups, migration rollback, runbooks.

-   Accessibility audit.

-   Cost/latency dashboards and alerts.

-   Load testing for SSE/upload.

**11. Recommended Implementation Order (Dependency-Aware)**

**1. Freeze + baseline ---** add .dockerignore, rotate secrets, pin
deps, add missing deps.

**2. Migration fix ---** add messages.cost_usd; verify clean alembic
upgrade head.

**3. Startup fix ---** boto3, import smoke test, clean docker compose
build.

**4. Auth bypass removal ---** backend + frontend + MCP.

**5. Frontend auth flow ---** login/register/logout, secure token
storage.

**6. Tenant isolation ---** history/session/document/retrieval/eval
ownership.

**7. Model replacement ---** chat + embedding models, config, reindex.

**8. Remove silent fallbacks ---** embeddings, LLM, judge, agent
confidence.

**9. Real streaming ---** complete_stream, StreamEvent fields, frontend
consumption.

**10. Retrieval quality ---** full chunk text, BM25, Qdrant user filter,
cleanup.

**11. Cache/memory correctness ---** user-scoped keys, invalidation,
duplicate writes.

**12. Guardrails/tools ---** Presidio, RestrictedPython, Matplotlib,
timeout process.

**13. Celery/infra ---** event loop, async offload, Langfuse, frontend
Docker, Railway.

**14. Observability ---** metrics, dashboards, health checks, alerts.

**15. Tests/CI ---** integration coverage, frontend CI, migration/secret
checks.

**16. UX polish ---** Markdown, charts, citations, cache badge, lint.

Do not start with UI polish or eval dashboards. They depend on a working
backend, real models, and correct data. The fastest path to a usable
product is Phase 0 → 1 → 2.

**12. Local Testing Commands**

**12.1 Static checks**

> cd \"I:\\AI Engineer Projects\\Quantiva\"\
> \
> \# Python syntax\
> & \".venv\\Scripts\\python.exe\" -m compileall -q backend\\app\
> \
> \# Frontend typecheck\
> cd frontend\
> npx tsc \--noEmit \--incremental false\
> \
> \# Frontend lint (currently fails: 6 errors, 8 warnings)\
> npm run lint\
> \
> \# Compose syntax\
> cd ..\
> docker compose config \--quiet

**12.2 Backend tests**

> cd \"I:\\AI Engineer Projects\\Quantiva\\backend\"\
> \
> \# Offline mode avoids Hugging Face downloads during tests\
> \$env:HF_HUB_OFFLINE = \"1\"\
> \
> & \"..\\.venv\\Scripts\\python.exe\" -m pytest tests/ -x -q
> \--tb=short\
> \
> \# After adding aiosqlite + requirements-test.txt, clean install:\
> \# pip install -r requirements.txt -r requirements-test.txt

**12.3 Docker stack**

> cd \"I:\\AI Engineer Projects\\Quantiva\"\
> \
> \# Copy and edit env first; never commit it\
> Copy-Item backend\\.env.example backend\\.env\
> \
> \# Build and start\
> docker compose up \--build\
> \
> \# Check services\
> docker compose ps\
> Invoke-RestMethod http://localhost:8000/health\
> Invoke-RestMethod http://localhost:8000/metrics\
> \
> \# Apply migrations (inside backend container)\
> docker compose exec backend alembic upgrade head\
> \
> \# Verify the missing column is fixed\
> docker compose exec postgres psql -U quantiva_user -d quantiva_db -c
> \"\\d messages\"

**12.4 API smoke tests**

> \$base = \"http://localhost:8000\"\
> \
> \# Register\
> \$body = \@{ email = \"audit@example.com\"; password =
> \"ChangeMe123!\" } \| ConvertTo-Json\
> \$token = (Invoke-RestMethod -Method Post
> \"\$base/api/v1/auth/register\" \`\
> -ContentType \"application/json\" -Body \$body).access_token\
> \
> \$headers = \@{ Authorization = \"Bearer \$token\" }\
> \
> \# Me\
> Invoke-RestMethod \"\$base/api/v1/auth/me\" -Headers \$headers\
> \
> \# Upload\
> \$form = \@{ file = Get-Item \".\\sample.pdf\" }\
> \$doc = Invoke-RestMethod -Method Post
> \"\$base/api/v1/documents/upload\" \`\
> -Headers \$headers -Form \$form\
> \$doc\
> \
> \# Wait for status\
> Invoke-RestMethod
> \"\$base/api/v1/documents/\$(\$doc.id)/status?token=\$token\"\
> \
> \# Create session\
> \$session = Invoke-RestMethod -Method Post
> \"\$base/api/v1/chat/sessions\" \`\
> -Headers \$headers -ContentType \"application/json\" \`\
> -Body (@{ document_ids = @(\$doc.id) } \| ConvertTo-Json)\
> \$session\
> \
> \# Ask (non-streaming)\
> \$ask = \@{ session_id = \$session.session_id; query = \"Summarize
> this document.\" } \| ConvertTo-Json\
> Invoke-RestMethod -Method Post \"\$base/api/v1/chat/message\" \`\
> -Headers \$headers -ContentType \"application/json\" -Body \$ask\
> \
> \# Stream (inspect raw SSE)\
> curl.exe -N
> \"\$base/api/v1/chat/message/stream?session_id=\$(\$session.session_id)&query=Hello&token=\$token\"\
> \
> \# Security regression: must return 401\
> try {\
> Invoke-RestMethod \"\$base/api/v1/chat/sessions\" \`\
> -Headers \@{ Authorization = \"Bearer mock-token\" }\
> Write-Host \"FAIL: mock-token accepted\"\
> } catch {\
> Write-Host \"PASS: mock-token rejected\"\
> }

**12.5 Cross-tenant regression test (manual)**

**1.** Register user A and user B.

**2.** Upload a document as A; record doc_id.

**3.** As B, call POST /api/v1/chat/sessions with A's doc_id.

**4.** Expected: 403/404. Current: 200 (bug).

**5.** As B, call GET /api/v1/chat/sessions/{A_session_id}/history.

**6.** Expected: 404. Current: may return A's Redis history (bug).

**12.6 Frontend E2E (after auth flow exists)**

> cd frontend\
> npm ci\
> npm run build\
> npm run start\
> \# Then test in browser:\
> \# - register/login\
> \# - upload PDF\
> \# - wait for ready\
> \# - ask question\
> \# - verify Markdown, citations, confidence, chart\
> \# - reload and verify history

**13. Final Acceptance Checklist**

**Build and startup**

-   **☐** Fresh clone + backend/.env from example starts with no import
    > errors.

-   **☐** docker compose up \--build brings all required services
    > healthy.

-   **☐** Backend image does not contain .env, .venv, tests, or secrets.

-   **☐** alembic upgrade head succeeds on an empty Postgres.

-   **☐** alembic check reports no model/migration drift.

-   **☐** messages.cost_usd exists; eval_runs.user_id exists.

-   **☐** GET /health returns 200; S3 is checked; errors are redacted.

-   **☐** GET /metrics is protected or network-restricted.

**Authentication and security**

-   **☐** Authorization: Bearer mock-token returns 401 in all
    > environments.

-   **☐** Frontend has login/register/logout and no mock-token default.

-   **☐** JWT is not sent in SSE query strings.

-   **☐** Access and refresh tokens have distinct types; refresh
    > rotates/revokes.

-   **☐** SECRET_KEY default is rejected when APP_ENV=production.

-   **☐** User A cannot read User B's session, history, document,
    > chunks, or eval.

-   **☐** Qdrant search filters by user_id.

-   **☐** CORS allows only explicit production origins.

-   **☐** Login/register/SSE/upload are rate-limited.

-   **☐** No secrets are committed or baked into images.

**Ingestion**

-   **☐** PDF, XLSX, CSV, and DOCX uploads parse correctly.

-   **☐** Upload enforces a server-side size limit and validates magic
    > bytes.

-   **☐** Ingestion uses a supported embedding model and real vectors.

-   **☐** A failed embedding marks the document failed, not ready.

-   **☐** Two sequential Celery tasks both succeed (event-loop
    > regression).

-   **☐** Qdrant contains points for the document; chunk_count matches.

-   **☐** Deleting a document removes S3 object, Postgres rows, and
    > Qdrant points.

-   **☐** Re-uploading the same file does not create duplicate vectors.

**Retrieval and agent**

-   **☐** Hybrid retrieval returns relevant chunks for a keyword after
    > character 200.

-   **☐** Reranker model is pre-downloaded or lazy-loaded without
    > blocking startup.

-   **☐** Agent uses a supported LLM model and no mock fallback.

-   **☐** Agent returns real token usage and confidence; no random
    > values.

-   **☐** Guardrails redact PII (Presidio installed) and do not block
    > benign text.

-   **☐** Empty-source hallucination score is 0/flagged, not 1.0.

-   **☐** Python tool executes safely and times out with a hard kill.

-   **☐** Chart tool generates a browser-reachable URL and the frontend
    > renders it.

**Streaming and frontend**

-   **☐** SSE emits real model deltas as they arrive.

-   **☐** Final SSE event includes citations, confidence, chart URL,
    > tokens, and cost.

-   **☐** Frontend renders Markdown, tables, code blocks, and citations.

-   **☐** Reload preserves citations, confidence, and chart.

-   **☐** Fallback/invalid session IDs return a clear error, not a new
    > session.

-   **☐** npm run lint passes with 0 errors.

-   **☐** npm run build passes.

-   **☐** Failed uploads can be removed/retried; temp IDs do not hit the
    > API.

**Eval and observability**

-   **☐** Eval uses a real judge, no random/mock fallback.

-   **☐** Eval is scoped to the owner and runs in a worker, not the API
    > process.

-   **☐** Langfuse traces are visible for real chat calls.

-   **☐** Grafana panels show HTTP, LLM, cache, stream, ingestion, and
    > retrieval metrics.

-   **☐** Prometheus port matches documentation.

**CI/CD**

-   **☐** CI installs requirements-test.txt including aiosqlite.

-   **☐** CI runs backend lint, typecheck, tests, and migration check.

-   **☐** CI runs frontend npm ci, lint, and build.

-   **☐** CI builds the frontend Docker image.

-   **☐** CI runs secret and dependency scans.

-   **☐** E2E/integration tests cover real ingestion, retrieval, agent,
    > SSE, eval, and MCP.

**14. Closing Assessment**

Quantiva has a strong skeleton: a clean FastAPI layout, a coherent
service split, a polished Next.js UI, and a sensible RAG design on
paper. The failure is not architectural taste; it is execution
integrity. The project repeatedly chooses silent degradation over
explicit failure: retired models become mock answers, failed embeddings
become random vectors, missing packages become no-op guardrails, and
parse failures become random confidence scores. That pattern makes the
system appear to work while producing untrustworthy results.

The shortest path to a genuinely usable product is:

**1.** Make failures visible (deps, models, migrations, auth).

**2.** Make tenancy real (ownership checks everywhere).

**3.** Make the RAG path real (supported models, full text, real
streaming).

**4.** Then polish the UI and observability.

Until Phase 0 and Phase 1 are complete, no demo, benchmark, or
production claim should be trusted.
