# Operations and outstanding acceptance

## Existing installations

1. Take and verify a database backup, and retain current image/config versions.
2. Review Alembic history. The new revisions add message cost, revocable login
   sessions, evaluation ownership (backfilled from documents), complete chunk text,
   and persisted answer metadata. Upgrade scripts do not drop existing tables.
3. Run `alembic upgrade head` during a controlled deployment, then `alembic check`.
   Do not use `stamp head` to hide drift or drop/recreate populated tables.
4. Legacy auth tokens intentionally no longer authenticate. Users must log in.
5. Legacy document indexes require explicit owner-initiated reindexing. New vector
   names include the configured prefix, embedding model, and dimensions. Keep old
   collections until backup/reindex verification and explicit cleanup approval.
6. Check `/ready`, queue connectivity, worker logs, a real upload/question, history
   reload, deletion propagation, and a second-user isolation test before release.

Never run destructive rollback scripts, volume removal, bulk vector cleanup or
secret rotation without operator approval. If images containing `.env` were
previously distributed, rotation of the potentially exposed Google/JWT keys is an
operator action; Docker ignore rules cannot revoke already leaked credentials.

## Deployment

- Compose is a loopback-only local stack. Existing named volumes, including the
  no-longer-used Langfuse volume declaration, remain intact.
- The frontend Docker build uses `npm ci`, Next standalone output and a non-root
  runtime. `BACKEND_URL` is a build argument. Keep browser `/api/v1` same-origin.
- The backend runs as a non-root user, with a readiness healthcheck and a
  predownloaded reranker. Actual Docker builds are still unverified in this repair
  environment because Docker Desktop's engine is not running.
- Use HTTPS in production, `APP_ENV=production`, explicit HTTPS `CORS_ORIGINS`,
  strong independent secrets, private data-store networking, encrypted backups,
  least-privileged storage credentials, resource limits and log rotation.
- Settings normalize `postgres://` and `postgresql://` to the asyncpg scheme.
- Railway: configure a backend service root of `backend`, Dockerfile `Dockerfile`,
  pre-deploy command `alembic upgrade head`, and readiness path `/ready`. Create a
  separate worker using the same image and `celery -A app.workers.celery_app worker`.
  Configure a frontend rooted at `frontend`, build-time `BACKEND_URL`, and external
  PostgreSQL, Redis, Qdrant and S3. Root `railway.json` alone does not provision them.
  This topology has not been deployed/tested with a Railway account.
- CI runs backend lint/typecheck/tests/manual migration drift check, frontend
  lint/build/browser contracts, production dependency audit, secret history scan and
  Python dependency audit. CI itself has not yet run on the hosted runner.

## Optional capabilities

- Langfuse: use an existing complete external instance and configure all of
  `LANGFUSE_HOST`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`. The adapter uses
  current SDK generation observations; traces intentionally omit private prompts
  and document content. Delivery to a real Langfuse server has not been verified.
- PII: pattern mode is always available but does not detect names. Provision the
  spaCy model before enabling Presidio mode. No unattended model downloads at import.
- OCR: the backend image installs Tesseract and English language data. Scanned-PDF
  acceptance on a real image remains outstanding; no empty parse is reported ready.
- Malware scanning: set `CLAMAV_HOST` (and optional `CLAMAV_PORT`) to use a ClamAV
  INSTREAM scanner. Uploads fail closed if the scanner is configured but unreachable.
  Leaving it unset means structural validation only, not antivirus scanning.
- External hardened execution sandbox, password reset/email verification and an
  audit-log retention policy remain follow-up hardening.
- `S3_PUBLIC_ENDPOINT_URL` must be reachable by the browser. Chat messages persist an
  owner-scoped API chart path; loading history re-signs the current object, so chart
  URLs no longer expire in stored conversations.

## Acceptance still required on real infrastructure

- Clean PostgreSQL upgrade and drift check; upgrade with existing representative rows.
- Docker build/context inspection, non-root container startup and all service health.
- Live S3/Redis/Qdrant/Celery ingestion of PDF/CSV/XLSX/DOCX; two sequential tasks.
- Full real-browser register/upload/wait/question/stream/reload/delete workflow.
- Cancellation/load tests, rate-limit concurrency/outage checks against actual Redis.
- Real evaluation worker execution and external trace/metrics delivery.
- Restore/rollback drill, TLS/proxy settings, production sandbox limits and scanning.

Regression tests and bounded provider/local-RAG smoke tests support these checks;
they do not replace them. See `REPAIR_PROGRESS.md` for actual executed results.
