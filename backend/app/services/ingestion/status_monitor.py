"""Watchdog for ingestion rows that no live worker can finish.

A document leaves ``processing`` only when its Celery task writes a terminal
status. If the worker is stopped, crashes, or a task never reaches a worker,
the row is never updated and clients poll forever. The checks below close such
rows without racing genuine work: the worker holds ``FOR UPDATE`` on its
document for the whole ingestion, so ``SKIP LOCKED`` skips live jobs and only
returns orphaned rows.
"""
import asyncio
import logging
import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.document import Document

logger = logging.getLogger(__name__)

WORKER_UNAVAILABLE_REASON = (
    "Ingestion worker unavailable; indexing did not start. Retry indexing."
)
TIMEOUT_REASON = "Indexing timed out before completion. Retry indexing."


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def processing_age_seconds(doc: Document) -> float:
    """Seconds since the job entered processing (falls back to creation time)."""
    started = doc.processing_started_at or doc.created_at
    if started is None:
        return 0.0
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    return max(0.0, (_utcnow() - started).total_seconds())


async def mark_failed(db: AsyncSession, doc: Document, reason: str) -> None:
    """Persist a terminal failed state with a client-safe explanation."""
    doc.status = "failed"
    doc.failure_reason = reason
    doc.processing_started_at = None
    await db.commit()


async def fail_stale_documents(
    db: AsyncSession,
    *,
    user_id: uuid.UUID | None = None,
    timeout_seconds: int | None = None,
) -> int:
    """Mark processing rows older than the timeout as failed.

    Rows locked by an active worker are skipped, so ingestion that is still
    running is never interrupted.
    """
    timeout = settings.INGESTION_STALE_SECONDS if timeout_seconds is None else timeout_seconds
    started = func.coalesce(Document.processing_started_at, Document.created_at)
    statement = (
        select(Document.id)
        .where(Document.status == "processing", started <= _utcnow() - timedelta(seconds=timeout))
        .with_for_update(skip_locked=True)
    )
    if user_id is not None:
        statement = statement.where(Document.user_id == user_id)

    stale_ids = list((await db.scalars(statement)).all())
    if not stale_ids:
        return 0

    await db.execute(
        update(Document)
        .where(Document.id.in_(stale_ids), Document.status == "processing")
        .values(status="failed", failure_reason=TIMEOUT_REASON, processing_started_at=None)
    )
    await db.commit()
    logger.warning("Marked %d orphaned ingestion job(s) as failed", len(stale_ids))
    return len(stale_ids)


async def worker_is_available() -> bool:
    """Return True when at least one Celery worker answers a broker ping."""
    # Imported lazily so the API does not load worker task modules at startup.
    from app.workers.celery_app import celery_app

    try:
        replies = await asyncio.wait_for(
            asyncio.to_thread(
                celery_app.control.ping,
                timeout=settings.WORKER_PING_TIMEOUT_SECONDS,
            ),
            timeout=settings.WORKER_PING_TIMEOUT_SECONDS + 1,
        )
    except Exception:
        logger.warning("Celery worker liveness probe failed", exc_info=True)
        return False
    return bool(replies)
