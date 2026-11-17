"""Persistent asyncio job manager for local-first Quantiva execution."""
from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Awaitable, Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from functools import partial
from typing import Any, TypeVar

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session, sessionmaker

from app.storage.models import (
    Document,
    DocumentChunk,
    IndexMetadata,
    ProcessingJob,
)

logger = logging.getLogger(__name__)
_T = TypeVar("_T")
CleanupCallback = Callable[[], Any | Awaitable[Any]]
JobHandler = Callable[["JobContext", Mapping[str, Any]], Awaitable[None]]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class JobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    INTERRUPTED = "INTERRUPTED"


class JobStage(str, Enum):
    VALIDATING = "VALIDATING"
    EXTRACTING = "EXTRACTING"
    OCR = "OCR"
    CHUNKING = "CHUNKING"
    EMBEDDING = "EMBEDDING"
    INDEXING_VECTOR = "INDEXING_VECTOR"
    INDEXING_SPARSE = "INDEXING_SPARSE"
    FINALIZING = "FINALIZING"


TERMINAL_STATUSES = frozenset(
    {
        JobStatus.COMPLETED.value,
        JobStatus.FAILED.value,
        JobStatus.CANCELLED.value,
        JobStatus.INTERRUPTED.value,
    }
)


class JobCancelledError(asyncio.CancelledError):
    """Cooperative stop raised at a stage checkpoint."""

    def __init__(self, reason: str = "user") -> None:
        super().__init__(f"Job cancelled ({reason})")
        self.reason = reason


class CancellationToken:
    def __init__(self) -> None:
        self._event = asyncio.Event()
        self._reason = "user"

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    @property
    def reason(self) -> str:
        return self._reason

    def cancel(self, reason: str = "user") -> None:
        self._reason = reason
        self._event.set()

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled:
            raise JobCancelledError(self.reason)

    async def checkpoint(self) -> None:
        self.raise_if_cancelled()
        await asyncio.sleep(0)
        self.raise_if_cancelled()


@dataclass(frozen=True, slots=True)
class JobProgressEvent:
    job_id: str
    status: str
    stage: str | None
    progress_percent: float
    message: str | None = None


class JobContext:
    """Capabilities exposed to one running job handler."""

    def __init__(
        self,
        manager: "LocalJobManager",
        job_id: str,
        token: CancellationToken,
    ) -> None:
        self.manager = manager
        self.job_id = job_id
        self.token = token
        self._cleanups: list[CleanupCallback] = []
        self._report_tasks: set[asyncio.Task[Any]] = set()
        self._cleaning = False

    async def checkpoint(self) -> None:
        await self.token.checkpoint()

    def raise_if_cancelled(self) -> None:
        self.token.raise_if_cancelled()

    async def report(
        self,
        stage: JobStage | str,
        progress_percent: float,
        message: str | None = None,
    ) -> None:
        self.raise_if_cancelled()
        await self.manager.update_progress(
            self.job_id,
            stage,
            progress_percent,
            message,
        )
        self.raise_if_cancelled()

    def report_nowait(
        self,
        stage: JobStage | str,
        progress_percent: float,
        message: str | None = None,
    ) -> None:
        """Schedule progress persistence from a synchronous pipeline callback."""
        self.raise_if_cancelled()
        task = asyncio.get_running_loop().create_task(
            self.report(stage, progress_percent, message)
        )
        self._report_tasks.add(task)

    async def flush_reports(self) -> None:
        if self._report_tasks:
            tasks = tuple(self._report_tasks)
            try:
                await asyncio.gather(*tasks)
            finally:
                self._report_tasks.difference_update(tasks)

    async def run_cpu(self, function: Callable[..., _T], *args, **kwargs) -> _T:
        if not self._cleaning:
            self.raise_if_cancelled()
        loop = asyncio.get_running_loop()
        result = await loop.run_in_executor(
            self.manager.executor,
            partial(function, *args, **kwargs),
        )
        if not self._cleaning:
            self.raise_if_cancelled()
        return result

    def add_cleanup(self, callback: CleanupCallback) -> CleanupCallback:
        self._cleanups.append(callback)
        return callback

    def remove_cleanup(self, callback: CleanupCallback) -> None:
        if callback in self._cleanups:
            self._cleanups.remove(callback)

    async def cleanup(self) -> None:
        self._cleaning = True
        try:
            for callback in reversed(self._cleanups):
                try:
                    result = callback()
                    if inspect.isawaitable(result):
                        await result
                except BaseException:
                    logger.exception("Job cleanup failed for %s", self.job_id)
            self._cleanups.clear()
        finally:
            self._cleaning = False

    async def update_payload(self, **values: Any) -> None:
        await self.manager.update_payload(self.job_id, values)


class LocalJobManager:
    """SQLite-persistent queue with cooperative cancellation and recovery."""

    def __init__(
        self,
        session_factory: sessionmaker[Session],
        *,
        worker_count: int = 2,
        cpu_workers: int | None = None,
        vector_store: Any | None = None,
        sparse_store: Any | None = None,
    ) -> None:
        if worker_count <= 0:
            raise ValueError("worker_count must be positive")
        self.session_factory = session_factory
        self.worker_count = int(worker_count)
        self.cpu_workers = int(cpu_workers or max(2, worker_count))
        self.vector_store = vector_store
        self.sparse_store = sparse_store
        self.executor = ThreadPoolExecutor(
            max_workers=self.cpu_workers,
            thread_name_prefix="quantiva-job-cpu",
        )
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._workers: list[asyncio.Task[None]] = []
        self._handlers: dict[str, JobHandler] = {}
        self._tokens: dict[str, CancellationToken] = {}
        self._completion_events: dict[str, asyncio.Event] = {}
        self._history: dict[str, list[JobProgressEvent]] = {}
        self._persistence_lock = asyncio.Lock()
        self._started = False
        self._stopping = False
        self._idle = asyncio.Event()
        self._idle.set()

    @property
    def is_started(self) -> bool:
        return self._started

    @property
    def active_job_ids(self) -> tuple[str, ...]:
        return tuple(self._tokens)

    def register_handler(self, job_type: str, handler: JobHandler) -> None:
        normalized = str(job_type).strip().upper()
        if not normalized:
            raise ValueError("job_type is required")
        self._handlers[normalized] = handler

    async def _database(self, operation: Callable[[Session], _T]) -> _T:
        def run() -> _T:
            with self.session_factory() as session:
                return operation(session)

        async with self._persistence_lock:
            return await asyncio.to_thread(run)

    def _record_history(
        self,
        job_id: str,
        status: str,
        stage: str | None,
        progress: float,
        message: str | None,
    ) -> None:
        self._history.setdefault(job_id, []).append(
            JobProgressEvent(job_id, status, stage, progress, message)
        )

    def history(self, job_id: str) -> tuple[JobProgressEvent, ...]:
        return tuple(self._history.get(job_id, ()))

    async def start(self) -> None:
        if self._started:
            return
        if getattr(self.executor, "_shutdown", False):
            self.executor = ThreadPoolExecutor(
                max_workers=self.cpu_workers,
                thread_name_prefix="quantiva-job-cpu",
            )
        self._queue = asyncio.Queue()
        self._stopping = False
        await self.recover_interrupted_jobs()
        pending_ids = await self._pending_job_ids()
        for job_id in pending_ids:
            await self._queue.put(job_id)
        self._workers = [
            asyncio.create_task(self._worker(index), name=f"quantiva-job-{index}")
            for index in range(self.worker_count)
        ]
        self._started = True

    async def stop(self, timeout_seconds: float = 5.0) -> None:
        if not self._started:
            self.executor.shutdown(wait=False, cancel_futures=True)
            return
        self._stopping = True
        deadline = asyncio.get_running_loop().time() + timeout_seconds
        for token in tuple(self._tokens.values()):
            token.cancel("shutdown")
        try:
            await asyncio.wait_for(self._idle.wait(), timeout=timeout_seconds)
        except asyncio.TimeoutError:
            logger.warning("Timed out waiting for local jobs to reach checkpoints")
        for worker in self._workers:
            worker.cancel()
        remaining = max(0.05, deadline - asyncio.get_running_loop().time())
        try:
            await asyncio.wait_for(
                asyncio.gather(*self._workers, return_exceptions=True),
                timeout=remaining,
            )
        except asyncio.TimeoutError:
            logger.warning("Local worker tasks did not exit within shutdown deadline")
        self._workers.clear()
        self._started = False
        self.executor.shutdown(wait=False, cancel_futures=True)

    async def submit(
        self,
        job_type: str,
        *,
        payload: Mapping[str, Any] | None = None,
        knowledge_base_id: str | None = None,
        document_id: str | None = None,
        stage: JobStage | str = JobStage.VALIDATING,
    ) -> str:
        normalized_type = str(job_type).strip().upper()
        stage_value = stage.value if isinstance(stage, JobStage) else str(stage)

        def create(session: Session) -> str:
            job = ProcessingJob(
                job_type=normalized_type,
                knowledge_base_id=knowledge_base_id,
                document_id=document_id,
                status=JobStatus.PENDING.value,
                stage=stage_value,
                progress_percent=0.0,
                message="Queued",
                payload_json=dict(payload or {}),
            )
            session.add(job)
            session.commit()
            return job.id

        job_id = await self._database(create)
        self._record_history(
            job_id,
            JobStatus.PENDING.value,
            stage_value,
            0.0,
            "Queued",
        )
        self._completion_events[job_id] = asyncio.Event()
        if self._started and not self._stopping:
            await self._queue.put(job_id)
        return job_id

    async def update_progress(
        self,
        job_id: str,
        stage: JobStage | str,
        progress_percent: float,
        message: str | None = None,
    ) -> None:
        stage_value = stage.value if isinstance(stage, JobStage) else str(stage)
        progress = min(100.0, max(0.0, float(progress_percent)))

        def persist(session: Session) -> tuple[str, str | None, float]:
            job = session.get(ProcessingJob, job_id)
            if job is None:
                raise KeyError(f"Unknown job: {job_id}")
            job.stage = stage_value
            job.progress_percent = progress
            if message is not None:
                job.message = message
            session.commit()
            return job.status, job.stage, job.progress_percent

        status, saved_stage, saved_progress = await self._database(persist)
        self._record_history(job_id, status, saved_stage, saved_progress, message)

    async def update_payload(
        self,
        job_id: str,
        values: Mapping[str, Any],
    ) -> None:
        def persist(session: Session) -> None:
            job = session.get(ProcessingJob, job_id)
            if job is None:
                raise KeyError(f"Unknown job: {job_id}")
            job.payload_json = {**dict(job.payload_json or {}), **dict(values)}
            session.commit()

        await self._database(persist)

    async def get_job(self, job_id: str) -> ProcessingJob | None:
        def load(session: Session) -> ProcessingJob | None:
            job = session.get(ProcessingJob, job_id)
            if job is not None:
                session.expunge(job)
            return job

        return await self._database(load)

    async def wait_for_job(
        self,
        job_id: str,
        *,
        timeout: float = 30.0,
    ) -> ProcessingJob:
        event = self._completion_events.setdefault(job_id, asyncio.Event())
        existing = await self.get_job(job_id)
        if existing is None:
            raise KeyError(f"Unknown job: {job_id}")
        if existing.status not in TERMINAL_STATUSES:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        result = await self.get_job(job_id)
        if result is None:  # pragma: no cover - FK deletion race guard
            raise KeyError(f"Job was deleted: {job_id}")
        return result

    async def cancel(self, job_id: str) -> bool:
        token = self._tokens.get(job_id)
        if token is not None:
            token.cancel("user")
            return True

        def cancel_pending(session: Session) -> bool:
            job = session.get(ProcessingJob, job_id)
            if job is None or job.status != JobStatus.PENDING.value:
                return False
            job.status = JobStatus.CANCELLED.value
            job.message = "Cancelled before execution"
            job.completed_at = utc_now()
            session.commit()
            return True

        cancelled = await self._database(cancel_pending)
        if cancelled:
            self._record_history(
                job_id,
                JobStatus.CANCELLED.value,
                None,
                0.0,
                "Cancelled before execution",
            )
            self._completion_events.setdefault(job_id, asyncio.Event()).set()
        return cancelled

    async def retry_job(self, job_id: str) -> str:
        previous = await self.get_job(job_id)
        if previous is None:
            raise KeyError(f"Unknown job: {job_id}")
        if previous.status not in TERMINAL_STATUSES:
            raise ValueError("Only terminal jobs can be retried")
        if previous.document_id:
            await self._set_document_status(previous.document_id, "PENDING", None)
        return await self.submit(
            previous.job_type,
            payload=previous.payload_json,
            knowledge_base_id=previous.knowledge_base_id,
            document_id=previous.document_id,
            stage=JobStage.VALIDATING,
        )

    resume_job = retry_job

    async def startup(self) -> None:
        await self.start()

    async def shutdown(self, timeout_seconds: float = 5.0) -> None:
        await self.stop(timeout_seconds)

    async def cancel_job(self, job_id: str) -> bool:
        return await self.cancel(job_id)

    async def _pending_job_ids(self) -> list[str]:
        def load(session: Session) -> list[str]:
            return list(
                session.scalars(
                    select(ProcessingJob.id)
                    .where(ProcessingJob.status == JobStatus.PENDING.value)
                    .order_by(ProcessingJob.created_at)
                )
            )

        return await self._database(load)

    async def _worker(self, worker_index: int) -> None:
        while True:
            try:
                job_id = await self._queue.get()
            except asyncio.CancelledError:
                return
            try:
                if self._stopping:
                    return
                await self._execute_job(job_id)
            finally:
                self._queue.task_done()

    async def _execute_job(self, job_id: str) -> None:
        snapshot = await self.get_job(job_id)
        if snapshot is None or snapshot.status != JobStatus.PENDING.value:
            return
        handler = self._handlers.get(snapshot.job_type.upper())
        if handler is None:
            await self._finish_job(
                job_id,
                JobStatus.FAILED,
                message=f"No handler registered for {snapshot.job_type}",
                error_details="Missing local job handler",
            )
            return

        token = CancellationToken()
        context = JobContext(self, job_id, token)
        self._tokens[job_id] = token
        self._idle.clear()
        await self._mark_running(job_id)
        try:
            await handler(context, dict(snapshot.payload_json or {}))
            await context.flush_reports()
            await context.checkpoint()
        except JobCancelledError as exc:
            await context.cleanup()
            status = (
                JobStatus.INTERRUPTED
                if exc.reason == "shutdown"
                else JobStatus.CANCELLED
            )
            await self._finish_job(
                job_id,
                status,
                message=(
                    "Interrupted during application shutdown"
                    if status is JobStatus.INTERRUPTED
                    else "Cancelled"
                ),
            )
            if snapshot.document_id:
                document_status = (
                    "INTERRUPTED_NEEDS_RETRY"
                    if status is JobStatus.INTERRUPTED
                    else "CANCELLED"
                )
                await self._set_document_status(snapshot.document_id, document_status, None)
        except asyncio.CancelledError:
            token.cancel("shutdown")
            await context.cleanup()
            await self._finish_job(
                job_id,
                JobStatus.INTERRUPTED,
                message="Interrupted during application shutdown",
            )
            if snapshot.document_id:
                await self._set_document_status(
                    snapshot.document_id,
                    "INTERRUPTED_NEEDS_RETRY",
                    None,
                )
            raise
        except Exception as exc:
            await context.cleanup()
            await self._finish_job(
                job_id,
                JobStatus.FAILED,
                message="Job failed",
                error_details=f"{type(exc).__name__}: {str(exc)[:2000]}",
            )
            if snapshot.document_id:
                await self._set_document_status(
                    snapshot.document_id,
                    "FAILED",
                    str(exc)[:2000],
                )
        else:
            await self._finish_job(
                job_id,
                JobStatus.COMPLETED,
                stage=JobStage.FINALIZING,
                progress=100.0,
                message="Completed",
            )
        finally:
            self._tokens.pop(job_id, None)
            if not self._tokens:
                self._idle.set()

    async def _mark_running(self, job_id: str) -> None:
        def persist(session: Session) -> tuple[str | None, float]:
            job = session.get(ProcessingJob, job_id)
            if job is None:
                raise KeyError(job_id)
            job.status = JobStatus.RUNNING.value
            job.started_at = utc_now()
            job.message = "Running"
            session.commit()
            return job.stage, job.progress_percent

        stage, progress = await self._database(persist)
        self._record_history(
            job_id,
            JobStatus.RUNNING.value,
            stage,
            progress,
            "Running",
        )

    async def _finish_job(
        self,
        job_id: str,
        status: JobStatus,
        *,
        stage: JobStage | str | None = None,
        progress: float | None = None,
        message: str | None = None,
        error_details: str | None = None,
    ) -> None:
        stage_value = (
            stage.value if isinstance(stage, JobStage) else str(stage) if stage else None
        )

        def persist(session: Session) -> tuple[str | None, float]:
            job = session.get(ProcessingJob, job_id)
            if job is None:
                raise KeyError(job_id)
            job.status = status.value
            if stage_value is not None:
                job.stage = stage_value
            if progress is not None:
                job.progress_percent = min(100.0, max(0.0, progress))
            if message is not None:
                job.message = message
            job.error_details = error_details
            job.completed_at = utc_now()
            session.commit()
            return job.stage, job.progress_percent

        saved_stage, saved_progress = await self._database(persist)
        self._record_history(
            job_id,
            status.value,
            saved_stage,
            saved_progress,
            message,
        )
        self._completion_events.setdefault(job_id, asyncio.Event()).set()

    async def _set_document_status(
        self,
        document_id: str,
        status: str,
        error_message: str | None,
    ) -> None:
        def persist(session: Session) -> None:
            document = session.get(Document, document_id)
            if document is not None:
                document.status = status
                document.error_message = error_message
                session.commit()

        await self._database(persist)

    async def recover_interrupted_jobs(self) -> list[str]:
        """Mark crash-stuck rows and purge partial document/index artifacts."""

        def recover(session: Session):
            running_jobs = list(
                session.scalars(
                    select(ProcessingJob).where(
                        func.upper(ProcessingJob.status) == JobStatus.RUNNING.value
                    )
                )
            )
            processing_documents = list(
                session.scalars(
                    select(Document).where(func.upper(Document.status) == "PROCESSING")
                )
            )
            documents: dict[str, Document] = {
                document.id: document for document in processing_documents
            }
            for job in running_jobs:
                job.status = JobStatus.INTERRUPTED.value
                job.message = "Interrupted by an unexpected application shutdown"
                job.completed_at = utc_now()
                if job.document_id:
                    document = session.get(Document, job.document_id)
                    if document is not None:
                        documents[document.id] = document

            cleanup_records: list[tuple[str, str, str]] = []
            for document in documents.values():
                document.status = "INTERRUPTED_NEEDS_RETRY"
                document.error_message = "Previous processing was interrupted"
                index = session.scalar(
                    select(IndexMetadata).where(
                        IndexMetadata.knowledge_base_id == document.knowledge_base_id
                    )
                )
                collection = (
                    index.vector_collection_name
                    if index is not None
                    else f"kb_{document.knowledge_base_id}"
                )
                cleanup_records.append(
                    (document.id, document.knowledge_base_id, collection)
                )
            if documents:
                session.execute(
                    delete(DocumentChunk).where(
                        DocumentChunk.document_id.in_(tuple(documents))
                    )
                )

            shadow_collections = [
                str(job.payload_json.get("shadow_collection"))
                for job in running_jobs
                if (
                    job.payload_json
                    and job.payload_json.get("shadow_collection")
                    and not job.payload_json.get("shadow_promoted")
                )
            ]
            session.commit()
            return running_jobs, cleanup_records, shadow_collections

        running_jobs, cleanup_records, shadow_collections = await self._database(recover)
        for document_id, knowledge_base_id, collection in cleanup_records:
            if self.vector_store is not None:
                try:
                    await self.vector_store.delete_by_document_id(
                        collection,
                        document_id,
                        filters={"knowledge_base_id": knowledge_base_id},
                    )
                except Exception:
                    logger.exception("Failed to purge interrupted document vectors")
            if self.sparse_store is not None:
                try:
                    await self.sparse_store.delete_by_document_id(
                        knowledge_base_id,
                        document_id,
                    )
                except Exception:
                    logger.exception("Failed to purge interrupted sparse rows")
        if self.vector_store is not None:
            for collection in shadow_collections:
                try:
                    await self.vector_store.delete_collection(collection)
                except Exception:
                    logger.exception("Failed to purge interrupted shadow collection")

        interrupted_ids = [job.id for job in running_jobs]
        for job in running_jobs:
            self._record_history(
                job.id,
                JobStatus.INTERRUPTED.value,
                job.stage,
                job.progress_percent,
                job.message,
            )
            self._completion_events.setdefault(job.id, asyncio.Event()).set()
        return interrupted_ids


_default_manager: LocalJobManager | None = None


def configure_default_job_manager(manager: LocalJobManager) -> LocalJobManager:
    global _default_manager
    _default_manager = manager
    return manager


def get_local_job_manager() -> LocalJobManager:
    if _default_manager is None:
        raise RuntimeError("LocalJobManager has not been configured during startup")
    return _default_manager


__all__ = [
    "CancellationToken",
    "JobCancelledError",
    "JobContext",
    "JobProgressEvent",
    "JobStage",
    "JobStatus",
    "LocalJobManager",
    "TERMINAL_STATUSES",
    "configure_default_job_manager",
    "get_local_job_manager",
]
