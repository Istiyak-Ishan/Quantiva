"""Orphaned ingestion rows must reach a terminal state without racing live work."""
from datetime import datetime, timedelta, timezone

from app.models.document import Document
from app.models.user import User
from app.services.ingestion.status_monitor import (
    TIMEOUT_REASON,
    fail_stale_documents,
    processing_age_seconds,
)


def _document(user_id, name: str, started_at, status: str = "processing") -> Document:
    return Document(
        user_id=user_id,
        filename=name,
        file_type="application/pdf",
        file_size=10,
        storage_path=name,
        status=status,
        processing_started_at=started_at,
    )


async def test_stale_processing_is_failed_but_fresh_processing_survives(db_session):
    user = User(email="watchdog@example.com", hashed_password="not-used")
    db_session.add(user)
    await db_session.flush()

    old = _document(user.id, "old.pdf", datetime.now(timezone.utc) - timedelta(hours=1))
    fresh = _document(user.id, "fresh.pdf", datetime.now(timezone.utc))
    ready = _document(user.id, "ready.pdf", None, status="ready")
    db_session.add_all([old, fresh, ready])
    await db_session.commit()

    changed = await fail_stale_documents(db_session, user_id=user.id)

    await db_session.refresh(old)
    await db_session.refresh(fresh)
    await db_session.refresh(ready)
    assert changed == 1
    assert old.status == "failed"
    assert old.failure_reason == TIMEOUT_REASON
    assert old.processing_started_at is None
    assert fresh.status == "processing"
    assert ready.status == "ready"


async def test_stale_sweep_is_scoped_to_the_requesting_user(db_session):
    alice = User(email="alice-watchdog@example.com", hashed_password="not-used")
    bob = User(email="bob-watchdog@example.com", hashed_password="not-used")
    db_session.add_all([alice, bob])
    await db_session.flush()

    stale_at = datetime.now(timezone.utc) - timedelta(hours=1)
    alice_doc = _document(alice.id, "alice.pdf", stale_at)
    bob_doc = _document(bob.id, "bob.pdf", stale_at)
    db_session.add_all([alice_doc, bob_doc])
    await db_session.commit()

    assert await fail_stale_documents(db_session, user_id=alice.id) == 1

    await db_session.refresh(alice_doc)
    await db_session.refresh(bob_doc)
    assert alice_doc.status == "failed"
    assert bob_doc.status == "processing"


def test_processing_age_falls_back_to_creation_time():
    doc = _document(None, "legacy.pdf", None)
    doc.created_at = datetime.now(timezone.utc) - timedelta(seconds=42)
    assert 40 <= processing_age_seconds(doc) < 60
