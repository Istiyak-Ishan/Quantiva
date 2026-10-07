import os
from celery import Celery
from app.config import settings

# Initialize Celery
# We use Redis as both the message broker and the result backend.
celery_app = Celery(
    "quantiva_worker",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["app.workers.ingestion_worker"]
)

# Optional configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
)
