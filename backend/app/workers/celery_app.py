from celery import Celery
from app.config import settings
import app.models  # noqa: F401 - registers all database models in Base.metadata for workers

# Initialize Celery
# We use Redis as both the message broker and the result backend.
celery_app = Celery(
    "quantiva_worker",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["app.workers.ingestion_worker", "app.workers.eval_worker"]
)

# Optional configuration
celery_app.conf.update(
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    broker_connection_timeout=3,
    task_publish_retry_policy={"max_retries": 2, "interval_start": 0, "interval_step": .2, "interval_max": .5},
)
