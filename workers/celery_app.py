from __future__ import annotations
from celery import Celery
from config import settings

celery_app = Celery(
    "distributed_asl",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=["workers.alphabet_tasks", "workers.phrase_tasks"],
)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=settings.task_status_expiration_seconds,
    task_routes={
        "workers.alphabet_tasks.predict_alphabet": {"queue": settings.alphabet_queue},
        "workers.phrase_tasks.predict_phrase": {"queue": settings.phrase_queue},
        "workers.phrase_tasks.normalize_phrase": {"queue": settings.phrase_queue},
    },
    worker_prefetch_multiplier=1,
    task_track_started=True,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_acks_on_failure_or_timeout=True,
    worker_cancel_long_running_tasks_on_connection_loss=True,
    task_soft_time_limit=settings.task_timeout_seconds,
    task_time_limit=settings.phrase_prediction_timeout_seconds,
)

# Import after app construction so Celery lifecycle signal handlers are registered once.
from workers import worker_lifecycle as _worker_lifecycle  # noqa: E402,F401
