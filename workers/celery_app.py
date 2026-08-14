from __future__ import annotations
import os
from celery import Celery

REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/0")
celery_app = Celery(
    "distributed_asl",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["workers.alphabet_tasks", "workers.phrase_tasks"],
)
celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    result_expires=300,
    task_routes={
        "workers.alphabet_tasks.predict_alphabet": {"queue": "alphabet_queue"},
        "workers.phrase_tasks.predict_phrase": {"queue": "phrase_queue"},
        "workers.phrase_tasks.normalize_phrase": {"queue": "phrase_queue"},
    },
    worker_prefetch_multiplier=1,
    task_track_started=True,
)
