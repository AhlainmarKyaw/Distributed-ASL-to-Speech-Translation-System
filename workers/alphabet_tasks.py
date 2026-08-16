from __future__ import annotations
import logging
import socket
import time
from functools import lru_cache
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError
from config import settings
from observability import get_logger, log_event
from input_validation import validate_alphabet_features
from workers.celery_app import celery_app
from workers.alphabet_predictor import AlphabetPredictor
from workers.worker_lifecycle import manager, task_worker_id
from workers.task_tracking import mark_completed, mark_failed, mark_processing
from workers.errors import TransientWorkerError

logger = get_logger("alphabet_worker")

@lru_cache(maxsize=1)
def _predictor(): return AlphabetPredictor()

@celery_app.task(
    bind=True,
    name="workers.alphabet_tasks.predict_alphabet",
    autoretry_for=(TransientWorkerError, RedisConnectionError, RedisTimeoutError),
    retry_backoff=True,
    retry_backoff_max=8,
    retry_jitter=True,
    retry_kwargs={"max_retries": 3},
)
def predict_alphabet(self, features: list[float], request_id: str | None = None) -> dict:
    task_name = "workers.alphabet_tasks.predict_alphabet"
    hostname = getattr(self.request, "hostname", None) or socket.gethostname()
    worker_id = task_worker_id("alphabet", settings.alphabet_queue, hostname)
    started = time.perf_counter()
    mark_processing(request_id, worker_id)
    try:
        features = validate_alphabet_features(features)
    except ValueError:
        processing_ms = (time.perf_counter() - started) * 1000
        mark_failed(request_id, worker_id, processing_ms, "INVALID_INPUT")
        manager.record_task(success=False, processing_ms=processing_ms)
        log_event(
            logger, "input_validation_failed", component="alphabet_worker",
            request_id=request_id, worker_id=worker_id, task_name=task_name,
            processing_ms=processing_ms, success=False, level=logging.WARNING,
        )
        raise
    try:
        result = _predictor().predict(features)
        result.update({"request_id": request_id, "worker_id": worker_id})
        processing_ms = (time.perf_counter() - started) * 1000
        model_available = result.get("status") != "model_missing"
        if not model_available:
            mark_failed(request_id, worker_id, processing_ms, "MODEL_UNAVAILABLE")
        else:
            mark_completed(request_id, worker_id, processing_ms)
        manager.record_task(success=model_available, processing_ms=processing_ms)
        log_event(
            logger, "prediction_completed" if model_available else "model_unavailable",
            component="alphabet_worker",
            request_id=request_id, worker_id=worker_id, task_name=task_name,
            processing_ms=processing_ms, success=model_available,
        )
        return result
    except (TransientWorkerError, RedisConnectionError, RedisTimeoutError):
        processing_ms = (time.perf_counter() - started) * 1000
        log_event(
            logger, "transient_prediction_failure", component="alphabet_worker",
            request_id=request_id, worker_id=worker_id, task_name=task_name,
            processing_ms=processing_ms, success=False, level=logging.WARNING,
        )
        raise
    except Exception:
        processing_ms = (time.perf_counter() - started) * 1000
        mark_failed(request_id, worker_id, processing_ms, "INFERENCE_FAILED")
        manager.record_task(success=False, processing_ms=processing_ms)
        log_event(
            logger, "prediction_failed", component="alphabet_worker",
            request_id=request_id, worker_id=worker_id, task_name=task_name,
            processing_ms=processing_ms, success=False,
            level=logging.ERROR,
        )
        raise
