from __future__ import annotations
import logging
import re
import socket
import time
from functools import lru_cache
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError
from config import settings
from observability import get_logger, log_event
from input_validation import validate_phrase_sequence, validate_phrase_text
from workers.celery_app import celery_app
from workers.phrase_predictor import PhrasePredictor
from workers.worker_lifecycle import manager, task_worker_id
from workers.task_tracking import mark_completed, mark_failed, mark_processing
from workers.errors import TransientWorkerError

logger = get_logger("phrase_worker")

COMMON = {
 "HELLO":"Hello","HI":"Hi","THANKYOU":"Thank you","THANKS":"Thank you",
 "GOODMORNING":"Good morning","PLEASE":"Please","YES":"Yes","NO":"No",
 "HELP":"Help me","HELPME":"Help me","SORRY":"Sorry","ILOVEYOU":"I love you",
 "HOWAREYOU":"How are you?","NICETOMEETYOU":"Nice to meet you","GOODBYE":"Goodbye","BYE":"Goodbye"
}
@lru_cache(maxsize=1)
def _predictor(): return PhrasePredictor()

def _worker_id(task) -> str:
    hostname = getattr(task.request, "hostname", None) or socket.gethostname()
    return task_worker_id("phrase", settings.phrase_queue, hostname)


@celery_app.task(
    bind=True,
    name="workers.phrase_tasks.predict_phrase",
    autoretry_for=(TransientWorkerError, RedisConnectionError, RedisTimeoutError),
    retry_backoff=True,
    retry_backoff_max=8,
    retry_jitter=True,
    retry_kwargs={"max_retries": 3},
)
def predict_phrase(self, sequence: list[list[float]], request_id: str | None = None) -> dict:
    task_name = "workers.phrase_tasks.predict_phrase"
    worker_id = _worker_id(self)
    started = time.perf_counter()
    mark_processing(request_id, worker_id)
    try:
        sequence = validate_phrase_sequence(sequence)
    except ValueError:
        processing_ms = (time.perf_counter() - started) * 1000
        mark_failed(request_id, worker_id, processing_ms, "INVALID_INPUT")
        manager.record_task(success=False, processing_ms=processing_ms)
        log_event(
            logger, "input_validation_failed", component="phrase_worker",
            request_id=request_id, worker_id=worker_id, task_name=task_name,
            processing_ms=processing_ms, success=False, level=logging.WARNING,
        )
        raise
    try:
        result = _predictor().predict(sequence)
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
            component="phrase_worker",
            request_id=request_id, worker_id=worker_id, task_name=task_name,
            processing_ms=processing_ms, success=model_available,
        )
        return result
    except (TransientWorkerError, RedisConnectionError, RedisTimeoutError):
        processing_ms = (time.perf_counter() - started) * 1000
        log_event(
            logger, "transient_prediction_failure", component="phrase_worker",
            request_id=request_id, worker_id=worker_id, task_name=task_name,
            processing_ms=processing_ms, success=False, level=logging.WARNING,
        )
        raise
    except Exception:
        processing_ms = (time.perf_counter() - started) * 1000
        mark_failed(request_id, worker_id, processing_ms, "INFERENCE_FAILED")
        manager.record_task(success=False, processing_ms=processing_ms)
        log_event(
            logger, "prediction_failed", component="phrase_worker",
            request_id=request_id, worker_id=worker_id, task_name=task_name,
            processing_ms=processing_ms, success=False,
            level=logging.ERROR,
        )
        raise

@celery_app.task(
    bind=True,
    name="workers.phrase_tasks.normalize_phrase",
    autoretry_for=(TransientWorkerError, RedisConnectionError, RedisTimeoutError),
    retry_backoff=True,
    retry_backoff_max=8,
    retry_jitter=True,
    retry_kwargs={"max_retries": 3},
)
def normalize_phrase(self, text: str, request_id: str | None = None) -> dict:
    task_name = "workers.phrase_tasks.normalize_phrase"
    worker_id = _worker_id(self)
    started = time.perf_counter()
    mark_processing(request_id, worker_id)
    try:
        text = validate_phrase_text(text)
    except ValueError:
        processing_ms = (time.perf_counter() - started) * 1000
        mark_failed(request_id, worker_id, processing_ms, "INVALID_INPUT")
        manager.record_task(success=False, processing_ms=processing_ms)
        log_event(
            logger, "input_validation_failed", component="phrase_worker",
            request_id=request_id, worker_id=worker_id, task_name=task_name,
            processing_ms=processing_ms, success=False, level=logging.WARNING,
        )
        raise
    try:
        key=re.sub(r"[^A-Za-z]","",text).upper()
        out=COMMON.get(key,text.strip())
        result = {"input":text,"text":out,"matched":out != text.strip(),
                  "request_id":request_id,"worker_id":worker_id}
        processing_ms = (time.perf_counter() - started) * 1000
        mark_completed(request_id, worker_id, processing_ms)
        manager.record_task(success=True, processing_ms=processing_ms)
        log_event(
            logger, "normalization_completed", component="phrase_worker",
            request_id=request_id, worker_id=worker_id, task_name=task_name,
            processing_ms=processing_ms, success=True,
        )
        return result
    except (TransientWorkerError, RedisConnectionError, RedisTimeoutError):
        processing_ms = (time.perf_counter() - started) * 1000
        log_event(
            logger, "transient_normalization_failure", component="phrase_worker",
            request_id=request_id, worker_id=worker_id, task_name=task_name,
            processing_ms=processing_ms, success=False, level=logging.WARNING,
        )
        raise
    except Exception:
        processing_ms = (time.perf_counter() - started) * 1000
        mark_failed(request_id, worker_id, processing_ms, "NORMALIZATION_FAILED")
        manager.record_task(success=False, processing_ms=processing_ms)
        log_event(
            logger, "normalization_failed", component="phrase_worker",
            request_id=request_id, worker_id=worker_id, task_name=task_name,
            processing_ms=processing_ms, success=False,
            level=logging.ERROR,
        )
        raise
