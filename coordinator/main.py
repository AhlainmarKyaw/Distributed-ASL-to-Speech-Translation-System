from __future__ import annotations
import logging
import time
from typing import Any
from uuid import uuid4

from celery.exceptions import TimeoutError as CeleryTimeoutError
from kombu.exceptions import OperationalError as BrokerOperationalError
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from redis import Redis
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError
from pydantic import BaseModel, Field, field_validator
from config import settings
from coordinator.request_limits import RequestSizeLimitMiddleware
from input_validation import (
    validate_alphabet_features,
    validate_phrase_sequence,
    validate_phrase_text,
)
from observability import get_logger, log_event, new_request_id, resolve_request_id
from coordinator.worker_registry import RegistryUnavailable, WorkerRegistry
from coordinator.result_store import (
    InvalidTaskTransition,
    TaskAlreadyExists,
    TaskNotFound,
    TaskStatusStore,
    TaskStoreUnavailable,
)
from coordinator.metrics_store import (
    MetricsStore,
    MetricsStoreUnavailable,
    empty_task_metrics,
)
from workers.alphabet_tasks import predict_alphabet
from workers.phrase_tasks import predict_phrase, normalize_phrase

app=FastAPI(title="Distributed ASL-to-Speech Translation System",version="2.0.0",
 description="Distributed ASL recognition using FastAPI, Redis and specialized Celery AI workers.")
app.add_middleware(RequestSizeLimitMiddleware, max_bytes=settings.api_max_request_bytes)
logger = get_logger("coordinator")
worker_registry = WorkerRegistry()
task_store = TaskStatusStore()
metrics_store = MetricsStore()
COORDINATOR_STARTED_AT = time.monotonic()
TRANSIENT_COORDINATOR_ERRORS = (
    BrokerOperationalError,
    RedisConnectionError,
    RedisTimeoutError,
)
MAX_TRANSIENT_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 0.1

class AlphabetRequest(BaseModel):
    features:list[float]
    client_request_id: str | None = None

    @field_validator("features", mode="before")
    @classmethod
    def strict_features(cls, value):
        return validate_alphabet_features(value)
class PhraseRequest(BaseModel):
    sequence:list[list[float]]
    client_request_id: str | None = None

    @field_validator("sequence", mode="before")
    @classmethod
    def strict_sequence(cls, value):
        return validate_phrase_sequence(value)
class PhraseTextRequest(BaseModel):
    text: str = Field(max_length=1000)
    client_request_id: str | None = None

    @field_validator("text", mode="before")
    @classmethod
    def strict_text(cls, value):
        return validate_phrase_text(value)
class TaskStatusResponse(BaseModel):
    request_id: str
    task_id: str
    task_type: str
    state: str
    worker_id: str | None = None
    submitted_at: str
    started_at: str | None = None
    completed_at: str | None = None
    processing_ms: float | None = None
    error_code: str | None = None


@app.exception_handler(HTTPException)
async def http_exception_handler(_request: Request, exc: HTTPException):
    if isinstance(exc.detail, dict) and "request_id" in exc.detail:
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_request: Request, exc: RequestValidationError):
    request_id = new_request_id()
    log_event(
        logger,
        "request_validation_failed",
        component="coordinator",
        request_id=request_id,
        success=False,
        level=logging.WARNING,
    )
    return JSONResponse(
        status_code=422,
        content=jsonable_encoder(
            {"message": "Invalid request", "request_id": request_id,
             "error_code": "INVALID_INPUT",
             "errors": [
                 {"field": ".".join(str(part) for part in error["loc"]),
                  "type": error["type"]}
                 for error in exc.errors()
             ]}
        ),
    )


def _request_id_or_error(client_request_id: str | None) -> str:
    try:
        return resolve_request_id(client_request_id)
    except ValueError as exc:
        request_id = new_request_id()
        log_event(
            logger,
            "request_id_validation_failed",
            component="coordinator",
            request_id=request_id,
            success=False,
            level=logging.WARNING,
        )
        raise HTTPException(
            400, {"message": str(exc), "request_id": request_id}
        ) from exc


def _run_task(
    task: Any,
    *,
    args: list[Any],
    queue: str,
    timeout: float,
    request_id: str,
    task_name: str,
    worker_type: str,
) -> dict[str, Any]:
    try:
        available_workers = worker_registry.list_workers()
    except RegistryUnavailable as exc:
        raise HTTPException(
            503,
            {"message": "Redis unavailable", "request_id": request_id,
             "error_code": "REDIS_UNAVAILABLE"},
        ) from exc
    if not any(
        worker["worker_type"] == worker_type and worker["status"] == "online"
        for worker in available_workers
    ):
        raise HTTPException(
            503,
            {"message": f"No online {worker_type} worker available",
             "request_id": request_id, "error_code": "NO_WORKER_AVAILABLE"},
        )
    started = time.perf_counter()
    task_id = str(uuid4())
    try:
        task_store.create_queued(request_id, task_id, task_name)
    except TaskAlreadyExists as exc:
        raise HTTPException(
            409, {"message": "request_id already exists", "request_id": request_id}
        ) from exc
    except TaskStoreUnavailable as exc:
        raise HTTPException(
            503,
            {"message": "Redis unavailable", "request_id": request_id,
             "error_code": "REDIS_UNAVAILABLE"},
        ) from exc
    log_event(
        logger,
        "task_submitted",
        component="coordinator",
        request_id=request_id,
        task_name=task_name,
        success=None,
    )
    try:
        async_result = None
        for attempt in range(MAX_TRANSIENT_ATTEMPTS):
            try:
                async_result = task.apply_async(
                    args=[*args, request_id], queue=queue, task_id=task_id
                )
                break
            except TRANSIENT_COORDINATOR_ERRORS:
                if attempt + 1 >= MAX_TRANSIENT_ATTEMPTS:
                    raise
                time.sleep(RETRY_BACKOFF_SECONDS * (2**attempt))
        if async_result is None:
            raise BrokerOperationalError("Task publication did not return a result")

        for attempt in range(MAX_TRANSIENT_ATTEMPTS):
            try:
                result = async_result.get(
                    timeout=timeout, disable_sync_subtasks=False
                )
                break
            except TRANSIENT_COORDINATOR_ERRORS:
                if attempt + 1 >= MAX_TRANSIENT_ATTEMPTS:
                    raise
                time.sleep(RETRY_BACKOFF_SECONDS * (2**attempt))
        else:
            raise BrokerOperationalError("Task result could not be retrieved")

        response = dict(result)
        response.setdefault("request_id", request_id)
        elapsed_ms = (time.perf_counter() - started) * 1000
        if response.get("status") == "model_missing":
            raise HTTPException(
                503,
                {"message": "Model unavailable", "request_id": request_id,
                 "error_code": "MODEL_UNAVAILABLE"},
            )
        try:
            current = task_store.get(request_id)
            if current and current["state"] not in {"completed", "failed", "timed_out"}:
                task_store.mark_completed(
                    request_id,
                    worker_id=response.get("worker_id"),
                    processing_ms=elapsed_ms,
                )
        except (TaskStoreUnavailable, TaskNotFound, InvalidTaskTransition):
            pass
        log_event(
            logger,
            "task_completed",
            component="coordinator",
            request_id=request_id,
            worker_id=response.get("worker_id"),
            task_name=task_name,
            processing_ms=elapsed_ms,
            success=True,
        )
        return response
    except HTTPException:
        raise
    except CeleryTimeoutError as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000
        try:
            task_store.mark_timed_out(request_id, processing_ms=elapsed_ms)
        except (TaskStoreUnavailable, TaskNotFound, InvalidTaskTransition):
            pass
        log_event(
            logger, "task_timed_out", component="coordinator",
            request_id=request_id, task_name=task_name,
            processing_ms=elapsed_ms, success=False, level=logging.ERROR,
        )
        raise HTTPException(
            504,
            {"message": "Task timed out", "request_id": request_id,
             "error_code": "TASK_TIMEOUT"},
        ) from exc
    except TRANSIENT_COORDINATOR_ERRORS as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000
        try:
            current = task_store.get(request_id)
            if current and current["state"] not in {"completed", "failed", "timed_out"}:
                task_store.mark_failed(
                    request_id,
                    error_code="REDIS_UNAVAILABLE",
                    processing_ms=elapsed_ms,
                )
        except (TaskStoreUnavailable, TaskNotFound, InvalidTaskTransition):
            pass
        raise HTTPException(
            503,
            {"message": "Redis unavailable", "request_id": request_id,
             "error_code": "REDIS_UNAVAILABLE"},
        ) from exc
    except ValueError as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000
        try:
            current = task_store.get(request_id)
            if current and current["state"] not in {"completed", "failed", "timed_out"}:
                task_store.mark_failed(
                    request_id, error_code="INVALID_INPUT", processing_ms=elapsed_ms
                )
        except (TaskStoreUnavailable, TaskNotFound, InvalidTaskTransition):
            pass
        raise HTTPException(
            400,
            {"message": "Invalid prediction input", "request_id": request_id,
             "error_code": "INVALID_INPUT"},
        ) from exc
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000
        try:
            current = task_store.get(request_id)
            if current and current["state"] not in {"completed", "failed", "timed_out"}:
                task_store.mark_failed(
                    request_id, error_code="TASK_FAILED", processing_ms=elapsed_ms
                )
        except (TaskStoreUnavailable, TaskNotFound, InvalidTaskTransition):
            pass
        log_event(
            logger,
            "task_failed",
            component="coordinator",
            request_id=request_id,
            task_name=task_name,
            processing_ms=elapsed_ms,
            success=False,
            level=logging.ERROR,
        )
        raise HTTPException(
            503,
            {"message": "Internal worker failure", "request_id": request_id,
             "error_code": "WORKER_INTERNAL_ERROR"},
        ) from exc

@app.get("/")
def root(): return {"project":"Distributed ASL-to-Speech Translation System","docs":"/docs","health":"/health"}

@app.get("/health")
def health():
    ok=False
    try:
        Redis.from_url(settings.redis_url).ping(); ok=True
    except Exception: pass
    return {"status":"ok","redis":ok,"architecture":"Client → FastAPI → Redis → Celery workers → TensorFlow → Speech"}


@app.get("/workers")
def workers_status():
    try:
        workers = worker_registry.list_workers()
        return {"registry_available": True, "workers": workers, "count": len(workers)}
    except RegistryUnavailable:
        log_event(
            logger, "worker_registry_unavailable", component="coordinator",
            success=False, level=logging.WARNING,
        )
        return {"registry_available": False, "workers": [], "count": 0}


@app.get("/workers/{worker_id}")
def worker_status(worker_id: str):
    try:
        worker = worker_registry.get(worker_id)
    except RegistryUnavailable as exc:
        log_event(
            logger, "worker_registry_unavailable", component="coordinator",
            worker_id=worker_id, success=False, level=logging.WARNING,
        )
        raise HTTPException(503, "Worker registry unavailable") from exc
    if worker is None:
        raise HTTPException(404, "Worker not found")
    return worker


@app.get(
    "/tasks/{request_id}",
    response_model=TaskStatusResponse,
    summary="Get distributed task status",
)
def task_status(request_id: str):
    try:
        task = task_store.get(request_id)
    except TaskStoreUnavailable as exc:
        raise HTTPException(503, "Task status store unavailable") from exc
    if task is None:
        raise HTTPException(404, "Task not found")
    return task


@app.get(
    "/metrics/summary",
    summary="Get lightweight distributed-system metrics",
)
def metrics_summary():
    try:
        task_metrics = metrics_store.snapshot()
        redis_connected = True
    except MetricsStoreUnavailable:
        task_metrics = empty_task_metrics()
        redis_connected = False

    workers = []
    if redis_connected:
        try:
            workers = worker_registry.list_workers()
        except RegistryUnavailable:
            redis_connected = False

    grouped = {
        "alphabet": {"online": 0, "offline": 0, "total": 0},
        "phrase": {"online": 0, "offline": 0, "total": 0},
    }
    online = offline = 0
    per_worker = []
    for worker in workers:
        worker_type = worker["worker_type"]
        status = worker["status"]
        grouped.setdefault(worker_type, {"online": 0, "offline": 0, "total": 0})
        grouped[worker_type][status] += 1
        grouped[worker_type]["total"] += 1
        if status == "online":
            online += 1
        else:
            offline += 1
        per_worker.append({
            "worker_id": worker["worker_id"],
            "worker_type": worker_type,
            "status": status,
            "completed_task_count": worker["completed_task_count"],
            "average_processing_ms": worker["average_processing_ms"],
        })

    return {
        "coordinator_uptime_seconds": round(
            time.monotonic() - COORDINATOR_STARTED_AT, 3
        ),
        "redis_connected": redis_connected,
        "workers": {"online": online, "offline": offline, "by_type": grouped},
        "tasks": task_metrics,
        "per_worker": per_worker,
    }

@app.post("/predict/alphabet")
def alphabet(req:AlphabetRequest):
    request_id = _request_id_or_error(req.client_request_id)
    return _run_task(
        predict_alphabet,
        args=[req.features],
        queue=settings.alphabet_queue,
        timeout=settings.task_timeout_seconds,
        request_id=request_id,
        task_name="workers.alphabet_tasks.predict_alphabet",
        worker_type="alphabet",
    )

@app.post("/predict/phrase")
def phrase_predict(req:PhraseRequest):
    request_id = _request_id_or_error(req.client_request_id)
    return _run_task(
        predict_phrase,
        args=[req.sequence],
        queue=settings.phrase_queue,
        timeout=settings.phrase_prediction_timeout_seconds,
        request_id=request_id,
        task_name="workers.phrase_tasks.predict_phrase",
        worker_type="phrase",
    )

@app.post("/phrase")
def phrase(payload:PhraseTextRequest):
    request_id = _request_id_or_error(payload.client_request_id)
    text=payload.text
    return _run_task(
        normalize_phrase,
        args=[text],
        queue=settings.phrase_queue,
        timeout=settings.phrase_normalization_timeout_seconds,
        request_id=request_id,
        task_name="workers.phrase_tasks.normalize_phrase",
        worker_type="phrase",
    )
