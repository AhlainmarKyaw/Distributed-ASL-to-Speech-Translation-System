"""Failure-isolated worker updates to the distributed task status store."""

from __future__ import annotations

import logging

from coordinator.result_store import (
    InvalidTaskTransition,
    TaskNotFound,
    TaskStatusStore,
    TaskStoreUnavailable,
)
from observability import get_logger, log_event


logger = get_logger("task_tracking")
task_store = TaskStatusStore()


def _safe_update(action, *, request_id: str | None, worker_id: str) -> None:
    if request_id is None:
        return
    try:
        action()
    except (TaskStoreUnavailable, TaskNotFound, InvalidTaskTransition):
        log_event(
            logger, "task_status_update_skipped", component="task_tracking",
            request_id=request_id, worker_id=worker_id,
            success=False, level=logging.WARNING,
        )


def mark_processing(request_id: str | None, worker_id: str) -> None:
    _safe_update(
        lambda: task_store.mark_processing(request_id, worker_id),
        request_id=request_id,
        worker_id=worker_id,
    )


def mark_completed(
    request_id: str | None, worker_id: str, processing_ms: float
) -> None:
    _safe_update(
        lambda: task_store.mark_completed(
            request_id, worker_id=worker_id, processing_ms=processing_ms
        ),
        request_id=request_id,
        worker_id=worker_id,
    )


def mark_failed(
    request_id: str | None, worker_id: str, processing_ms: float, error_code: str
) -> None:
    _safe_update(
        lambda: task_store.mark_failed(
            request_id,
            worker_id=worker_id,
            processing_ms=processing_ms,
            error_code=error_code,
        ),
        request_id=request_id,
        worker_id=worker_id,
    )
