"""Redis-backed distributed task status storage with atomic transitions."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any, Callable

from redis import Redis

from config import settings
from coordinator.metrics_store import MetricsStore, MetricsStoreUnavailable


TASK_KEY_PREFIX = "distributed_asl:tasks:"
TERMINAL_STATES = {"completed", "failed", "timed_out"}
ALLOWED_TRANSITIONS = {
    "queued": {"processing", "completed", "failed", "timed_out"},
    "processing": {"completed", "failed", "timed_out"},
    "completed": {"completed"},
    "failed": {"failed"},
    "timed_out": {"timed_out"},
}


CREATE_SCRIPT = """
-- TASK_STATUS_CREATE
if redis.call('EXISTS', KEYS[1]) == 1 then return 0 end
redis.call('HSET', KEYS[1],
  'request_id', ARGV[1], 'task_id', ARGV[2], 'task_type', ARGV[3],
  'state', 'queued', 'worker_id', '', 'submitted_at', ARGV[4],
  'started_at', '', 'completed_at', '', 'processing_ms', '', 'error_code', '')
redis.call('EXPIRE', KEYS[1], ARGV[5])
return 1
"""


TRANSITION_SCRIPT = """
-- TASK_STATUS_TRANSITION
if redis.call('EXISTS', KEYS[1]) == 0 then return -1 end
local current = redis.call('HGET', KEYS[1], 'state')
local target = ARGV[1]
local allowed = false
if current == target then
  redis.call('EXPIRE', KEYS[1], ARGV[7])
  return 2
end
for index = 8, #ARGV do
  if current == ARGV[index] then allowed = true end
end
if not allowed then return 0 end
redis.call('HSET', KEYS[1], 'state', target)
if ARGV[2] ~= '' then redis.call('HSET', KEYS[1], ARGV[2], ARGV[3]) end
if ARGV[4] ~= '' then redis.call('HSET', KEYS[1], 'worker_id', ARGV[4]) end
if ARGV[5] ~= '' then redis.call('HSET', KEYS[1], 'processing_ms', ARGV[5]) end
if ARGV[6] ~= '' then redis.call('HSET', KEYS[1], 'error_code', ARGV[6]) end
redis.call('EXPIRE', KEYS[1], ARGV[7])
return 1
"""


class TaskStoreUnavailable(RuntimeError):
    pass


class TaskAlreadyExists(RuntimeError):
    pass


class TaskNotFound(RuntimeError):
    pass


class InvalidTaskTransition(RuntimeError):
    pass


def _iso(epoch_seconds: float) -> str:
    return datetime.fromtimestamp(epoch_seconds, timezone.utc).isoformat()


def _decode(values: dict[Any, Any]) -> dict[str, str]:
    return {
        (key.decode() if isinstance(key, bytes) else str(key)):
        (value.decode() if isinstance(value, bytes) else str(value))
        for key, value in values.items()
    }


class TaskStatusStore:
    def __init__(
        self,
        redis_client: Any | None = None,
        *,
        expiration_seconds: int | None = None,
        clock: Callable[[], float] = time.time,
        metrics_store: MetricsStore | None = None,
    ) -> None:
        self.redis = redis_client or Redis.from_url(settings.redis_url)
        self.expiration_seconds = (
            expiration_seconds or settings.task_status_expiration_seconds
        )
        self.clock = clock
        self.metrics = metrics_store or MetricsStore(self.redis, clock=clock)

    def _key(self, request_id: str) -> str:
        return f"{TASK_KEY_PREFIX}{request_id}"

    def create_queued(self, request_id: str, task_id: str, task_type: str) -> None:
        try:
            created = self.redis.eval(
                CREATE_SCRIPT,
                1,
                self._key(request_id),
                request_id,
                task_id,
                task_type,
                _iso(self.clock()),
                self.expiration_seconds,
            )
        except Exception as exc:
            raise TaskStoreUnavailable("Task status store is unavailable") from exc
        if int(created) != 1:
            raise TaskAlreadyExists("request_id already exists")
        try:
            self.metrics.record_submitted()
        except MetricsStoreUnavailable:
            pass

    def mark_processing(self, request_id: str, worker_id: str) -> None:
        self._transition(
            request_id,
            "processing",
            timestamp_field="started_at",
            worker_id=worker_id,
        )

    def mark_completed(
        self, request_id: str, *, worker_id: str | None, processing_ms: float
    ) -> bool:
        changed = self._transition(
            request_id,
            "completed",
            timestamp_field="completed_at",
            worker_id=worker_id,
            processing_ms=processing_ms,
        )
        if changed:
            self._record_terminal_metric("completed", processing_ms)
        return changed

    def mark_failed(
        self,
        request_id: str,
        *,
        error_code: str,
        worker_id: str | None = None,
        processing_ms: float | None = None,
    ) -> bool:
        changed = self._transition(
            request_id,
            "failed",
            timestamp_field="completed_at",
            worker_id=worker_id,
            processing_ms=processing_ms,
            error_code=error_code,
        )
        if changed:
            self._record_terminal_metric("failed", processing_ms)
        return changed

    def mark_timed_out(self, request_id: str, *, processing_ms: float) -> bool:
        changed = self._transition(
            request_id,
            "timed_out",
            timestamp_field="completed_at",
            processing_ms=processing_ms,
            error_code="TASK_TIMEOUT",
        )
        if changed:
            self._record_terminal_metric("timed_out", processing_ms)
        return changed

    def _transition(
        self,
        request_id: str,
        target: str,
        *,
        timestamp_field: str = "",
        worker_id: str | None = None,
        processing_ms: float | None = None,
        error_code: str | None = None,
    ) -> bool:
        allowed_from = [
            state for state, targets in ALLOWED_TRANSITIONS.items() if target in targets
        ]
        try:
            result = self.redis.eval(
                TRANSITION_SCRIPT,
                1,
                self._key(request_id),
                target,
                timestamp_field,
                _iso(self.clock()) if timestamp_field else "",
                worker_id or "",
                "" if processing_ms is None else str(processing_ms),
                error_code or "",
                self.expiration_seconds,
                *allowed_from,
            )
        except Exception as exc:
            raise TaskStoreUnavailable("Task status store is unavailable") from exc
        if int(result) == -1:
            raise TaskNotFound(request_id)
        if int(result) == 0:
            raise InvalidTaskTransition(f"Cannot transition task to {target}")
        return int(result) == 1

    def _record_terminal_metric(
        self, state: str, processing_ms: float | None
    ) -> None:
        try:
            self.metrics.record_terminal(state, processing_ms)
        except MetricsStoreUnavailable:
            pass

    def get(self, request_id: str) -> dict[str, Any] | None:
        try:
            values = _decode(self.redis.hgetall(self._key(request_id)))
        except Exception as exc:
            raise TaskStoreUnavailable("Task status store is unavailable") from exc
        if not values:
            return None
        return {
            "request_id": values["request_id"],
            "task_id": values["task_id"],
            "task_type": values["task_type"],
            "state": values["state"],
            "worker_id": values.get("worker_id") or None,
            "submitted_at": values["submitted_at"],
            "started_at": values.get("started_at") or None,
            "completed_at": values.get("completed_at") or None,
            "processing_ms": (
                float(values["processing_ms"]) if values.get("processing_ms") else None
            ),
            "error_code": values.get("error_code") or None,
        }
