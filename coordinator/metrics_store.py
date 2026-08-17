"""Low-cardinality Redis counters for distributed-system metrics."""

from __future__ import annotations

import time
from typing import Any, Callable

from redis import Redis

from config import settings


TASK_METRICS_KEY = "distributed_asl:metrics:tasks"
THROUGHPUT_KEY_PREFIX = "distributed_asl:metrics:submitted_minute:"
THROUGHPUT_BUCKET_TTL_SECONDS = 120


class MetricsStoreUnavailable(RuntimeError):
    pass


def _decode(values: dict[Any, Any]) -> dict[str, str]:
    return {
        (key.decode() if isinstance(key, bytes) else str(key)):
        (value.decode() if isinstance(value, bytes) else str(value))
        for key, value in values.items()
    }


class MetricsStore:
    """Maintain fixed counters and one expiring counter per active UTC minute."""

    def __init__(
        self,
        redis_client: Any | None = None,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.redis = redis_client or Redis.from_url(settings.redis_url)
        self.clock = clock

    def _minute_key(self) -> str:
        return f"{THROUGHPUT_KEY_PREFIX}{int(self.clock() // 60)}"

    def record_submitted(self) -> None:
        minute_key = self._minute_key()
        try:
            self.redis.hincrby(TASK_METRICS_KEY, "submitted", 1)
            self.redis.incr(minute_key)
            self.redis.expire(minute_key, THROUGHPUT_BUCKET_TTL_SECONDS)
        except Exception as exc:
            raise MetricsStoreUnavailable("Metrics store is unavailable") from exc

    def record_terminal(self, state: str, processing_ms: float | None) -> None:
        if state not in {"completed", "failed", "timed_out"}:
            return
        try:
            self.redis.hincrby(TASK_METRICS_KEY, state, 1)
            if processing_ms is not None:
                self.redis.hincrbyfloat(
                    TASK_METRICS_KEY, "total_latency_ms", processing_ms
                )
                self.redis.hincrby(TASK_METRICS_KEY, "latency_count", 1)
        except Exception as exc:
            raise MetricsStoreUnavailable("Metrics store is unavailable") from exc

    def snapshot(self) -> dict[str, int | float]:
        try:
            self.redis.ping()
            values = _decode(self.redis.hgetall(TASK_METRICS_KEY))
            throughput = self.redis.get(self._minute_key())
        except Exception as exc:
            raise MetricsStoreUnavailable("Metrics store is unavailable") from exc
        if isinstance(throughput, bytes):
            throughput = throughput.decode()
        submitted = int(values.get("submitted", 0))
        completed = int(values.get("completed", 0))
        failed = int(values.get("failed", 0))
        timed_out = int(values.get("timed_out", 0))
        latency_count = int(values.get("latency_count", 0))
        total_latency = float(values.get("total_latency_ms", 0))
        return {
            "submitted": submitted,
            "completed": completed,
            "failed": failed,
            "timed_out": timed_out,
            "average_latency_ms": total_latency / latency_count if latency_count else 0.0,
            "recent_throughput_tasks_per_minute": int(throughput or 0),
        }


def empty_task_metrics() -> dict[str, int | float]:
    return {
        "submitted": 0,
        "completed": 0,
        "failed": 0,
        "timed_out": 0,
        "average_latency_ms": 0.0,
        "recent_throughput_tasks_per_minute": 0,
    }
