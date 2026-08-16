"""Tests for bounded distributed-system metric calculations."""

from __future__ import annotations

from uuid import uuid4

from fastapi.testclient import TestClient

import coordinator.main as coordinator_main
from coordinator.metrics_store import MetricsStore, MetricsStoreUnavailable
from coordinator.result_store import TaskStatusStore
from tests.test_task_status import FakeTaskRedis


class MetricsRedis(FakeTaskRedis):
    def __init__(self):
        super().__init__()
        self.values = {}

    def ping(self):
        return True

    def hincrby(self, key, field, amount):
        record = self.hashes.setdefault(key, {})
        value = int(record.get(field, 0)) + amount
        record[field] = str(value)
        return value

    def hincrbyfloat(self, key, field, amount):
        record = self.hashes.setdefault(key, {})
        value = float(record.get(field, 0)) + amount
        record[field] = str(value)
        return value

    def incr(self, key):
        value = int(self.values.get(key, 0)) + 1
        self.values[key] = str(value)
        return value

    def get(self, key):
        return self.values.get(key)

    def expire(self, key, seconds):
        self.expirations[key] = seconds
        return True


def test_empty_metrics_are_zero_safe() -> None:
    metrics = MetricsStore(MetricsRedis(), clock=lambda: 120.0)

    assert metrics.snapshot() == {
        "submitted": 0,
        "completed": 0,
        "failed": 0,
        "timed_out": 0,
        "average_latency_ms": 0.0,
        "recent_throughput_tasks_per_minute": 0,
    }


def test_task_counters_average_latency_and_throughput() -> None:
    now = [120.0]
    redis = MetricsRedis()
    metrics = MetricsStore(redis, clock=lambda: now[0])

    metrics.record_submitted()
    metrics.record_submitted()
    metrics.record_terminal("completed", 10)
    metrics.record_terminal("failed", 30)
    metrics.record_terminal("timed_out", None)

    assert metrics.snapshot() == {
        "submitted": 2,
        "completed": 1,
        "failed": 1,
        "timed_out": 1,
        "average_latency_ms": 20.0,
        "recent_throughput_tasks_per_minute": 2,
    }
    assert set(redis.values) == {"distributed_asl:metrics:submitted_minute:2"}
    assert next(iter(redis.expirations.values())) == 120


def test_throughput_uses_bounded_expiring_minute_buckets() -> None:
    now = [120.0]
    redis = MetricsRedis()
    metrics = MetricsStore(redis, clock=lambda: now[0])
    metrics.record_submitted()

    now[0] = 180.0
    assert metrics.snapshot()["recent_throughput_tasks_per_minute"] == 0
    metrics.record_submitted()

    assert metrics.snapshot()["recent_throughput_tasks_per_minute"] == 1
    assert len(redis.values) == 2
    assert all(ttl == 120 for key, ttl in redis.expirations.items()
               if "submitted_minute" in key)


def test_terminal_retry_is_not_double_counted() -> None:
    now = [300.0]
    redis = MetricsRedis()
    metrics = MetricsStore(redis, clock=lambda: now[0])
    tasks = TaskStatusStore(
        redis, expiration_seconds=600, clock=lambda: now[0], metrics_store=metrics
    )
    request_id = str(uuid4())
    tasks.create_queued(request_id, str(uuid4()), "alphabet_prediction")
    tasks.mark_processing(request_id, "worker-a")
    assert tasks.mark_completed(request_id, worker_id="worker-a", processing_ms=12) is True
    assert tasks.mark_completed(request_id, worker_id="worker-a", processing_ms=12) is False

    snapshot = metrics.snapshot()
    assert snapshot["submitted"] == 1
    assert snapshot["completed"] == 1
    assert snapshot["average_latency_ms"] == 12


class MetricsSnapshot:
    def snapshot(self):
        return {
            "submitted": 10,
            "completed": 7,
            "failed": 2,
            "timed_out": 1,
            "average_latency_ms": 25.5,
            "recent_throughput_tasks_per_minute": 3,
        }


class WorkerSnapshot:
    def list_workers(self):
        return [
            {
                "worker_id": "alphabet-a",
                "worker_type": "alphabet",
                "status": "online",
                "completed_task_count": 5,
                "average_processing_ms": 20.0,
            },
            {
                "worker_id": "phrase-b",
                "worker_type": "phrase",
                "status": "offline",
                "completed_task_count": 2,
                "average_processing_ms": 40.0,
            },
        ]


def test_metrics_endpoint_groups_workers_and_exposes_per_worker_data(monkeypatch) -> None:
    monkeypatch.setattr(coordinator_main, "metrics_store", MetricsSnapshot())
    monkeypatch.setattr(coordinator_main, "worker_registry", WorkerSnapshot())

    response = TestClient(coordinator_main.app).get("/metrics/summary")

    assert response.status_code == 200
    data = response.json()
    assert data["coordinator_uptime_seconds"] >= 0
    assert data["redis_connected"] is True
    assert data["workers"]["online"] == 1
    assert data["workers"]["offline"] == 1
    assert data["workers"]["by_type"]["alphabet"]["online"] == 1
    assert data["workers"]["by_type"]["phrase"]["offline"] == 1
    assert data["tasks"]["submitted"] == 10
    assert data["per_worker"][0]["completed_task_count"] == 5


class UnavailableMetrics:
    def snapshot(self):
        raise MetricsStoreUnavailable("unavailable")


def test_metrics_endpoint_handles_redis_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(coordinator_main, "metrics_store", UnavailableMetrics())

    response = TestClient(coordinator_main.app).get("/metrics/summary")

    assert response.status_code == 200
    data = response.json()
    assert data["redis_connected"] is False
    assert data["tasks"]["submitted"] == 0
    assert data["tasks"]["average_latency_ms"] == 0
    assert data["workers"]["online"] == 0
    assert data["per_worker"] == []
