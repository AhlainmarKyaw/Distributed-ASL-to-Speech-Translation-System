"""Unit tests for worker registration, heartbeat, and failure detection."""

from __future__ import annotations

import fnmatch
from copy import deepcopy

from fastapi.testclient import TestClient

import coordinator.main as coordinator_main
from coordinator.worker_registry import RegistryUnavailable, WorkerRegistry, build_worker_id
from workers.worker_lifecycle import WorkerLifecycleManager


class FakeRedis:
    def __init__(self):
        self.hashes = {}
        self.expirations = {}

    def hset(self, key, mapping):
        self.hashes.setdefault(key, {}).update({str(k): str(v) for k, v in mapping.items()})

    def hgetall(self, key):
        return deepcopy(self.hashes.get(key, {}))

    def expire(self, key, seconds):
        self.expirations[key] = seconds
        return True

    def exists(self, key):
        return key in self.hashes

    def scan_iter(self, match):
        return (key for key in self.hashes if fnmatch.fnmatch(key, match))

    def hincrby(self, key, field, amount):
        value = int(self.hashes[key].get(field, 0)) + amount
        self.hashes[key][field] = str(value)
        return value

    def hincrbyfloat(self, key, field, amount):
        value = float(self.hashes[key].get(field, 0)) + amount
        self.hashes[key][field] = str(value)
        return value


class FailingRedis:
    def __getattr__(self, _name):
        def fail(*_args, **_kwargs):
            raise ConnectionError("Redis unavailable")
        return fail


def make_registry(now, redis_client=None, offline_seconds=15):
    return WorkerRegistry(
        redis_client or FakeRedis(),
        offline_seconds=offline_seconds,
        heartbeat_seconds=5,
        clock=lambda: now[0],
    )


def test_registration_stores_complete_public_worker_record() -> None:
    now = [1_700_000_000.0]
    redis = FakeRedis()
    registry = make_registry(now, redis)
    worker_id = build_worker_id("alphabet", "private-laptop.local", "alphabet_queue")

    worker = registry.register(
        worker_id, "alphabet", "private-laptop.local", "alphabet_queue"
    )

    assert worker == {
        "worker_id": worker_id,
        "worker_type": "alphabet",
        "hostname": worker["hostname"],
        "queue_name": "alphabet_queue",
        "status": "online",
        "registered_at": "2023-11-14T22:13:20+00:00",
        "last_heartbeat": "2023-11-14T22:13:20+00:00",
        "completed_task_count": 0,
        "failed_task_count": 0,
        "average_processing_ms": 0.0,
    }
    assert worker["hostname"].startswith("host-")
    assert "private-laptop" not in str(worker)
    assert next(iter(redis.expirations.values())) == 30


def test_heartbeat_refreshes_timestamp_and_expiration() -> None:
    now = [100.0]
    redis = FakeRedis()
    registry = make_registry(now, redis)
    worker_id = build_worker_id("phrase", "host-a", "phrase_queue")
    registry.register(worker_id, "phrase", "host-a", "phrase_queue")

    now[0] = 109.0
    assert registry.heartbeat(worker_id) is True

    worker = registry.get(worker_id)
    assert worker["status"] == "online"
    assert worker["last_heartbeat"] == "1970-01-01T00:01:49+00:00"
    assert redis.expirations[registry._key(worker_id)] == 30


def test_worker_is_marked_offline_after_threshold() -> None:
    now = [200.0]
    registry = make_registry(now, offline_seconds=15)
    worker_id = build_worker_id("alphabet", "host-a", "alphabet_queue")
    registry.register(worker_id, "alphabet", "host-a", "alphabet_queue")

    now[0] = 216.0

    assert registry.get(worker_id)["status"] == "offline"
    assert registry.list_workers()[0]["status"] == "offline"


def test_task_statistics_are_updated() -> None:
    now = [300.0]
    registry = make_registry(now)
    worker_id = build_worker_id("alphabet", "host-a", "alphabet_queue")
    registry.register(worker_id, "alphabet", "host-a", "alphabet_queue")

    registry.record_task(worker_id, success=True, processing_ms=10)
    registry.record_task(worker_id, success=False, processing_ms=30)

    worker = registry.get(worker_id)
    assert worker["completed_task_count"] == 1
    assert worker["failed_task_count"] == 1
    assert worker["average_processing_ms"] == 20


def test_redis_unavailable_raises_safe_registry_error() -> None:
    registry = WorkerRegistry(FailingRedis())

    try:
        registry.list_workers()
    except RegistryUnavailable as exc:
        assert str(exc) == "Worker registry is unavailable"
        assert "Redis unavailable" not in str(exc)
    else:
        raise AssertionError("RegistryUnavailable was not raised")


def test_list_endpoint_survives_redis_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(
        coordinator_main, "worker_registry", WorkerRegistry(FailingRedis())
    )

    response = TestClient(coordinator_main.app).get("/workers")

    assert response.status_code == 200
    assert response.json() == {
        "registry_available": False,
        "workers": [],
        "count": 0,
    }


def test_worker_endpoint_returns_503_when_registry_is_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(
        coordinator_main, "worker_registry", WorkerRegistry(FailingRedis())
    )

    response = TestClient(coordinator_main.app).get("/workers/alphabet-example")

    assert response.status_code == 503
    assert response.json() == {"detail": "Worker registry unavailable"}


def test_lifecycle_start_is_idempotent_and_stop_marks_offline() -> None:
    now = [400.0]
    registry = make_registry(now)
    lifecycle = WorkerLifecycleManager(registry)

    first = lifecycle.start(
        worker_type="alphabet", hostname="host-a", queue_name="alphabet_queue"
    )
    first_thread = lifecycle._thread
    second = lifecycle.start(
        worker_type="alphabet", hostname="host-a", queue_name="alphabet_queue"
    )

    assert first == second
    assert lifecycle._thread is first_thread
    lifecycle.stop()
    assert registry.get(first)["status"] == "offline"
