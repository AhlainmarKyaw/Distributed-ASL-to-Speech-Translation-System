"""Tests for distributed task status storage and API lookup."""

from __future__ import annotations

from copy import deepcopy
from uuid import uuid4

import pytest
from celery.exceptions import TimeoutError as CeleryTimeoutError
from fastapi.testclient import TestClient

import coordinator.main as coordinator_main
from coordinator.result_store import (
    CREATE_SCRIPT,
    TRANSITION_SCRIPT,
    InvalidTaskTransition,
    TaskStatusStore,
)


class OnlineWorkerRegistry:
    def list_workers(self):
        return [{"worker_type": "alphabet", "status": "online"}]


@pytest.fixture(autouse=True)
def online_worker(monkeypatch):
    monkeypatch.setattr(coordinator_main, "worker_registry", OnlineWorkerRegistry())


class FakeTaskRedis:
    def __init__(self):
        self.hashes = {}
        self.expirations = {}

    def eval(self, script, _number_of_keys, key, *args):
        if script == CREATE_SCRIPT:
            if key in self.hashes:
                return 0
            request_id, task_id, task_type, submitted_at, expiration = args
            self.hashes[key] = {
                "request_id": str(request_id),
                "task_id": str(task_id),
                "task_type": str(task_type),
                "state": "queued",
                "worker_id": "",
                "submitted_at": str(submitted_at),
                "started_at": "",
                "completed_at": "",
                "processing_ms": "",
                "error_code": "",
            }
            self.expirations[key] = int(expiration)
            return 1
        if script == TRANSITION_SCRIPT:
            if key not in self.hashes:
                return -1
            target, timestamp_field, timestamp, worker_id, processing_ms, error_code, expiration, *allowed = args
            record = self.hashes[key]
            if record["state"] == target:
                self.expirations[key] = int(expiration)
                return 2
            if record["state"] not in allowed:
                return 0
            record["state"] = str(target)
            if timestamp_field:
                record[str(timestamp_field)] = str(timestamp)
            if worker_id:
                record["worker_id"] = str(worker_id)
            if processing_ms != "":
                record["processing_ms"] = str(processing_ms)
            if error_code:
                record["error_code"] = str(error_code)
            self.expirations[key] = int(expiration)
            return 1
        raise AssertionError("Unknown Lua script")

    def hgetall(self, key):
        return deepcopy(self.hashes.get(key, {}))


def make_store(now=None):
    current = now or [1_700_000_000.0]
    return TaskStatusStore(
        FakeTaskRedis(), expiration_seconds=600, clock=lambda: current[0]
    ), current


def create_task(store, request_id=None):
    request_id = request_id or str(uuid4())
    task_id = str(uuid4())
    store.create_queued(request_id, task_id, "alphabet_prediction")
    return request_id, task_id


def test_successful_task_records_all_timestamps_and_worker() -> None:
    store, now = make_store()
    request_id, task_id = create_task(store)

    now[0] += 1
    store.mark_processing(request_id, "alphabet-a1b2")
    now[0] += 0.025
    store.mark_completed(
        request_id, worker_id="alphabet-a1b2", processing_ms=25.0
    )

    task = store.get(request_id)
    assert task["request_id"] == request_id
    assert task["task_id"] == task_id
    assert task["task_type"] == "alphabet_prediction"
    assert task["state"] == "completed"
    assert task["worker_id"] == "alphabet-a1b2"
    assert task["submitted_at"]
    assert task["started_at"]
    assert task["completed_at"]
    assert task["processing_ms"] == 25.0
    assert task["error_code"] is None


def test_failed_task_records_safe_error_code() -> None:
    store, now = make_store()
    request_id, _task_id = create_task(store)
    store.mark_processing(request_id, "phrase-a1b2")
    now[0] += 0.1

    store.mark_failed(
        request_id,
        worker_id="phrase-a1b2",
        processing_ms=100,
        error_code="INFERENCE_FAILED",
    )

    task = store.get(request_id)
    assert task["state"] == "failed"
    assert task["error_code"] == "INFERENCE_FAILED"
    assert "trace" not in task


def test_timed_out_task_is_terminal() -> None:
    store, _now = make_store()
    request_id, _task_id = create_task(store)

    store.mark_timed_out(request_id, processing_ms=10_000)

    task = store.get(request_id)
    assert task["state"] == "timed_out"
    assert task["error_code"] == "TASK_TIMEOUT"
    assert task["completed_at"]
    with pytest.raises(InvalidTaskTransition):
        store.mark_completed(request_id, worker_id="late-worker", processing_ms=11_000)


def test_status_records_use_configured_expiration() -> None:
    store, _now = make_store()
    request_id, _task_id = create_task(store)

    assert store.redis.expirations[store._key(request_id)] == 600
    store.mark_processing(request_id, "worker-a")
    assert store.redis.expirations[store._key(request_id)] == 600


def test_unknown_task_endpoint_returns_404(monkeypatch) -> None:
    store, _now = make_store()
    monkeypatch.setattr(coordinator_main, "task_store", store)

    response = TestClient(coordinator_main.app).get(f"/tasks/{uuid4()}")

    assert response.status_code == 404
    assert response.json() == {"detail": "Task not found"}


def test_task_endpoint_returns_stored_status(monkeypatch) -> None:
    store, _now = make_store()
    request_id, task_id = create_task(store)
    monkeypatch.setattr(coordinator_main, "task_store", store)

    response = TestClient(coordinator_main.app).get(f"/tasks/{request_id}")

    assert response.status_code == 200
    assert response.json()["request_id"] == request_id
    assert response.json()["task_id"] == task_id
    assert response.json()["state"] == "queued"


class CoordinatorTask:
    def __init__(self, *, result=None, error=None):
        self.result = result
        self.error = error
        self.task_id = None

    def apply_async(self, *, args, queue, task_id):
        self.task_id = task_id
        return self

    def get(self, **_kwargs):
        if self.error:
            raise self.error
        return self.result


def test_synchronous_success_finishes_status_record(monkeypatch) -> None:
    store, _now = make_store()
    request_id = str(uuid4())
    task = CoordinatorTask(result={"status": "predicted", "worker_id": "worker-a"})
    monkeypatch.setattr(coordinator_main, "task_store", store)
    monkeypatch.setattr(coordinator_main, "predict_alphabet", task)

    response = TestClient(coordinator_main.app).post(
        "/predict/alphabet",
        json={"features": [0.0] * 63, "client_request_id": request_id},
    )

    assert response.status_code == 200
    status = store.get(request_id)
    assert status["state"] == "completed"
    assert status["task_id"] == task.task_id
    assert status["worker_id"] == "worker-a"


def test_synchronous_failure_records_failed_without_trace(monkeypatch) -> None:
    store, _now = make_store()
    request_id = str(uuid4())
    task = CoordinatorTask(error=RuntimeError("private internal traceback detail"))
    monkeypatch.setattr(coordinator_main, "task_store", store)
    monkeypatch.setattr(coordinator_main, "predict_alphabet", task)

    response = TestClient(coordinator_main.app).post(
        "/predict/alphabet",
        json={"features": [0.0] * 63, "client_request_id": request_id},
    )

    assert response.status_code == 503
    assert "private" not in response.text
    assert store.get(request_id)["state"] == "failed"
    assert store.get(request_id)["error_code"] == "TASK_FAILED"


def test_synchronous_timeout_records_timed_out(monkeypatch) -> None:
    store, _now = make_store()
    request_id = str(uuid4())
    task = CoordinatorTask(error=CeleryTimeoutError("internal timeout detail"))
    monkeypatch.setattr(coordinator_main, "task_store", store)
    monkeypatch.setattr(coordinator_main, "predict_alphabet", task)

    response = TestClient(coordinator_main.app).post(
        "/predict/alphabet",
        json={"features": [0.0] * 63, "client_request_id": request_id},
    )

    assert response.status_code == 504
    assert response.json()["error_code"] == "TASK_TIMEOUT"
    assert "internal" not in response.text
    assert store.get(request_id)["state"] == "timed_out"
