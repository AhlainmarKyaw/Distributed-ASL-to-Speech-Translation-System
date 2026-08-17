"""Fault-tolerance simulations for coordinator and Celery workers."""

from __future__ import annotations

from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
from redis.exceptions import ConnectionError as RedisConnectionError

import coordinator.main as coordinator_main
import workers.alphabet_tasks as alphabet_tasks
from coordinator.result_store import TaskStatusStore
from coordinator.worker_registry import RegistryUnavailable
from tests.test_task_status import FakeTaskRedis
from workers.errors import PermanentWorkerError, TransientWorkerError


class OnlineRegistry:
    def list_workers(self):
        return [{"worker_type": "alphabet", "status": "online"}]


class NoWorkerRegistry:
    def list_workers(self):
        return []


class UnavailableRegistry:
    def list_workers(self):
        raise RegistryUnavailable("safe failure")


class ImmediateResult:
    def __init__(self, value):
        self.value = value

    def get(self, **_kwargs):
        return self.value


class TransientPublicationTask:
    def __init__(self):
        self.attempts = 0

    def apply_async(self, **_kwargs):
        self.attempts += 1
        if self.attempts < 3:
            raise RedisConnectionError("temporary broker failure")
        return ImmediateResult({"status": "predicted", "worker_id": "worker-a"})


def configured_store():
    return TaskStatusStore(FakeTaskRedis(), expiration_seconds=600)


def test_transient_publication_failure_retries_then_succeeds(monkeypatch) -> None:
    task = TransientPublicationTask()
    request_id = str(uuid4())
    store = configured_store()
    sleeps = []
    monkeypatch.setattr(coordinator_main, "predict_alphabet", task)
    monkeypatch.setattr(coordinator_main, "worker_registry", OnlineRegistry())
    monkeypatch.setattr(coordinator_main, "task_store", store)
    monkeypatch.setattr(coordinator_main.time, "sleep", sleeps.append)

    response = TestClient(coordinator_main.app).post(
        "/predict/alphabet",
        json={"features": [0.0] * 63, "client_request_id": request_id},
    )

    assert response.status_code == 200
    assert task.attempts == 3
    assert sleeps == [0.1, 0.2]
    assert store.get(request_id)["state"] == "completed"


def test_no_worker_available_is_not_published(monkeypatch) -> None:
    task = TransientPublicationTask()
    monkeypatch.setattr(coordinator_main, "predict_alphabet", task)
    monkeypatch.setattr(coordinator_main, "worker_registry", NoWorkerRegistry())

    response = TestClient(coordinator_main.app).post(
        "/predict/alphabet", json={"features": [0.0] * 63}
    )

    assert response.status_code == 503
    assert response.json()["error_code"] == "NO_WORKER_AVAILABLE"
    assert task.attempts == 0


def test_unavailable_redis_returns_specific_error_without_publication(monkeypatch) -> None:
    task = TransientPublicationTask()
    monkeypatch.setattr(coordinator_main, "predict_alphabet", task)
    monkeypatch.setattr(coordinator_main, "worker_registry", UnavailableRegistry())

    response = TestClient(coordinator_main.app).post(
        "/predict/alphabet", json={"features": [0.0] * 63}
    )

    assert response.status_code == 503
    assert response.json()["error_code"] == "REDIS_UNAVAILABLE"
    assert task.attempts == 0
    assert "safe failure" not in response.text


def test_transient_worker_failure_retries_then_succeeds(monkeypatch) -> None:
    calls = 0

    class Predictor:
        def predict(self, _features):
            nonlocal calls
            calls += 1
            if calls < 3:
                raise TransientWorkerError("temporary accelerator failure")
            return {"status": "predicted", "letter": "A", "confidence": 0.9}

    monkeypatch.setattr(alphabet_tasks, "_predictor", lambda: Predictor())
    monkeypatch.setattr(alphabet_tasks, "mark_processing", lambda *_args: None)
    monkeypatch.setattr(alphabet_tasks, "mark_completed", lambda *_args: None)
    monkeypatch.setattr(alphabet_tasks, "mark_failed", lambda *_args: None)

    result = alphabet_tasks.predict_alphabet.apply(
        args=[[0.0] * 63, str(uuid4())]
    ).get(propagate=True)

    assert result["letter"] == "A"
    assert calls == 3


def test_permanent_worker_failure_is_not_retried(monkeypatch) -> None:
    calls = 0

    class Predictor:
        def predict(self, _features):
            nonlocal calls
            calls += 1
            raise PermanentWorkerError("permanent model contract failure")

    monkeypatch.setattr(alphabet_tasks, "_predictor", lambda: Predictor())
    monkeypatch.setattr(alphabet_tasks, "mark_processing", lambda *_args: None)
    monkeypatch.setattr(alphabet_tasks, "mark_failed", lambda *_args: None)

    with pytest.raises(PermanentWorkerError):
        alphabet_tasks.predict_alphabet.apply(
            args=[[0.0] * 63, str(uuid4())]
        ).get(propagate=True)

    assert calls == 1


def test_celery_worker_loss_and_time_limits_are_finite() -> None:
    configuration = alphabet_tasks.celery_app.conf

    assert configuration.task_acks_late is True
    assert configuration.task_reject_on_worker_lost is True
    assert configuration.task_soft_time_limit > 0
    assert configuration.task_time_limit > configuration.task_soft_time_limit
    assert alphabet_tasks.predict_alphabet.max_retries == 3
