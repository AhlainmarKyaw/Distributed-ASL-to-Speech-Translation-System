"""End-to-end request-ID propagation and structured logging tests."""

from __future__ import annotations

import json
import logging
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

import coordinator.main as coordinator_main
import client.api_client as api_client
import workers.alphabet_tasks as alphabet_tasks
import workers.phrase_tasks as phrase_tasks
from observability import JsonFormatter, resolve_request_id


class ImmediateResult:
    def __init__(self, value):
        self.value = value

    def get(self, **_kwargs):
        return self.value


class RecordingTask:
    def __init__(self, value):
        self.value = value
        self.args = None
        self.queue = None

    def apply_async(self, *, args, queue, task_id):
        self.args = args
        self.queue = queue
        self.task_id = task_id
        return ImmediateResult(self.value)


class MemoryTaskStore:
    def __init__(self):
        self.values = {}

    def create_queued(self, request_id, task_id, task_type):
        self.values[request_id] = {"state": "queued", "task_id": task_id,
                                   "task_type": task_type}

    def get(self, request_id):
        return self.values.get(request_id)

    def mark_completed(self, request_id, **values):
        self.values[request_id].update(state="completed", **values)

    def mark_failed(self, request_id, **values):
        self.values[request_id].update(state="failed", **values)

    def mark_timed_out(self, request_id, **values):
        self.values[request_id].update(state="timed_out", **values)


class OnlineWorkerRegistry:
    def list_workers(self):
        return [
            {"worker_type": "alphabet", "status": "online"},
            {"worker_type": "phrase", "status": "online"},
        ]


@pytest.fixture(autouse=True)
def isolated_task_store(monkeypatch):
    monkeypatch.setattr(coordinator_main, "task_store", MemoryTaskStore())
    monkeypatch.setattr(coordinator_main, "worker_registry", OnlineWorkerRegistry())


def test_missing_client_request_id_is_generated_and_propagated(monkeypatch) -> None:
    task = RecordingTask({"status": "predicted", "letter": "A", "worker_id": "worker-1"})
    monkeypatch.setattr(coordinator_main, "predict_alphabet", task)

    response = TestClient(coordinator_main.app).post(
        "/predict/alphabet", json={"features": [0.0] * 63}
    )

    assert response.status_code == 200
    request_id = response.json()["request_id"]
    assert str(UUID(request_id)) == request_id
    assert task.args == [[0.0] * 63, request_id]
    assert task.queue == coordinator_main.settings.alphabet_queue


def test_valid_client_request_id_is_preserved(monkeypatch) -> None:
    supplied = str(uuid4())
    task = RecordingTask({"text": "Thank you", "worker_id": "phrase-worker"})
    monkeypatch.setattr(coordinator_main, "normalize_phrase", task)

    response = TestClient(coordinator_main.app).post(
        "/phrase", json={"text": "THANKYOU", "client_request_id": supplied}
    )

    assert response.status_code == 200
    assert response.json()["request_id"] == supplied
    assert task.args == ["THANKYOU", supplied]


def test_phrase_prediction_propagates_request_id(monkeypatch) -> None:
    supplied = str(uuid4())
    task = RecordingTask({"status": "model_missing", "worker_id": "phrase-worker"})
    monkeypatch.setattr(coordinator_main, "predict_phrase", task)

    response = TestClient(coordinator_main.app).post(
        "/predict/phrase",
        json={"sequence": [[0.0] * 63 for _ in range(30)], "client_request_id": supplied},
    )

    assert response.status_code == 503
    assert response.json()["request_id"] == supplied
    assert response.json()["error_code"] == "MODEL_UNAVAILABLE"
    assert task.args == [[[0.0] * 63 for _ in range(30)], supplied]


def test_invalid_client_request_id_returns_traceable_error(monkeypatch) -> None:
    task = RecordingTask({"status": "predicted"})
    monkeypatch.setattr(coordinator_main, "predict_alphabet", task)

    response = TestClient(coordinator_main.app).post(
        "/predict/alphabet",
        json={"features": [0.0] * 63, "client_request_id": "not-a-uuid"},
    )

    assert response.status_code == 400
    assert "valid UUID" in response.json()["message"]
    assert str(UUID(response.json()["request_id"])) == response.json()["request_id"]
    assert task.args is None


def test_worker_failure_returns_request_id_without_internal_error(monkeypatch) -> None:
    class FailingTask:
        def apply_async(self, **_kwargs):
            raise RuntimeError("redis://user:secret@internal-host:6379/0")

    supplied = str(uuid4())
    monkeypatch.setattr(coordinator_main, "predict_alphabet", FailingTask())

    response = TestClient(coordinator_main.app).post(
        "/predict/alphabet",
        json={"features": [0.0] * 63, "client_request_id": supplied},
    )

    assert response.status_code == 503
    assert response.json() == {
        "message": "Internal worker failure",
        "request_id": supplied,
        "error_code": "WORKER_INTERNAL_ERROR",
    }
    assert "secret" not in response.text


def test_schema_validation_error_has_generated_request_id() -> None:
    response = TestClient(coordinator_main.app).post(
        "/predict/alphabet", json={"features": [0.0]}
    )

    assert response.status_code == 422
    assert str(UUID(response.json()["request_id"])) == response.json()["request_id"]


def test_request_id_resolver_canonicalizes_valid_uuid() -> None:
    supplied = uuid4()
    assert resolve_request_id(str(supplied).upper()) == str(supplied)


def test_structured_log_contains_required_fields_and_no_secret() -> None:
    record = logging.LogRecord(
        name="distributed_asl.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="task_completed",
        args=(),
        exc_info=None,
    )
    record.component = "test"
    record.request_id = str(uuid4())
    record.worker_id = "worker-1"
    record.task_name = "test.task"
    record.processing_ms = 12.5
    record.success = True

    payload = json.loads(JsonFormatter().format(record))

    assert set(
        ["timestamp", "level", "component", "request_id", "worker_id", "task_name",
         "processing_ms", "success"]
    ).issubset(payload)
    assert payload["success"] is True
    assert "redis://" not in json.dumps(payload)


def test_alphabet_worker_returns_request_and_worker_ids(monkeypatch) -> None:
    class Predictor:
        def predict(self, _features):
            return {"status": "predicted", "letter": "A", "confidence": 0.9}

    supplied = str(uuid4())
    monkeypatch.setattr(alphabet_tasks, "_predictor", lambda: Predictor())
    monkeypatch.setattr(alphabet_tasks, "mark_processing", lambda *_args: None)
    monkeypatch.setattr(alphabet_tasks, "mark_completed", lambda *_args: None)

    result = alphabet_tasks.predict_alphabet.run([0.0] * 63, supplied)

    assert result["request_id"] == supplied
    assert result["worker_id"]


def test_phrase_worker_returns_request_and_worker_ids(monkeypatch) -> None:
    supplied = str(uuid4())
    monkeypatch.setattr(phrase_tasks, "mark_processing", lambda *_args: None)
    monkeypatch.setattr(phrase_tasks, "mark_completed", lambda *_args: None)

    result = phrase_tasks.normalize_phrase.run("THANKYOU", supplied)

    assert result["request_id"] == supplied
    assert result["worker_id"]
    assert result["text"] == "Thank you"


def test_client_sends_request_id(monkeypatch) -> None:
    captured = {}

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "status": "predicted",
                "letter": "A",
                "request_id": captured["json"]["client_request_id"],
                "worker_id": "worker-1",
            }

    def fake_post(url, *, json, timeout):
        captured.update({"url": url, "json": json, "timeout": timeout})
        return Response()

    monkeypatch.setattr(api_client.requests, "post", fake_post)
    supplied = str(uuid4())

    result = api_client.post_alphabet([0.0] * 63, supplied)

    request_id = captured["json"]["client_request_id"]
    assert request_id == supplied
    assert captured["json"]["features"] == [0.0] * 63
    assert result["request_id"] == supplied
