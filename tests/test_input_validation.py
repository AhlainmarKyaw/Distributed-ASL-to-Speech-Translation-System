"""Strict API and worker-boundary input validation tests."""

from __future__ import annotations

import json
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

import coordinator.main as coordinator_main
import workers.alphabet_tasks as alphabet_tasks
import workers.phrase_tasks as phrase_tasks
from coordinator.result_store import TaskStatusStore
from tests.test_task_status import FakeTaskRedis


class OnlineRegistry:
    def list_workers(self):
        return [
            {"worker_type": "alphabet", "status": "online"},
            {"worker_type": "phrase", "status": "online"},
        ]


class ImmediateTask:
    def __init__(self, value):
        self.value = value
        self.args = None

    def apply_async(self, *, args, queue, task_id):
        self.args = args
        return self

    def get(self, **_kwargs):
        return self.value


@pytest.fixture(autouse=True)
def isolated_services(monkeypatch):
    monkeypatch.setattr(coordinator_main, "worker_registry", OnlineRegistry())
    monkeypatch.setattr(
        coordinator_main,
        "task_store",
        TaskStatusStore(FakeTaskRedis(), expiration_seconds=600),
    )


@pytest.mark.parametrize(
    "features",
    [
        [0.0] * 62,
        [0.0] * 64,
        ["0"] + [0.0] * 62,
        [True] + [0.0] * 62,
        [None] + [0.0] * 62,
    ],
)
def test_api_rejects_malformed_alphabet_features(features) -> None:
    response = TestClient(coordinator_main.app).post(
        "/predict/alphabet", json={"features": features}
    )

    assert response.status_code == 422
    assert response.json()["error_code"] == "INVALID_INPUT"
    assert "traceback" not in response.text.lower()


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
def test_api_rejects_non_finite_alphabet_values(token) -> None:
    values = ",".join([token] + ["0"] * 62)
    response = TestClient(coordinator_main.app).post(
        "/predict/alphabet",
        content=f'{{"features":[{values}]}}',
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 422
    assert response.json()["error_code"] == "INVALID_INPUT"


def test_api_rejects_missing_alphabet_features() -> None:
    response = TestClient(coordinator_main.app).post("/predict/alphabet", json={})

    assert response.status_code == 422
    assert response.json()["error_code"] == "INVALID_INPUT"


def test_valid_existing_alphabet_payload_is_preserved(monkeypatch) -> None:
    task = ImmediateTask({"status": "predicted", "letter": "A", "worker_id": "worker-a"})
    monkeypatch.setattr(coordinator_main, "predict_alphabet", task)

    response = TestClient(coordinator_main.app).post(
        "/predict/alphabet", json={"features": [0] * 63}
    )

    assert response.status_code == 200
    assert task.args[0] == [0.0] * 63


@pytest.mark.parametrize(
    "sequence",
    [
        [[0.0] * 63 for _ in range(29)],
        [[0.0] * 63 for _ in range(31)],
        [[0.0] * 62] + [[0.0] * 63 for _ in range(29)],
        [[0.0] * 64] + [[0.0] * 63 for _ in range(29)],
        [["0"] + [0.0] * 62] + [[0.0] * 63 for _ in range(29)],
    ],
)
def test_api_rejects_invalid_phrase_sequence_shape_or_values(sequence) -> None:
    response = TestClient(coordinator_main.app).post(
        "/predict/phrase", json={"sequence": sequence}
    )

    assert response.status_code == 422
    assert response.json()["error_code"] == "INVALID_INPUT"


def test_api_rejects_non_finite_phrase_value() -> None:
    sequence = [[float("nan")] + [0.0] * 62] + [[0.0] * 63 for _ in range(29)]
    response = TestClient(coordinator_main.app).post(
        "/predict/phrase",
        content=json.dumps({"sequence": sequence}),
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 422
    assert response.json()["error_code"] == "INVALID_INPUT"


def test_valid_phrase_sequence_reaches_worker(monkeypatch) -> None:
    task = ImmediateTask({"status": "predicted", "phrase": "HELLO", "worker_id": "worker-p"})
    monkeypatch.setattr(coordinator_main, "predict_phrase", task)

    response = TestClient(coordinator_main.app).post(
        "/predict/phrase", json={"sequence": [[0] * 63 for _ in range(30)]}
    )

    assert response.status_code == 200
    assert task.args[0][0] == [0.0] * 63


def test_oversized_request_is_rejected_before_schema_parsing() -> None:
    oversized = "x" * (coordinator_main.settings.api_max_request_bytes + 1)

    response = TestClient(coordinator_main.app).post(
        "/phrase", content=oversized,
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 413
    assert response.json()["error_code"] == "REQUEST_TOO_LARGE"
    assert "x" * 20 not in response.text


@pytest.mark.parametrize(
    "features",
    [[0.0] * 62, ["0"] + [0.0] * 62, [float("inf")] + [0.0] * 62],
)
def test_alphabet_worker_revalidates_direct_tasks(monkeypatch, features) -> None:
    monkeypatch.setattr(alphabet_tasks, "mark_processing", lambda *_args: None)
    monkeypatch.setattr(alphabet_tasks, "mark_failed", lambda *_args: None)

    with pytest.raises(ValueError):
        alphabet_tasks.predict_alphabet.run(features, str(uuid4()))


@pytest.mark.parametrize(
    "sequence",
    [
        [[0.0] * 63 for _ in range(29)],
        [[0.0] * 62 for _ in range(30)],
        [[float("inf")] + [0.0] * 62 for _ in range(30)],
    ],
)
def test_phrase_worker_revalidates_direct_tasks(monkeypatch, sequence) -> None:
    monkeypatch.setattr(phrase_tasks, "mark_processing", lambda *_args: None)
    monkeypatch.setattr(phrase_tasks, "mark_failed", lambda *_args: None)

    with pytest.raises(ValueError):
        phrase_tasks.predict_phrase.run(sequence, str(uuid4()))


def test_validation_error_does_not_echo_submitted_string() -> None:
    secret_like_value = "do-not-echo-this-value"
    response = TestClient(coordinator_main.app).post(
        "/predict/alphabet",
        json={"features": [secret_like_value] + [0.0] * 62},
    )

    assert response.status_code == 422
    assert secret_like_value not in response.text
