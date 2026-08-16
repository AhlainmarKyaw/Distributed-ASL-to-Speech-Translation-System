"""Health endpoint tests that never connect to a real Redis service."""

from __future__ import annotations

from fastapi.testclient import TestClient

import coordinator.main as coordinator_main


class HealthyRedis:
    def ping(self):
        return True


def test_health_reports_coordinator_and_mocked_redis_online(monkeypatch) -> None:
    monkeypatch.setattr(
        coordinator_main.Redis, "from_url", lambda _url: HealthyRedis()
    )

    response = TestClient(coordinator_main.app).get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "redis": True,
        "architecture": "Client → FastAPI → Redis → Celery workers → TensorFlow → Speech",
    }


def test_health_remains_live_when_mocked_redis_is_unavailable(monkeypatch) -> None:
    def unavailable(_url):
        raise ConnectionError("private Redis host")

    monkeypatch.setattr(coordinator_main.Redis, "from_url", unavailable)

    response = TestClient(coordinator_main.app).get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["redis"] is False
    assert "private Redis host" not in response.text
