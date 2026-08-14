"""Small smoke tests for the Phase 1 scaffold."""

from fastapi.testclient import TestClient

from coordinator.main import app


def test_health_endpoint() -> None:
    """The coordinator exposes a liveness endpoint before later features exist."""
    response = TestClient(app).get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "phase": "scaffolding"}

