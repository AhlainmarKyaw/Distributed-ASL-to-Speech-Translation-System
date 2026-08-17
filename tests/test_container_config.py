"""Static validation for the server-side container configuration."""

from __future__ import annotations

from pathlib import Path

import yaml

from workers.phrase_predictor import PhrasePredictor


ROOT = Path(__file__).resolve().parents[1]


def compose_config():
    return yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))


def test_compose_defines_all_server_side_services() -> None:
    services = compose_config()["services"]

    assert set(services) == {"redis", "coordinator", "alphabet-worker", "phrase-worker"}
    assert "client" not in services
    assert "ui" not in services


def test_application_services_share_one_image_and_build() -> None:
    services = compose_config()["services"]
    app_services = [services[name] for name in ("coordinator", "alphabet-worker", "phrase-worker")]

    assert {service["image"] for service in app_services} == {"distributed-asl-server:local"}
    assert all(service["build"]["dockerfile"] == "Dockerfile" for service in app_services)


def test_every_service_has_a_healthcheck() -> None:
    services = compose_config()["services"]

    assert all("healthcheck" in service for service in services.values())


def test_workers_are_scalable_and_use_environment_queues() -> None:
    services = compose_config()["services"]
    alphabet = services["alphabet-worker"]
    phrase = services["phrase-worker"]

    assert "container_name" not in alphabet
    assert "container_name" not in phrase
    assert "$${ALPHABET_QUEUE}" in alphabet["command"][-1]
    assert "$${PHRASE_QUEUE}" in phrase["command"][-1]
    assert alphabet["environment"]["ALPHABET_MODEL"] == "/app/models/alphabet_landmarks.keras"


def test_redis_persistence_is_explicit() -> None:
    config = compose_config()

    assert config["services"]["redis"]["volumes"] == ["redis_data:/data"]
    assert "redis_data" in config["volumes"]


def test_docker_image_excludes_host_only_and_large_content() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    ignored = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()

    assert "COPY --chown=asl:asl models ./models" in dockerfile
    assert "COPY client" not in dockerfile
    assert "COPY datasets" not in dockerfile
    for required_pattern in ("datasets/", ".venv/", "client/", "training/", "tests/", "logs/"):
        assert required_pattern in ignored


def test_missing_phrase_model_reports_model_unavailable_without_loading_tensorflow() -> None:
    predictor = PhrasePredictor()

    result = predictor.predict([[0.0] * 63 for _ in range(30)])

    assert result["status"] == "model_missing"
    assert result["phrase"] is None
    assert "Train phrase model" in result["message"]
