"""Tests for centralized environment-based configuration."""

from __future__ import annotations

import pytest

from config import load_settings


def test_default_settings_are_local_and_preserve_existing_timeouts() -> None:
    configured = load_settings({})

    assert configured.api_host == "127.0.0.1"
    assert configured.api_port == 8000
    assert configured.api_url == "http://127.0.0.1:8000"
    assert configured.redis_url == "redis://127.0.0.1:6379/0"
    assert configured.alphabet_queue == "alphabet_queue"
    assert configured.phrase_queue == "phrase_queue"
    assert configured.task_timeout_seconds == 10
    assert configured.health_timeout_seconds == 2
    assert configured.phrase_prediction_timeout_seconds == 12
    assert configured.phrase_normalization_timeout_seconds == 6
    assert configured.worker_heartbeat_seconds == 5
    assert configured.worker_offline_seconds == 15
    assert configured.task_status_expiration_seconds == 3600
    assert configured.api_max_request_bytes == 262144


def test_environment_overrides_all_supported_settings() -> None:
    configured = load_settings(
        {
            "API_URL": "http://192.0.2.10:9000/",
            "API_HOST": "0.0.0.0",
            "API_PORT": "9000",
            "REDIS_URL": "redis://redis.internal:6380/2",
            "ALPHABET_QUEUE": "letters",
            "PHRASE_QUEUE": "signs",
            "TASK_TIMEOUT_SECONDS": "20.5",
            "WORKER_HEARTBEAT_SECONDS": "3",
            "WORKER_OFFLINE_SECONDS": "12.5",
            "TASK_STATUS_EXPIRATION_SECONDS": "7200",
            "API_MAX_REQUEST_BYTES": "131072",
        }
    )

    assert configured.api_url == "http://192.0.2.10:9000"
    assert configured.api_host == "0.0.0.0"
    assert configured.api_port == 9000
    assert configured.redis_url == "redis://redis.internal:6380/2"
    assert configured.alphabet_queue == "letters"
    assert configured.phrase_queue == "signs"
    assert configured.task_timeout_seconds == 20.5
    assert configured.worker_heartbeat_seconds == 3
    assert configured.worker_offline_seconds == 12.5
    assert configured.task_status_expiration_seconds == 7200
    assert configured.api_max_request_bytes == 131072


def test_api_url_is_derived_from_host_and_port_when_not_explicit() -> None:
    configured = load_settings({"API_HOST": "192.0.2.20", "API_PORT": "8080"})

    assert configured.api_url == "http://192.0.2.20:8080"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("API_PORT", "0"),
        ("API_PORT", "not-a-port"),
        ("TASK_TIMEOUT_SECONDS", "0"),
        ("TASK_TIMEOUT_SECONDS", "nan"),
        ("TASK_TIMEOUT_SECONDS", "inf"),
        ("WORKER_HEARTBEAT_SECONDS", "-1"),
        ("WORKER_OFFLINE_SECONDS", "not-a-number"),
        ("TASK_STATUS_EXPIRATION_SECONDS", "0"),
        ("API_MAX_REQUEST_BYTES", "-1"),
    ],
)
def test_invalid_numeric_settings_are_rejected(name: str, value: str) -> None:
    with pytest.raises(ValueError, match=name):
        load_settings({name: value})
