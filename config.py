"""Central environment-based configuration for all runtime components."""

from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Mapping


DEFAULT_API_HOST = "127.0.0.1"
DEFAULT_API_PORT = 8000
DEFAULT_REDIS_URL = "redis://127.0.0.1:6379/0"
DEFAULT_ALPHABET_QUEUE = "alphabet_queue"
DEFAULT_PHRASE_QUEUE = "phrase_queue"
DEFAULT_TASK_TIMEOUT_SECONDS = 10.0
DEFAULT_WORKER_HEARTBEAT_SECONDS = 5.0
DEFAULT_WORKER_OFFLINE_SECONDS = 15.0
DEFAULT_TASK_STATUS_EXPIRATION_SECONDS = 3600
DEFAULT_API_MAX_REQUEST_BYTES = 262144


def _positive_int(environment: Mapping[str, str], name: str, default: int) -> int:
    raw = environment.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _positive_float(environment: Mapping[str, str], name: str, default: float) -> float:
    raw = environment.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _text(environment: Mapping[str, str], name: str, default: str) -> str:
    value = environment.get(name, "").strip()
    return value or default


@dataclass(frozen=True)
class Settings:
    api_url: str
    api_host: str
    api_port: int
    redis_url: str
    alphabet_queue: str
    phrase_queue: str
    task_timeout_seconds: float
    worker_heartbeat_seconds: float
    worker_offline_seconds: float
    task_status_expiration_seconds: int
    api_max_request_bytes: int

    @property
    def health_timeout_seconds(self) -> float:
        """Keep the existing short health-check timeout."""
        return min(2.0, self.task_timeout_seconds)

    @property
    def phrase_prediction_timeout_seconds(self) -> float:
        """Keep the phrase prediction's existing two-second allowance."""
        return self.task_timeout_seconds + 2.0

    @property
    def phrase_normalization_timeout_seconds(self) -> float:
        """Keep the lightweight phrase-normalization timeout below inference."""
        return max(1.0, self.task_timeout_seconds - 4.0)


def load_settings(environment: Mapping[str, str] | None = None) -> Settings:
    """Read and validate configuration from an environment mapping."""
    env = os.environ if environment is None else environment
    api_host = _text(env, "API_HOST", DEFAULT_API_HOST)
    api_port = _positive_int(env, "API_PORT", DEFAULT_API_PORT)
    api_url = _text(env, "API_URL", f"http://{api_host}:{api_port}").rstrip("/")

    return Settings(
        api_url=api_url,
        api_host=api_host,
        api_port=api_port,
        redis_url=_text(env, "REDIS_URL", DEFAULT_REDIS_URL),
        alphabet_queue=_text(env, "ALPHABET_QUEUE", DEFAULT_ALPHABET_QUEUE),
        phrase_queue=_text(env, "PHRASE_QUEUE", DEFAULT_PHRASE_QUEUE),
        task_timeout_seconds=_positive_float(
            env, "TASK_TIMEOUT_SECONDS", DEFAULT_TASK_TIMEOUT_SECONDS
        ),
        worker_heartbeat_seconds=_positive_float(
            env, "WORKER_HEARTBEAT_SECONDS", DEFAULT_WORKER_HEARTBEAT_SECONDS
        ),
        worker_offline_seconds=_positive_float(
            env, "WORKER_OFFLINE_SECONDS", DEFAULT_WORKER_OFFLINE_SECONDS
        ),
        task_status_expiration_seconds=_positive_int(
            env,
            "TASK_STATUS_EXPIRATION_SECONDS",
            DEFAULT_TASK_STATUS_EXPIRATION_SECONDS,
        ),
        api_max_request_bytes=_positive_int(
            env, "API_MAX_REQUEST_BYTES", DEFAULT_API_MAX_REQUEST_BYTES
        ),
    )


settings = load_settings()
