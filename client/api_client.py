"""Headless HTTP boundary used by the desktop client."""

from __future__ import annotations

from typing import Any

import requests

from config import settings


def post_alphabet(features: list[float], client_request_id: str) -> dict[str, Any]:
    response = requests.post(
        settings.api_url + "/predict/alphabet",
        json={"features": features, "client_request_id": client_request_id},
        timeout=settings.task_timeout_seconds,
    )
    response.raise_for_status()
    return response.json()


def post_phrase_text(text: str, client_request_id: str) -> dict[str, Any]:
    response = requests.post(
        settings.api_url + "/phrase",
        json={"text": text, "client_request_id": client_request_id},
        timeout=settings.phrase_normalization_timeout_seconds,
    )
    response.raise_for_status()
    return response.json()
