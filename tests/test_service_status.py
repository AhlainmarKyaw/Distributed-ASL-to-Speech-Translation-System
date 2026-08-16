"""Headless service-status parsing and single-thread polling tests."""

from __future__ import annotations

import queue
import threading

import requests

from client.service_status import (
    ServiceStatus,
    StatusPoller,
    fetch_service_status,
    parse_service_status,
)


def test_status_parser_uses_metrics_worker_counts_and_task_fields() -> None:
    status = parse_service_status(
        {"status": "ok", "redis": True},
        {"workers": []},
        {
            "redis_connected": True,
            "workers": {
                "by_type": {
                    "alphabet": {"online": 2},
                    "phrase": {"online": 1},
                }
            },
        },
        {
            "request_id": "request-1",
            "worker_id": "alphabet-worker",
            "processing_ms": 24.5,
            "state": "completed",
        },
    )

    assert status == ServiceStatus(
        coordinator_online=True,
        redis_online=True,
        alphabet_workers_online=2,
        phrase_workers_online=1,
        last_request_id="request-1",
        last_worker_id="alphabet-worker",
        last_processing_ms=24.5,
        last_task_status="completed",
    )


def test_status_parser_falls_back_to_workers_endpoint() -> None:
    status = parse_service_status(
        {"redis": True},
        {
            "workers": [
                {"worker_type": "alphabet", "status": "online"},
                {"worker_type": "alphabet", "status": "offline"},
                {"worker_type": "phrase", "status": "online"},
            ]
        },
        {"redis_connected": True},
    )

    assert status.alphabet_workers_online == 1
    assert status.phrase_workers_online == 1


class Response:
    def __init__(self, payload, status=200):
        self.payload = payload
        self.status = status

    def raise_for_status(self):
        if self.status >= 400:
            raise requests.HTTPError(str(self.status))

    def json(self):
        return self.payload


class Session:
    def __init__(self, responses):
        self.responses = responses
        self.urls = []

    def get(self, url, timeout):
        self.urls.append(url)
        response = self.responses[url]
        if isinstance(response, Exception):
            raise response
        return response


def test_fetcher_uses_health_workers_metrics_and_task_apis() -> None:
    base = "http://coordinator"
    session = Session({
        base + "/health": Response({"redis": True}),
        base + "/workers": Response({"workers": []}),
        base + "/metrics/summary": Response({
            "redis_connected": True,
            "workers": {"by_type": {"alphabet": {"online": 1},
                                      "phrase": {"online": 0}}},
        }),
        base + "/tasks/request-1": Response({
            "request_id": "request-1", "worker_id": "worker-a",
            "processing_ms": 12, "state": "completed",
        }),
    })

    status = fetch_service_status(base, "request-1", session=session)

    assert session.urls == [
        base + "/health", base + "/workers", base + "/metrics/summary",
        base + "/tasks/request-1",
    ]
    assert status.coordinator_online is True
    assert status.last_task_status == "completed"


def test_coordinator_downtime_returns_offline_snapshot() -> None:
    base = "http://coordinator"
    session = Session({base + "/health": requests.ConnectionError("offline")})

    status = fetch_service_status(base, "request-1", session=session)

    assert status.coordinator_online is False
    assert status.redis_online is False
    assert status.last_request_id == "request-1"
    assert session.urls == [base + "/health"]


def test_poller_is_idempotent_and_fetches_off_main_thread() -> None:
    output = queue.Queue(maxsize=1)
    calls = []
    fetched = threading.Event()

    def fetcher(api_url, request_id):
        calls.append((threading.current_thread().name, api_url, request_id))
        fetched.set()
        return ServiceStatus(coordinator_online=True, last_request_id=request_id)

    poller = StatusPoller(
        output, api_url="http://coordinator", interval_seconds=10, fetcher=fetcher
    )
    poller.set_last_request_id("request-1")
    poller.start()
    first_thread = poller._thread
    poller.start()

    assert fetched.wait(1)
    assert poller._thread is first_thread
    assert calls[0][0] == "service-status-poller"
    assert calls[0][0] != threading.current_thread().name
    assert output.get(timeout=1).last_request_id == "request-1"
    poller.stop()
    assert poller._thread is None
