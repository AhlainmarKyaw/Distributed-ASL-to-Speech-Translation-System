"""Headless distributed-service status fetching and polling."""

from __future__ import annotations

import threading
import queue as queue_module
from dataclasses import dataclass
from typing import Any, Callable

import requests

from config import settings


@dataclass(frozen=True)
class ServiceStatus:
    coordinator_online: bool = False
    redis_online: bool = False
    alphabet_workers_online: int = 0
    phrase_workers_online: int = 0
    last_request_id: str | None = None
    last_worker_id: str | None = None
    last_processing_ms: float | None = None
    last_task_status: str | None = None


def _online_worker_counts(
    worker_payload: dict[str, Any], metrics_payload: dict[str, Any]
) -> tuple[int, int]:
    grouped = metrics_payload.get("workers", {}).get("by_type", {})
    if grouped:
        return (
            int(grouped.get("alphabet", {}).get("online", 0)),
            int(grouped.get("phrase", {}).get("online", 0)),
        )
    workers = worker_payload.get("workers", [])
    return (
        sum(1 for worker in workers if worker.get("worker_type") == "alphabet"
            and worker.get("status") == "online"),
        sum(1 for worker in workers if worker.get("worker_type") == "phrase"
            and worker.get("status") == "online"),
    )


def parse_service_status(
    health_payload: dict[str, Any],
    worker_payload: dict[str, Any],
    metrics_payload: dict[str, Any],
    task_payload: dict[str, Any] | None = None,
    *,
    last_request_id: str | None = None,
) -> ServiceStatus:
    alphabet, phrase = _online_worker_counts(worker_payload, metrics_payload)
    task = task_payload or {}
    return ServiceStatus(
        coordinator_online=True,
        redis_online=bool(health_payload.get("redis"))
        and bool(metrics_payload.get("redis_connected", True)),
        alphabet_workers_online=alphabet,
        phrase_workers_online=phrase,
        last_request_id=task.get("request_id") or last_request_id,
        last_worker_id=task.get("worker_id"),
        last_processing_ms=task.get("processing_ms"),
        last_task_status=task.get("state"),
    )


def _get_json(session, url: str) -> dict[str, Any]:
    response = session.get(url, timeout=settings.health_timeout_seconds)
    response.raise_for_status()
    return response.json()


def fetch_service_status(
    api_url: str,
    last_request_id: str | None = None,
    *,
    session=requests,
) -> ServiceStatus:
    """Fetch all status APIs; coordinator failure returns an offline snapshot."""
    try:
        health = _get_json(session, api_url + "/health")
    except requests.RequestException:
        return ServiceStatus(last_request_id=last_request_id)

    try:
        workers = _get_json(session, api_url + "/workers")
    except requests.RequestException:
        workers = {"workers": []}
    try:
        metrics = _get_json(session, api_url + "/metrics/summary")
    except requests.RequestException:
        metrics = {"redis_connected": False, "workers": {"by_type": {}}}

    task = None
    if last_request_id:
        try:
            task = _get_json(session, api_url + f"/tasks/{last_request_id}")
        except requests.RequestException:
            task = None
    return parse_service_status(
        health, workers, metrics, task, last_request_id=last_request_id
    )


class StatusPoller:
    """A single idempotent background poller with a bounded output queue."""

    def __init__(
        self,
        output_queue,
        *,
        api_url: str,
        interval_seconds: float = 3.0,
        fetcher: Callable[[str, str | None], ServiceStatus] = fetch_service_status,
    ) -> None:
        self.output_queue = output_queue
        self.api_url = api_url
        self.interval_seconds = interval_seconds
        self.fetcher = fetcher
        self._last_request_id: str | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def set_last_request_id(self, request_id: str | None) -> None:
        with self._lock:
            self._last_request_id = request_id

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop = threading.Event()
            self._thread = threading.Thread(
                target=self._run, name="service-status-poller", daemon=True
            )
            self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                request_id = self._last_request_id
            status = self.fetcher(self.api_url, request_id)
            # The UI only needs the newest snapshot; discard one stale item.
            try:
                self.output_queue.put_nowait(status)
            except queue_module.Full:
                try:
                    self.output_queue.get_nowait()
                except queue_module.Empty:
                    pass
                try:
                    self.output_queue.put_nowait(status)
                except queue_module.Full:
                    pass
            self._stop.wait(self.interval_seconds)

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=settings.health_timeout_seconds + 0.5)
        if thread is None or not thread.is_alive():
            self._thread = None
