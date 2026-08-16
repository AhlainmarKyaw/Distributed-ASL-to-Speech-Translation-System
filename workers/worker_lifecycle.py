"""Idempotent Celery worker registration and heartbeat lifecycle."""

from __future__ import annotations

import logging
import socket
import threading
from typing import Iterable

from celery.signals import worker_ready, worker_shutdown

from config import settings
from coordinator.worker_registry import (
    RegistryUnavailable,
    WorkerRegistry,
    build_worker_id,
)
from observability import get_logger, log_event


logger = get_logger("worker_lifecycle")


def worker_type_for_queue(queue_name: str) -> str | None:
    if queue_name == settings.alphabet_queue:
        return "alphabet"
    if queue_name == settings.phrase_queue:
        return "phrase"
    return None


def _queue_names(sender) -> list[str]:
    queues: Iterable = getattr(getattr(sender, "task_consumer", None), "queues", ())
    return [str(getattr(queue, "name", queue)) for queue in queues]


class WorkerLifecycleManager:
    def __init__(self, registry: WorkerRegistry | None = None) -> None:
        self.registry = registry or WorkerRegistry()
        self.worker_id: str | None = None
        self.worker_type: str | None = None
        self.hostname: str | None = None
        self.queue_name: str | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self, *, worker_type: str, hostname: str, queue_name: str) -> str:
        worker_id = build_worker_id(worker_type, hostname, queue_name)
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return self.worker_id or worker_id
            self.worker_id = worker_id
            self.worker_type = worker_type
            self.hostname = hostname
            self.queue_name = queue_name
            self._stop_event = threading.Event()
            self._register_safely()
            self._thread = threading.Thread(
                target=self._heartbeat_loop,
                name=f"heartbeat-{worker_id}",
                daemon=True,
            )
            self._thread.start()
        return worker_id

    def _register_safely(self) -> None:
        if not all((self.worker_id, self.worker_type, self.hostname, self.queue_name)):
            return
        try:
            self.registry.register(
                self.worker_id, self.worker_type, self.hostname, self.queue_name
            )
            log_event(
                logger, "worker_registered", component="worker_lifecycle",
                worker_id=self.worker_id, success=True,
            )
        except RegistryUnavailable:
            log_event(
                logger, "worker_registration_failed", component="worker_lifecycle",
                worker_id=self.worker_id, success=False, level=logging.WARNING,
            )

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.wait(settings.worker_heartbeat_seconds):
            if self.worker_id is None:
                continue
            try:
                if not self.registry.heartbeat(self.worker_id):
                    self._register_safely()
            except RegistryUnavailable:
                log_event(
                    logger, "worker_heartbeat_failed", component="worker_lifecycle",
                    worker_id=self.worker_id, success=False, level=logging.WARNING,
                )

    def stop(self) -> None:
        with self._lock:
            thread = self._thread
            self._stop_event.set()
            if self.worker_id:
                try:
                    self.registry.mark_offline(self.worker_id)
                except RegistryUnavailable:
                    pass
            self._thread = None
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=min(settings.worker_heartbeat_seconds + 1, 5))

    def record_task(self, *, success: bool, processing_ms: float) -> None:
        if self.worker_id is None:
            return
        try:
            self.registry.record_task(
                self.worker_id, success=success, processing_ms=processing_ms
            )
        except RegistryUnavailable:
            log_event(
                logger, "worker_metrics_update_failed", component="worker_lifecycle",
                worker_id=self.worker_id, processing_ms=processing_ms,
                success=False, level=logging.WARNING,
            )


manager = WorkerLifecycleManager()


def task_worker_id(worker_type: str, queue_name: str, hostname: str) -> str:
    """Return the registered ID, or a deterministic fallback during eager tests."""
    if manager.worker_id and manager.worker_type == worker_type:
        return manager.worker_id
    return build_worker_id(worker_type, hostname, queue_name)


@worker_ready.connect
def register_worker(sender=None, **_kwargs) -> None:
    hostname = str(getattr(sender, "hostname", None) or socket.gethostname())
    for queue_name in _queue_names(sender):
        worker_type = worker_type_for_queue(queue_name)
        if worker_type:
            manager.start(
                worker_type=worker_type, hostname=hostname, queue_name=queue_name
            )
            return
    log_event(
        logger, "worker_registration_skipped", component="worker_lifecycle",
        success=False, level=logging.WARNING,
    )


@worker_shutdown.connect
def unregister_worker(**_kwargs) -> None:
    manager.stop()
