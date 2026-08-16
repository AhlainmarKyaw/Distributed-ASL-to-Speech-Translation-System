"""Request tracing and structured JSON logging shared by all components."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from uuid import UUID, uuid4


LOG_FIELDS = (
    "component",
    "request_id",
    "worker_id",
    "task_name",
    "processing_ms",
    "success",
)


class JsonFormatter(logging.Formatter):
    """Format operational events as one safe JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "component": getattr(record, "component", record.name),
            "request_id": getattr(record, "request_id", None),
            "worker_id": getattr(record, "worker_id", None),
            "task_name": getattr(record, "task_name", None),
            "processing_ms": getattr(record, "processing_ms", None),
            "success": getattr(record, "success", None),
            "event": record.getMessage(),
        }
        return json.dumps(payload, separators=(",", ":"), ensure_ascii=False)


def get_logger(component: str) -> logging.Logger:
    """Return a component logger with exactly one JSON stream handler."""
    logger = logging.getLogger(f"distributed_asl.{component}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not any(getattr(handler, "_distributed_asl", False) for handler in logger.handlers):
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        handler._distributed_asl = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    return logger


def log_event(
    logger: logging.Logger,
    event: str,
    *,
    component: str,
    request_id: str | None = None,
    worker_id: str | None = None,
    task_name: str | None = None,
    processing_ms: float | None = None,
    success: bool | None = None,
    level: int = logging.INFO,
) -> None:
    """Write a structured event without request payloads or infrastructure URLs."""
    logger.log(
        level,
        event,
        extra={
            "component": component,
            "request_id": request_id,
            "worker_id": worker_id,
            "task_name": task_name,
            "processing_ms": round(processing_ms, 3) if processing_ms is not None else None,
            "success": success,
        },
    )


def new_request_id() -> str:
    return str(uuid4())


def resolve_request_id(client_request_id: str | None) -> str:
    """Return a canonical UUID, generating one when the client omitted it."""
    if client_request_id is None:
        return new_request_id()
    if not isinstance(client_request_id, str):
        raise ValueError("client_request_id must be a UUID string")
    value = client_request_id.strip()
    if not value or len(value) > 36:
        raise ValueError("client_request_id must be a valid UUID")
    try:
        return str(UUID(value))
    except (ValueError, AttributeError) as exc:
        raise ValueError("client_request_id must be a valid UUID") from exc
