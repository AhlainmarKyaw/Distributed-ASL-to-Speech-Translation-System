"""Redis-backed worker registration, heartbeat, and failure detection."""

from __future__ import annotations

import hashlib
import math
import time
from datetime import datetime, timezone
from typing import Any, Callable

from redis import Redis

from config import settings


REGISTRY_KEY_PREFIX = "distributed_asl:workers:"


class RegistryUnavailable(RuntimeError):
    """Raised when the worker registry cannot communicate with Redis."""


def build_worker_id(worker_type: str, hostname: str, queue_name: str) -> str:
    """Build a stable public identifier without exposing the machine hostname."""
    digest = hashlib.sha256(f"{hostname}|{queue_name}".encode("utf-8")).hexdigest()[:12]
    return f"{worker_type}-{digest}"


def _utc_iso(epoch_seconds: float) -> str:
    return datetime.fromtimestamp(epoch_seconds, timezone.utc).isoformat()


def _decode_mapping(values: dict[Any, Any]) -> dict[str, str]:
    return {
        (key.decode() if isinstance(key, bytes) else str(key)):
        (value.decode() if isinstance(value, bytes) else str(value))
        for key, value in values.items()
    }


class WorkerRegistry:
    """Store expiring worker records and derive online/offline state."""

    def __init__(
        self,
        redis_client: Any | None = None,
        *,
        offline_seconds: float | None = None,
        heartbeat_seconds: float | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.redis = redis_client or Redis.from_url(settings.redis_url)
        self.offline_seconds = offline_seconds or settings.worker_offline_seconds
        heartbeat = heartbeat_seconds or settings.worker_heartbeat_seconds
        self.ttl_seconds = math.ceil(max(self.offline_seconds * 2, heartbeat * 3))
        self.clock = clock

    def _key(self, worker_id: str) -> str:
        return f"{REGISTRY_KEY_PREFIX}{worker_id}"

    def register(
        self, worker_id: str, worker_type: str, hostname: str, queue_name: str
    ) -> dict[str, Any]:
        now = self.clock()
        mapping = {
            "worker_id": worker_id,
            "worker_type": worker_type,
            "hostname": hostname,
            "queue_name": queue_name,
            "status": "online",
            "registered_at": str(now),
            "last_heartbeat": str(now),
            "completed_task_count": "0",
            "failed_task_count": "0",
            "total_processing_ms": "0",
            "average_processing_ms": "0",
        }
        try:
            self.redis.hset(self._key(worker_id), mapping=mapping)
            self.redis.expire(self._key(worker_id), self.ttl_seconds)
        except Exception as exc:
            raise RegistryUnavailable("Worker registry is unavailable") from exc
        return self.get(worker_id) or {}

    def heartbeat(self, worker_id: str) -> bool:
        now = self.clock()
        key = self._key(worker_id)
        try:
            if not self.redis.exists(key):
                return False
            self.redis.hset(
                key, mapping={"last_heartbeat": str(now), "status": "online"}
            )
            self.redis.expire(key, self.ttl_seconds)
            return True
        except Exception as exc:
            raise RegistryUnavailable("Worker registry is unavailable") from exc

    def record_task(self, worker_id: str, *, success: bool, processing_ms: float) -> bool:
        key = self._key(worker_id)
        try:
            if not self.redis.exists(key):
                return False
            counter = "completed_task_count" if success else "failed_task_count"
            self.redis.hincrby(key, counter, 1)
            total = float(self.redis.hincrbyfloat(key, "total_processing_ms", processing_ms))
            values = _decode_mapping(self.redis.hgetall(key))
            task_count = int(values.get("completed_task_count", 0)) + int(
                values.get("failed_task_count", 0)
            )
            average = total / task_count if task_count else 0.0
            self.redis.hset(key, mapping={"average_processing_ms": str(average)})
            self.redis.expire(key, self.ttl_seconds)
            return True
        except Exception as exc:
            raise RegistryUnavailable("Worker registry is unavailable") from exc

    def mark_offline(self, worker_id: str) -> bool:
        key = self._key(worker_id)
        try:
            if not self.redis.exists(key):
                return False
            self.redis.hset(key, mapping={"status": "offline"})
            self.redis.expire(key, self.ttl_seconds)
            return True
        except Exception as exc:
            raise RegistryUnavailable("Worker registry is unavailable") from exc

    def get(self, worker_id: str) -> dict[str, Any] | None:
        try:
            values = _decode_mapping(self.redis.hgetall(self._key(worker_id)))
        except Exception as exc:
            raise RegistryUnavailable("Worker registry is unavailable") from exc
        if not values:
            return None
        return self._public_record(values)

    def list_workers(self) -> list[dict[str, Any]]:
        try:
            keys = list(self.redis.scan_iter(match=f"{REGISTRY_KEY_PREFIX}*"))
            records = []
            for key in keys:
                values = _decode_mapping(self.redis.hgetall(key))
                if values:
                    records.append(self._public_record(values))
            return sorted(records, key=lambda item: item["worker_id"])
        except Exception as exc:
            raise RegistryUnavailable("Worker registry is unavailable") from exc

    def _public_record(self, values: dict[str, str]) -> dict[str, Any]:
        now = self.clock()
        heartbeat = float(values["last_heartbeat"])
        stored_status = values.get("status", "online")
        status = (
            "offline"
            if stored_status == "offline" or now - heartbeat > self.offline_seconds
            else "online"
        )
        hostname_hash = hashlib.sha256(values["hostname"].encode("utf-8")).hexdigest()[:8]
        return {
            "worker_id": values["worker_id"],
            "worker_type": values["worker_type"],
            "hostname": f"host-{hostname_hash}",
            "queue_name": values["queue_name"],
            "status": status,
            "registered_at": _utc_iso(float(values["registered_at"])),
            "last_heartbeat": _utc_iso(heartbeat),
            "completed_task_count": int(values.get("completed_task_count", 0)),
            "failed_task_count": int(values.get("failed_task_count", 0)),
            "average_processing_ms": float(values.get("average_processing_ms", 0)),
        }
