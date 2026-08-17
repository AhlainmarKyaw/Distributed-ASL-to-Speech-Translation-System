"""Health probe used by coordinator and Celery worker containers."""

from __future__ import annotations

import argparse
import json
import socket
import sys
from urllib.request import urlopen


def check_coordinator() -> bool:
    with urlopen("http://127.0.0.1:8000/health", timeout=4) as response:
        payload = json.load(response)
    return response.status == 200 and payload.get("redis") is True


def check_worker(worker_type: str) -> bool:
    from workers.celery_app import celery_app

    node_name = f"{worker_type}@{socket.gethostname()}"
    replies = celery_app.control.inspect(
        destination=[node_name], timeout=4
    ).ping()
    return bool(replies and node_name in replies)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=("coordinator", "worker"))
    parser.add_argument("worker_type", nargs="?", choices=("alphabet", "phrase"))
    args = parser.parse_args()
    try:
        if args.role == "coordinator":
            healthy = check_coordinator()
        else:
            healthy = bool(args.worker_type and check_worker(args.worker_type))
    except Exception:
        healthy = False
    return 0 if healthy else 1


if __name__ == "__main__":
    sys.exit(main())
