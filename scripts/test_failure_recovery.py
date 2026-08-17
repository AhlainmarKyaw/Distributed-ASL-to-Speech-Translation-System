"""Finite, operator-controlled demonstration of worker failure and recovery."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence
from uuid import uuid4

import requests

try:
    from scripts.benchmark_distributed import synthetic_alphabet_features
except ModuleNotFoundError:  # Direct execution: python scripts/test_failure_recovery.py
    from benchmark_distributed import synthetic_alphabet_features


TRANSIENT_HTTP_STATUSES = {502, 503, 504}


@dataclass(frozen=True)
class AttemptResult:
    attempt: int
    request_id: str
    success: bool
    status_code: int | None
    latency_ms: float
    worker_id: str | None
    task_status: str | None
    error_code: str | None
    timed_out: bool


@dataclass(frozen=True)
class RequestResult:
    number: int
    success: bool
    attempts: tuple[AttemptResult, ...]


def safe_json(response: requests.Response) -> dict:
    try:
        payload = response.json()
        return payload if isinstance(payload, dict) else {}
    except ValueError:
        return {}


def fetch_task_status(
    session: requests.Session, api_url: str, request_id: str, timeout_seconds: float
) -> dict:
    try:
        response = session.get(
            f"{api_url.rstrip('/')}/tasks/{request_id}", timeout=timeout_seconds
        )
        return safe_json(response) if response.status_code == 200 else {}
    except requests.RequestException:
        return {}


def fetch_alphabet_workers(
    session: requests.Session, api_url: str, timeout_seconds: float
) -> dict[str, str] | None:
    """Return worker ID to state, or None when the registry cannot be observed."""
    try:
        response = session.get(f"{api_url.rstrip('/')}/workers", timeout=timeout_seconds)
        if response.status_code != 200:
            return None
        payload = safe_json(response)
        if not payload.get("registry_available", False):
            return None
        return {
            worker["worker_id"]: worker.get("status", "unknown")
            for worker in payload.get("workers", [])
            if worker.get("worker_type") == "alphabet" and worker.get("worker_id")
        }
    except requests.RequestException:
        return None


def observe_worker_changes(
    previous: dict[str, str] | None, current: dict[str, str] | None
) -> list[str]:
    if current is None:
        return ["worker registry unavailable"] if previous is not None else []
    if previous is None:
        return [
            f"worker {worker_id} observed {status}"
            for worker_id, status in sorted(current.items())
        ]
    events = []
    for worker_id in sorted(set(previous) | set(current)):
        before = previous.get(worker_id, "not_registered")
        after = current.get(worker_id, "not_registered")
        if before != after:
            action = "recovered and registered again" if after == "online" else f"is {after}"
            events.append(f"worker {worker_id} changed {before} -> {after}: {action}")
    return events


def send_attempt(
    session: requests.Session,
    api_url: str,
    attempt: int,
    timeout_seconds: float,
) -> AttemptResult:
    request_id = str(uuid4())
    started = time.perf_counter()
    try:
        response = session.post(
            f"{api_url.rstrip('/')}/predict/alphabet",
            json={
                "features": synthetic_alphabet_features(),
                "client_request_id": request_id,
            },
            timeout=timeout_seconds,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        payload = safe_json(response)
        request_id = payload.get("request_id", request_id)
        task = fetch_task_status(session, api_url, request_id, timeout_seconds)
        task_status = task.get("state") or payload.get("status")
        worker_id = payload.get("worker_id") or task.get("worker_id")
        error_code = payload.get("error_code") or task.get("error_code")
        timed_out = response.status_code == 504 or task_status == "timed_out"
        return AttemptResult(
            attempt=attempt,
            request_id=request_id,
            success=200 <= response.status_code < 300,
            status_code=response.status_code,
            latency_ms=latency_ms,
            worker_id=worker_id,
            task_status=task_status,
            error_code=error_code,
            timed_out=timed_out,
        )
    except requests.Timeout:
        return AttemptResult(
            attempt, request_id, False, None,
            (time.perf_counter() - started) * 1000,
            None, None, "CLIENT_TIMEOUT", True,
        )
    except requests.RequestException as exc:
        return AttemptResult(
            attempt, request_id, False, None,
            (time.perf_counter() - started) * 1000,
            None, None, type(exc).__name__, False,
        )


def is_transient(attempt: AttemptResult) -> bool:
    return attempt.status_code is None or attempt.status_code in TRANSIENT_HTTP_STATUSES


def run_request(
    *,
    number: int,
    session: requests.Session,
    api_url: str,
    timeout_seconds: float,
    max_retries: int,
    retry_delay_seconds: float,
    sender: Callable[..., AttemptResult] = send_attempt,
    sleeper: Callable[[float], None] = time.sleep,
) -> RequestResult:
    attempts = []
    for attempt_number in range(1, max_retries + 2):
        attempt = sender(session, api_url, attempt_number, timeout_seconds)
        attempts.append(attempt)
        state = attempt.task_status or attempt.error_code or "unknown"
        worker = attempt.worker_id or "unavailable"
        label = "SUCCESS" if attempt.success else "FAILURE"
        print(
            f"[{label}] request={number} attempt={attempt_number} "
            f"request_id={attempt.request_id} worker={worker} state={state} "
            f"http={attempt.status_code} latency_ms={attempt.latency_ms:.1f}"
        )
        if attempt.success or not is_transient(attempt) or attempt_number > max_retries:
            break
        print(
            f"[RETRY] request={number} next_attempt={attempt_number + 1} "
            f"after={retry_delay_seconds:.1f}s reason={state}"
        )
        sleeper(retry_delay_seconds)
    return RequestResult(number, attempts[-1].success, tuple(attempts))


def summarize(
    results: Sequence[RequestResult], worker_events: Sequence[str], duration_seconds: float
) -> dict:
    attempts = [attempt for result in results for attempt in result.attempts]
    workers = Counter(
        attempt.worker_id for attempt in attempts
        if attempt.success and attempt.worker_id is not None
    )
    successful_latencies = [
        attempt.latency_ms for attempt in attempts if attempt.success
    ]
    return {
        "duration_seconds": round(duration_seconds, 3),
        "controlled_request_count": len(results),
        "successful_requests": sum(result.success for result in results),
        "failed_requests": sum(not result.success for result in results),
        "total_attempts": len(attempts),
        "client_retries": len(attempts) - len(results),
        "timeouts_observed": sum(attempt.timed_out for attempt in attempts),
        "mean_success_latency_ms": (
            round(statistics.fmean(successful_latencies), 3)
            if successful_latencies else 0.0
        ),
        "successful_request_distribution": dict(sorted(workers.items())),
        "worker_events": list(worker_events),
    }


def export_summary(path: Path, summary: dict, results: Sequence[RequestResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summary,
        "requests": [
            {
                "number": result.number,
                "success": result.success,
                "attempt_count": len(result.attempts),
                "final_request_id": result.attempts[-1].request_id,
                "final_worker_id": result.attempts[-1].worker_id,
                "final_status": result.attempts[-1].task_status,
            }
            for result in results
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be zero or greater")
    return parsed


def non_negative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("value must be zero or greater")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--request-count", type=positive_int, default=30)
    parser.add_argument("--interval-seconds", type=non_negative_float, default=2.0)
    parser.add_argument("--timeout-seconds", type=positive_float, default=15.0)
    parser.add_argument("--max-retries", type=non_negative_int, default=2)
    parser.add_argument("--retry-delay-seconds", type=non_negative_float, default=1.0)
    parser.add_argument(
        "--output", type=Path, default=Path("failure_recovery_summary.json")
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    session = requests.Session()
    results = []
    worker_events = []
    previous_workers = None
    started = time.perf_counter()
    print(
        f"Starting finite recovery demo: {args.request_count} requests, "
        f"{args.interval_seconds:.1f}s interval, at most {args.max_retries} client retries."
    )
    print("This script does not stop or restart workers. Follow the documented operator steps.")
    try:
        for number in range(1, args.request_count + 1):
            workers = fetch_alphabet_workers(session, args.api_url, args.timeout_seconds)
            events = observe_worker_changes(previous_workers, workers)
            for event in events:
                print(f"[WORKER] {event}")
            worker_events.extend(events)
            if workers is not None:
                previous_workers = workers

            results.append(run_request(
                number=number,
                session=session,
                api_url=args.api_url,
                timeout_seconds=args.timeout_seconds,
                max_retries=args.max_retries,
                retry_delay_seconds=args.retry_delay_seconds,
            ))
            if number < args.request_count:
                time.sleep(args.interval_seconds)
    except KeyboardInterrupt:
        print("\nInterrupted by operator; exporting observations collected so far.")
    finally:
        session.close()

    summary = summarize(results, worker_events, time.perf_counter() - started)
    export_summary(args.output, summary, results)
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Summary written to {args.output.resolve()}")
    return 0 if results and summary["failed_requests"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
