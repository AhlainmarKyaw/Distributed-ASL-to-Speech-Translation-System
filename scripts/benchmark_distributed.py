"""Reproducible concurrent benchmark for distributed alphabet prediction."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence
from uuid import uuid4

import requests


@dataclass(frozen=True)
class RequestResult:
    index: int
    success: bool
    status_code: int | None
    latency_ms: float
    request_id: str
    worker_id: str | None = None
    prediction_status: str | None = None
    error_code: str | None = None


_thread_local = threading.local()


def synthetic_alphabet_features() -> list[float]:
    """Return one deterministic, finite 21x3 normalized landmark vector."""
    features = [0.0, 0.0, 0.0]
    for landmark in range(1, 21):
        features.extend(
            [
                round((landmark % 5) / 5.0, 6),
                round((landmark // 5) / 5.0, 6),
                round(-landmark / 100.0, 6),
            ]
        )
    return features


def _session() -> requests.Session:
    session = getattr(_thread_local, "session", None)
    if session is None:
        session = requests.Session()
        _thread_local.session = session
    return session


def send_prediction(
    index: int,
    api_url: str,
    features: list[float],
    timeout_seconds: float,
) -> RequestResult:
    request_id = str(uuid4())
    started = time.perf_counter()
    try:
        response = _session().post(
            api_url.rstrip("/") + "/predict/alphabet",
            json={"features": features, "client_request_id": request_id},
            timeout=timeout_seconds,
        )
        latency_ms = (time.perf_counter() - started) * 1000
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        success = 200 <= response.status_code < 300
        return RequestResult(
            index=index,
            success=success,
            status_code=response.status_code,
            latency_ms=latency_ms,
            request_id=payload.get("request_id", request_id),
            worker_id=payload.get("worker_id"),
            prediction_status=payload.get("status"),
            error_code=None if success else payload.get("error_code", "HTTP_ERROR"),
        )
    except requests.RequestException as exc:
        return RequestResult(
            index=index,
            success=False,
            status_code=None,
            latency_ms=(time.perf_counter() - started) * 1000,
            request_id=request_id,
            error_code=type(exc).__name__,
        )


def percentile(values: Sequence[float], percentage: float) -> float:
    """Calculate a linearly interpolated percentile without third-party libraries."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * percentage / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


def calculate_metrics(
    results: Sequence[RequestResult], total_execution_seconds: float
) -> dict:
    latencies = [result.latency_ms for result in results]
    success_count = sum(result.success for result in results)
    failure_count = len(results) - success_count
    distribution = Counter(
        result.worker_id for result in results if result.worker_id is not None
    )
    return {
        "total_requests": len(results),
        "total_execution_seconds": total_execution_seconds,
        "throughput_requests_per_second": (
            len(results) / total_execution_seconds if total_execution_seconds > 0 else 0.0
        ),
        "mean_latency_ms": statistics.fmean(latencies) if latencies else 0.0,
        "median_latency_ms": statistics.median(latencies) if latencies else 0.0,
        "p95_latency_ms": percentile(latencies, 95),
        "p99_latency_ms": percentile(latencies, 99),
        "success_count": success_count,
        "failure_count": failure_count,
        "requests_without_worker_id": sum(
            result.worker_id is None for result in results
        ),
        "worker_distribution": dict(sorted(distribution.items())),
    }


def run_benchmark(
    *,
    api_url: str,
    total_requests: int,
    concurrency: int,
    warmup_count: int,
    timeout_seconds: float,
    sender: Callable[[int, str, list[float], float], RequestResult] = send_prediction,
) -> tuple[dict, list[RequestResult]]:
    features = synthetic_alphabet_features()

    for index in range(warmup_count):
        sender(-(index + 1), api_url, features, timeout_seconds)

    started = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(sender, index, api_url, features, timeout_seconds)
            for index in range(total_requests)
        ]
        for future in as_completed(futures):
            results.append(future.result())
    total_seconds = time.perf_counter() - started
    results.sort(key=lambda result: result.index)
    metrics = calculate_metrics(results, total_seconds)
    metrics.update({
        "api_url": api_url.rstrip("/"),
        "concurrency": concurrency,
        "warmup_count": warmup_count,
        "timeout_seconds": timeout_seconds,
        "feature_source": "deterministic_synthetic_63_landmarks",
    })
    return metrics, results


def export_results(path: Path, metrics: dict, results: Sequence[RequestResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".csv":
        distribution = json.dumps(metrics["worker_distribution"], sort_keys=True)
        metric_columns = [key for key in metrics if key != "worker_distribution"]
        request_columns = list(asdict(results[0]).keys()) if results else list(
            RequestResult.__dataclass_fields__
        )
        fieldnames = request_columns + metric_columns + ["worker_distribution"]
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            rows = results or [None]
            for result in rows:
                row = asdict(result) if result is not None else {
                    column: None for column in request_columns
                }
                row.update({key: metrics[key] for key in metric_columns})
                row["worker_distribution"] = distribution
                writer.writerow(row)
        return
    if path.suffix.lower() not in {".json", ""}:
        raise ValueError("Output file must use .json or .csv")
    payload = {
        "summary": metrics,
        "requests": [asdict(result) for result in results],
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


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--total-requests", "--requests", dest="total_requests",
        type=positive_int, default=100,
    )
    parser.add_argument("--concurrency", type=positive_int, default=4)
    parser.add_argument("--warmup-count", "--warmup", dest="warmup_count",
                        type=non_negative_int, default=5)
    parser.add_argument("--output", type=Path, default=Path("benchmark_results.json"))
    parser.add_argument("--timeout-seconds", type=positive_float, default=15.0)
    return parser


def _rounded_summary(metrics: dict) -> dict:
    rounded = dict(metrics)
    for key in (
        "total_execution_seconds", "throughput_requests_per_second",
        "mean_latency_ms", "median_latency_ms", "p95_latency_ms", "p99_latency_ms",
    ):
        rounded[key] = round(float(rounded[key]), 3)
    return rounded


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    metrics, results = run_benchmark(
        api_url=args.api_url,
        total_requests=args.total_requests,
        concurrency=args.concurrency,
        warmup_count=args.warmup_count,
        timeout_seconds=args.timeout_seconds,
    )
    export_results(args.output, metrics, results)
    print(json.dumps(_rounded_summary(metrics), indent=2, sort_keys=True))
    print(f"Results written to {args.output.resolve()}")
    return 0 if metrics["failure_count"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
