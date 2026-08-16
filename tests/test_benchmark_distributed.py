"""Tests for the reproducible distributed benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import math

import pytest

from scripts.benchmark_distributed import (
    RequestResult,
    calculate_metrics,
    export_results,
    percentile,
    positive_float,
    run_benchmark,
    synthetic_alphabet_features,
)


@pytest.mark.parametrize("value", ["nan", "inf", "0", "-1"])
def test_timeout_must_be_positive_and_finite(value) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        positive_float(value)


def result(index: int, latency_ms: float, worker_id: str | None, success: bool = True):
    return RequestResult(
        index=index,
        success=success,
        status_code=200 if success else 503,
        latency_ms=latency_ms,
        request_id=f"request-{index}",
        worker_id=worker_id,
        prediction_status="completed" if success else "failed",
        error_code=None if success else "WORKER_UNAVAILABLE",
    )


def test_synthetic_features_are_valid_alphabet_input() -> None:
    features = synthetic_alphabet_features()

    assert len(features) == 63
    assert all(isinstance(value, float) and math.isfinite(value) for value in features)
    assert features == synthetic_alphabet_features()


@pytest.mark.parametrize(
    ("values", "percentage", "expected"),
    [
        ([], 95, 0.0),
        ([7], 99, 7.0),
        ([10, 20, 30, 40], 50, 25.0),
        ([10, 20, 30, 40], 95, 38.5),
        ([10, 20, 30, 40], 99, 39.7),
    ],
)
def test_percentile_uses_linear_interpolation(values, percentage, expected) -> None:
    assert percentile(values, percentage) == pytest.approx(expected)


def test_metric_calculation_and_worker_distribution() -> None:
    results = [
        result(0, 10, "alphabet-a"),
        result(1, 20, "alphabet-b"),
        result(2, 30, "alphabet-a"),
        result(3, 40, None),
        result(4, 50, "alphabet-b", success=False),
    ]

    metrics = calculate_metrics(results, total_execution_seconds=2.0)

    assert metrics["total_requests"] == 5
    assert metrics["total_execution_seconds"] == 2.0
    assert metrics["throughput_requests_per_second"] == 2.5
    assert metrics["mean_latency_ms"] == 30.0
    assert metrics["median_latency_ms"] == 30.0
    assert metrics["p95_latency_ms"] == pytest.approx(48.0)
    assert metrics["p99_latency_ms"] == pytest.approx(49.6)
    assert metrics["success_count"] == 4
    assert metrics["failure_count"] == 1
    assert metrics["worker_distribution"] == {
        "alphabet-a": 2,
        "alphabet-b": 2,
    }
    assert metrics["requests_without_worker_id"] == 1


def test_empty_metrics_are_safe() -> None:
    metrics = calculate_metrics([], total_execution_seconds=0)

    assert metrics["throughput_requests_per_second"] == 0.0
    assert metrics["mean_latency_ms"] == 0.0
    assert metrics["median_latency_ms"] == 0.0
    assert metrics["p95_latency_ms"] == 0.0
    assert metrics["p99_latency_ms"] == 0.0
    assert metrics["worker_distribution"] == {}
    assert metrics["requests_without_worker_id"] == 0


def test_run_benchmark_excludes_warmups_and_records_workers() -> None:
    calls = []

    def sender(index, api_url, features, timeout_seconds):
        calls.append((index, api_url, len(features), timeout_seconds))
        return result(index, 5 + max(index, 0), f"alphabet-{max(index, 0) % 2}")

    metrics, results = run_benchmark(
        api_url="http://localhost:8000/",
        total_requests=4,
        concurrency=2,
        warmup_count=2,
        timeout_seconds=3,
        sender=sender,
    )

    assert len(calls) == 6
    assert sorted(call[0] for call in calls[:2]) == [-2, -1]
    assert [item.index for item in results] == [0, 1, 2, 3]
    assert metrics["total_requests"] == 4
    assert metrics["success_count"] == 4
    assert metrics["worker_distribution"] == {"alphabet-0": 2, "alphabet-1": 2}
    assert metrics["api_url"] == "http://localhost:8000"
    assert metrics["warmup_count"] == 2
    assert all(call[2] == 63 for call in calls)


def test_json_and_csv_exports(tmp_path) -> None:
    results = [result(0, 12.5, "alphabet-a")]
    metrics = calculate_metrics(results, 0.5)

    json_path = tmp_path / "results.json"
    export_results(json_path, metrics, results)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["summary"]["throughput_requests_per_second"] == 2.0
    assert payload["requests"][0]["worker_id"] == "alphabet-a"

    csv_path = tmp_path / "results.csv"
    export_results(csv_path, metrics, results)
    with csv_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["worker_id"] == "alphabet-a"
    assert float(rows[0]["throughput_requests_per_second"]) == 2.0
    assert json.loads(rows[0]["worker_distribution"]) == {"alphabet-a": 1}


def test_export_rejects_unsupported_format(tmp_path) -> None:
    with pytest.raises(ValueError, match=".json or .csv"):
        export_results(tmp_path / "results.txt", calculate_metrics([], 0), [])
