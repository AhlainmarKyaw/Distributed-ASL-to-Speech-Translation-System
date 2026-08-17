"""Tests for the finite worker failure/recovery demonstration."""

from __future__ import annotations

import argparse
import json

import pytest

from scripts.test_failure_recovery import (
    AttemptResult,
    RequestResult,
    export_summary,
    non_negative_float,
    observe_worker_changes,
    positive_float,
    run_request,
    summarize,
)


@pytest.mark.parametrize("value", ["nan", "inf", "-1"])
def test_duration_options_reject_non_finite_or_negative_values(value) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        non_negative_float(value)


@pytest.mark.parametrize("value", ["nan", "inf", "0", "-1"])
def test_timeout_must_be_positive_and_finite(value) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        positive_float(value)


def attempt(
    number: int,
    *,
    success: bool,
    status_code: int | None,
    worker_id: str | None = None,
    timed_out: bool = False,
) -> AttemptResult:
    return AttemptResult(
        attempt=number,
        request_id=f"request-{number}",
        success=success,
        status_code=status_code,
        latency_ms=number * 10.0,
        worker_id=worker_id,
        task_status="completed" if success else ("timed_out" if timed_out else "failed"),
        error_code=None if success else ("TASK_TIMEOUT" if timed_out else "ERROR"),
        timed_out=timed_out,
    )


def test_transient_failure_is_retried_then_succeeds(capsys) -> None:
    supplied = [
        attempt(1, success=False, status_code=503),
        attempt(2, success=True, status_code=200, worker_id="alphabet-b"),
    ]
    sleeps = []

    def sender(_session, _url, _attempt_number, _timeout):
        return supplied.pop(0)

    result = run_request(
        number=3,
        session=object(),
        api_url="http://localhost:8000",
        timeout_seconds=5,
        max_retries=2,
        retry_delay_seconds=0.25,
        sender=sender,
        sleeper=sleeps.append,
    )

    assert result.success is True
    assert len(result.attempts) == 2
    assert sleeps == [0.25]
    output = capsys.readouterr().out
    assert "[RETRY]" in output
    assert "worker=alphabet-b" in output


def test_permanent_failure_is_not_retried() -> None:
    calls = []

    def sender(*_args):
        calls.append(True)
        return attempt(1, success=False, status_code=400)

    result = run_request(
        number=1,
        session=object(),
        api_url="http://localhost:8000",
        timeout_seconds=5,
        max_retries=2,
        retry_delay_seconds=0,
        sender=sender,
        sleeper=lambda _seconds: None,
    )

    assert result.success is False
    assert len(calls) == 1


def test_worker_transitions_show_failure_and_recovery() -> None:
    stopped = observe_worker_changes(
        {"worker-a": "online", "worker-b": "online"},
        {"worker-a": "offline", "worker-b": "online"},
    )
    recovered = observe_worker_changes(
        {"worker-a": "offline", "worker-b": "online"},
        {"worker-a": "online", "worker-b": "online"},
    )

    assert stopped == ["worker worker-a changed online -> offline: is offline"]
    assert recovered == [
        "worker worker-a changed offline -> online: recovered and registered again"
    ]


def test_summary_counts_retries_timeouts_and_worker_distribution(tmp_path) -> None:
    timed_out = attempt(1, success=False, status_code=504, timed_out=True)
    recovered = attempt(2, success=True, status_code=200, worker_id="worker-b")
    direct = attempt(1, success=True, status_code=200, worker_id="worker-a")
    results = [
        RequestResult(1, True, (timed_out, recovered)),
        RequestResult(2, True, (direct,)),
    ]

    summary = summarize(results, ["worker worker-a recovered"], 12.3456)

    assert summary["controlled_request_count"] == 2
    assert summary["successful_requests"] == 2
    assert summary["failed_requests"] == 0
    assert summary["total_attempts"] == 3
    assert summary["client_retries"] == 1
    assert summary["timeouts_observed"] == 1
    assert summary["mean_success_latency_ms"] == 15.0
    assert summary["successful_request_distribution"] == {
        "worker-a": 1,
        "worker-b": 1,
    }

    output = tmp_path / "summary.json"
    export_summary(output, summary, results)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["summary"]["duration_seconds"] == 12.346
    assert payload["requests"][0]["attempt_count"] == 2
    assert payload["requests"][0]["final_worker_id"] == "worker-b"
