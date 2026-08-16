"""Shared test classification: unmarked tests are isolated unit tests."""

from __future__ import annotations

import pytest


def pytest_collection_modifyitems(items) -> None:
    for item in items:
        if "integration" not in item.keywords and "unit" not in item.keywords:
            item.add_marker(pytest.mark.unit)

