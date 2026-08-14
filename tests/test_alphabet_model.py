"""Phase 4 integration tests for the real bundled SignSpell model."""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import inspect
import re

import pytest

from scripts import test_alphabet_model as audit_module


@pytest.fixture(scope="module")
def audit():
    """Load and shape-test the real model once for this test module."""
    return audit_module.collect_audit()


def test_distribution_is_installed() -> None:
    assert importlib.metadata.version("signspell") == "2.0.1"


def test_actual_module_imports() -> None:
    assert importlib.import_module("asl_alphabet").__name__ == "asl_alphabet"


def test_module_version_mismatch_is_recorded(audit) -> None:
    assert audit.version == "2.0.1"
    assert audit.module_version == "0.1.0"


def test_documented_import_mismatch_is_detected() -> None:
    assert importlib.util.find_spec("signspell") is None
    assert importlib.util.find_spec("asl_alphabet") is not None


def test_bundled_model_exists(audit) -> None:
    assert audit.model_path.is_file()
    assert audit.model_size_bytes > 0


def test_model_checksum_is_sha256(audit) -> None:
    assert re.fullmatch(r"[0-9a-f]{64}", audit.model_sha256)


def test_exactly_26_labels(audit) -> None:
    assert len(audit.labels) == 26


def test_labels_are_ordered_a_to_z(audit) -> None:
    assert audit.labels == tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def test_labels_have_no_duplicates(audit) -> None:
    assert len(set(audit.labels)) == len(audit.labels)


def test_j_exists(audit) -> None:
    assert "J" in audit.labels


def test_z_exists(audit) -> None:
    assert "Z" in audit.labels


def test_sequence_length_is_30(audit) -> None:
    assert audit.sequence_length == 30


def test_feature_dimension_is_63(audit) -> None:
    assert audit.feature_dimension == 63
    assert audit.landmarks_per_frame == 21
    assert audit.coordinates_per_landmark == 3


def test_model_input_shape(audit) -> None:
    assert audit.input_shape == (None, 30, 63)


def test_model_output_shape(audit) -> None:
    assert audit.output_shape == (None, 26)


def test_synthetic_shape_output_is_finite(audit) -> None:
    assert audit.synthetic_output_size == 26
    assert audit.synthetic_output_finite


def test_audit_performs_no_training() -> None:
    source = inspect.getsource(audit_module)
    assert ".fit(" not in source
    assert ".train_on_batch(" not in source
    assert "training=False" in source


def test_audit_never_opens_webcam() -> None:
    source = inspect.getsource(audit_module)
    assert "VideoCapture" not in source
