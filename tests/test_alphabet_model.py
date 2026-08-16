"""Integration tests for the real bundled alphabet TensorFlow model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "alphabet_landmarks.keras"
LABEL_PATH = ROOT / "models" / "alphabet_labels.json"


@pytest.fixture(scope="module")
def real_model():
    pytest.importorskip("tensorflow", reason="TensorFlow is required for model integration tests")
    if not MODEL_PATH.is_file():
        pytest.skip("bundled alphabet model is unavailable")
    from tensorflow import keras

    model = keras.models.load_model(MODEL_PATH, compile=False)
    yield model
    keras.backend.clear_session()


def test_bundled_alphabet_labels_are_exactly_a_to_z() -> None:
    assert LABEL_PATH.is_file()
    labels = json.loads(LABEL_PATH.read_text(encoding="utf-8"))
    assert labels == list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    assert len(set(labels)) == 26


def test_real_model_has_static_alphabet_contract(real_model) -> None:
    assert tuple(real_model.input_shape) == (None, 63)
    assert tuple(real_model.output_shape) == (None, 26)


def test_real_model_returns_26_finite_outputs(real_model) -> None:
    output = np.asarray(real_model(np.zeros((1, 63), dtype=np.float32), training=False))

    assert output.shape == (1, 26)
    assert np.isfinite(output).all()
