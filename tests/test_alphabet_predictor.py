"""Isolated tests for the static alphabet TensorFlow boundary."""

from __future__ import annotations

import numpy as np
import pytest

from workers.alphabet_predictor import FEATURE_DIMENSION, LABELS, AlphabetPredictor


class FakeModel:
    def __init__(self, index: int = 0, confidence: float = 0.9, output_size: int = 26):
        self.index = index
        self.confidence = confidence
        self.output_size = output_size
        self.calls = []

    def predict(self, batch, *, verbose):
        self.calls.append((np.asarray(batch), verbose))
        output = np.zeros((1, self.output_size), dtype=np.float32)
        output[0, min(self.index, self.output_size - 1)] = self.confidence
        return output


@pytest.fixture
def missing_model_path(tmp_path):
    return tmp_path / "missing.keras"


def predictor(missing_model_path, model=None, **kwargs):
    return AlphabetPredictor(
        model=model,
        model_path=missing_model_path,
        label_path=missing_model_path.with_suffix(".json"),
        **kwargs,
    )


def test_labels_and_feature_contract_are_a_to_z_and_63() -> None:
    assert FEATURE_DIMENSION == 63
    assert LABELS == tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def test_inference_receives_one_by_63_batch(missing_model_path) -> None:
    model = FakeModel(index=25)
    result = predictor(missing_model_path, model).predict([0.0] * 63)

    batch, verbose = model.calls[0]
    assert batch.shape == (1, 63)
    assert batch.dtype == np.float32
    assert verbose == 0
    assert result["status"] == "predicted"
    assert result["letter"] == "Z"
    assert result["confidence"] == pytest.approx(0.9)
    assert result["inference_ms"] >= 0


def test_confidence_below_threshold_is_uncertain(missing_model_path) -> None:
    result = predictor(
        missing_model_path, FakeModel(index=1, confidence=0.49),
        confidence_threshold=0.5,
    ).predict([0.0] * 63)

    assert result["status"] == "uncertain"
    assert result["letter"] is None
    assert result["confidence"] == pytest.approx(0.49)


@pytest.mark.parametrize(
    "features",
    [[0.0] * 62, [0.0] * 64, ["x"] * 63,
     [float("nan")] + [0.0] * 62, [float("inf")] + [0.0] * 62],
)
def test_predictor_rejects_invalid_features(missing_model_path, features) -> None:
    with pytest.raises(ValueError, match="63 finite"):
        predictor(missing_model_path, FakeModel()).predict(features)


def test_missing_model_is_reported_without_importing_tensorflow(missing_model_path) -> None:
    loader_calls = []
    instance = AlphabetPredictor(
        model_path=missing_model_path,
        label_path=missing_model_path.with_suffix(".json"),
        model_loader=lambda path: loader_calls.append(path),
    )

    result = instance.predict([0.0] * 63)

    assert instance.ready is False
    assert loader_calls == []
    assert result["status"] == "model_missing"
    assert result["letter"] is None
    assert result["confidence"] is None


def test_existing_model_is_loaded_once(missing_model_path) -> None:
    model_path = missing_model_path
    model_path.touch()
    model = FakeModel()
    calls = []

    def loader(path):
        calls.append(path)
        return model

    instance = AlphabetPredictor(
        model_path=model_path,
        label_path=model_path.with_suffix(".json"),
        model_loader=loader,
    )
    instance.predict([0.0] * 63)
    instance.predict([0.0] * 63)

    assert calls == [model_path.resolve()]
    assert len(model.calls) == 2


def test_model_output_must_match_labels(missing_model_path) -> None:
    with pytest.raises(ValueError, match="output"):
        predictor(missing_model_path, FakeModel(output_size=25)).predict([0.0] * 63)


@pytest.mark.parametrize("threshold", [-0.1, 1.1])
def test_invalid_confidence_threshold_is_rejected(missing_model_path, threshold) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        predictor(missing_model_path, FakeModel(), confidence_threshold=threshold)

