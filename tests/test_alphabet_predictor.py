"""Phase 5 tests that never open a webcam or use fake production output."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from scripts import test_alphabet_model as live_script
from workers.alphabet_predictor import (
    FEATURE_DIMENSION,
    LABELS,
    AlphabetPredictor,
    PredictionStabilizer,
)


class InferenceBoundary:
    """Controlled unit-test boundary; production uses the real Keras model."""

    input_shape = (None, 30, 63)
    output_shape = (None, 26)

    def __init__(self, index: int = 0, confidence: float = 0.9) -> None:
        self.index = index
        self.confidence = confidence
        self.calls: list[tuple[np.ndarray, bool]] = []

    def __call__(self, batch, *, training):
        self.calls.append((np.asarray(batch), training))
        output = np.zeros((1, 26), dtype=np.float32)
        output[0, self.index] = self.confidence
        return output


@pytest.fixture
def model_path(tmp_path):
    path = tmp_path / "model.h5"
    path.touch()
    return path


def build_predictor(model_path, model=None, **kwargs):
    return AlphabetPredictor(model=model or InferenceBoundary(), model_path=model_path, **kwargs)


def test_labels_are_ordered_a_to_z() -> None:
    assert LABELS == tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    assert "J" in LABELS and "Z" in LABELS


def test_buffers_until_30_frames(model_path) -> None:
    model = InferenceBoundary()
    predictor = build_predictor(model_path, model)
    for expected in range(1, 30):
        result = predictor.add_frame(np.zeros(63))
        assert result.status == "buffering"
        assert result.buffer_size == expected
    assert model.calls == []


def test_inference_receives_valid_30_by_63_sequence(model_path) -> None:
    model = InferenceBoundary(index=25)
    predictor = build_predictor(model_path, model)
    for value in range(30):
        result = predictor.add_frame(np.full(63, value))
    batch, training = model.calls[0]
    assert batch.shape == (1, 30, 63)
    assert training is False
    assert result.letter == "Z"


def test_rolling_buffer_discards_oldest_frame(model_path) -> None:
    predictor = build_predictor(model_path)
    for value in range(31):
        predictor.add_frame(np.full(63, value))
    assert predictor.sequence_array.shape == (30, 63)
    assert np.all(predictor.sequence_array[0] == 1)
    assert np.all(predictor.sequence_array[-1] == 30)


@pytest.mark.parametrize("bad", [[1, 2], np.zeros((1, 63)), ["x"] * 63])
def test_rejects_malformed_features(model_path, bad) -> None:
    with pytest.raises(ValueError):
        build_predictor(model_path).add_frame(bad)


@pytest.mark.parametrize("bad_value", [np.nan, np.inf, -np.inf])
def test_rejects_non_finite_features(model_path, bad_value) -> None:
    vector = np.zeros(FEATURE_DIMENSION)
    vector[0] = bad_value
    with pytest.raises(ValueError):
        build_predictor(model_path).add_frame(vector)


def test_none_uses_missing_hand_zero_vector(model_path) -> None:
    predictor = build_predictor(model_path)
    predictor.add_frame(None)
    assert np.array_equal(predictor.sequence_array[0], np.zeros(63, dtype=np.float32))


def test_confidence_threshold_returns_uncertain(model_path) -> None:
    predictor = build_predictor(
        model_path, InferenceBoundary(index=1, confidence=0.49), confidence_threshold=0.5
    )
    for _ in range(30):
        result = predictor.add_frame(None)
    assert result.status == "uncertain"
    assert result.letter is None
    assert result.confidence == pytest.approx(0.49)


@pytest.mark.parametrize(
    "input_shape,output_shape",
    [((None, 29, 63), (None, 26)), ((None, 30, 63), (None, 25))],
)
def test_rejects_invalid_model_contract(model_path, input_shape, output_shape) -> None:
    model = InferenceBoundary()
    model.input_shape = input_shape
    model.output_shape = output_shape
    with pytest.raises(ValueError):
        build_predictor(model_path, model)


def test_stability_requires_consecutive_agreement() -> None:
    stable = PredictionStabilizer(3)
    assert stable.update("A") == (None, None)
    assert stable.update("B") == (None, None)
    assert stable.update("B") == (None, None)
    assert stable.update("B") == ("B", "B")


def test_held_letter_is_not_emitted_twice() -> None:
    stable = PredictionStabilizer(2)
    assert stable.update("C") == (None, None)
    assert stable.update("C") == ("C", "C")
    assert stable.update("C") == ("C", None)
    stable.update(None)
    stable.update("C")
    assert stable.update("C") == ("C", "C")


def test_reset_clears_buffer(model_path) -> None:
    predictor = build_predictor(model_path)
    predictor.add_frame(None)
    predictor.reset()
    assert predictor.buffer_size == 0


def test_model_loader_is_called_once(model_path) -> None:
    calls = []

    def loader(path):
        calls.append(path)
        return InferenceBoundary()

    predictor = AlphabetPredictor(model_loader=loader, model_path=model_path)
    for _ in range(35):
        predictor.add_frame(None)
    assert calls == [model_path.resolve()]


def test_phase5_code_never_trains_model() -> None:
    source = inspect.getsource(__import__("workers.alphabet_predictor", fromlist=["*"]))
    assert ".fit(" not in source
    assert ".train_on_batch(" not in source
    assert "training=False" in source


def test_importing_and_unit_testing_do_not_open_webcam(monkeypatch) -> None:
    import cv2

    monkeypatch.setattr(cv2, "VideoCapture", lambda *_: pytest.fail("webcam opened"))
    assert live_script.main([]) == 0


def test_live_extraction_uses_anatomical_right_and_mirrors_x() -> None:
    class Point:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z

    class Hand:
        landmark = [Point(0.25, 0.5, -0.1) for _ in range(21)]

    class Results:
        right_hand_landmarks = Hand()
        left_hand_landmarks = None

    features = live_script.extract_right_hand(Results())
    assert features.shape == (63,)
    assert features[0] == pytest.approx(0.75)
    assert features[1] == pytest.approx(0.5)
    assert features[2] == pytest.approx(-0.1)


def test_left_hand_is_never_used_as_alphabet_features() -> None:
    class Results:
        right_hand_landmarks = None
        left_hand_landmarks = object()

    assert live_script.extract_right_hand(Results()) is None
