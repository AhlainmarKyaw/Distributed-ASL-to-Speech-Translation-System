"""Automated Phase 3 tests that never access a physical webcam."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts.test_webcam import (
    _handedness_text,
    build_parser,
    run_webcam,
)


class FakeFrame:
    """Minimal frame object with the writeable flag used by the script."""

    def __init__(self) -> None:
        self.flags = SimpleNamespace(writeable=True)


def fake_cv2(*, opened: bool, read_result: tuple[bool, object | None], key: int = 0):
    """Return an OpenCV-like mock configured for one loop iteration."""
    capture = Mock()
    capture.isOpened.return_value = opened
    capture.read.return_value = read_result

    module = Mock()
    module.VideoCapture.return_value = capture
    module.COLOR_BGR2RGB = 1
    module.FONT_HERSHEY_SIMPLEX = 2
    module.LINE_AA = 3
    module.flip.side_effect = lambda frame, _axis: frame
    module.cvtColor.side_effect = lambda frame, _code: frame
    module.waitKey.return_value = key
    return module, capture


def fake_mediapipe(results: object):
    """Return a MediaPipe-like mock whose Hands instance yields results."""
    hands_instance = Mock()
    hands_instance.process.return_value = results
    hands_api = Mock()
    hands_api.Hands.return_value = hands_instance
    hands_api.HAND_CONNECTIONS = "connections"
    drawing = Mock()
    module = SimpleNamespace(
        solutions=SimpleNamespace(hands=hands_api, drawing_utils=drawing)
    )
    return module, hands_instance, drawing


def test_parser_rejects_negative_camera_index() -> None:
    """A negative OpenCV camera index is rejected before capture starts."""
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--camera", "-1"])


def test_unavailable_camera_returns_error_and_releases_resources(capsys) -> None:
    """An unavailable camera produces a useful error and still cleans up."""
    cv2, capture = fake_cv2(opened=False, read_result=(False, None))
    media_pipe, hands, _drawing = fake_mediapipe(SimpleNamespace())

    status = run_webcam(4, cv2_module=cv2, mediapipe_module=media_pipe)

    assert status == 2
    assert "camera index 4" in capsys.readouterr().err
    hands.close.assert_not_called()
    capture.release.assert_called_once_with()
    cv2.destroyAllWindows.assert_called_once_with()


def test_no_frame_returns_error_and_closes_hands(capsys) -> None:
    """Capture failure after opening closes both MediaPipe and OpenCV."""
    cv2, capture = fake_cv2(opened=True, read_result=(False, None))
    media_pipe, hands, _drawing = fake_mediapipe(SimpleNamespace())

    status = run_webcam(0, cv2_module=cv2, mediapipe_module=media_pipe)

    assert status == 3
    assert "no frame" in capsys.readouterr().err
    hands.close.assert_called_once_with()
    capture.release.assert_called_once_with()
    cv2.destroyAllWindows.assert_called_once_with()


def test_no_hand_status_and_q_exit() -> None:
    """No-hand frames show a clear status and Q exits normally."""
    frame = FakeFrame()
    cv2, capture = fake_cv2(opened=True, read_result=(True, frame), key=ord("q"))
    results = SimpleNamespace(multi_hand_landmarks=None, multi_handedness=None)
    media_pipe, hands, drawing = fake_mediapipe(results)
    times = iter((1.0, 1.1))

    status = run_webcam(
        0,
        cv2_module=cv2,
        mediapipe_module=media_pipe,
        clock=lambda: next(times),
    )

    assert status == 0
    displayed_text = [call.args[1] for call in cv2.putText.call_args_list]
    assert "NO HAND DETECTED" in displayed_text
    assert "FPS: 10.0" in displayed_text
    drawing.draw_landmarks.assert_not_called()
    hands.close.assert_called_once_with()
    capture.release.assert_called_once_with()


def test_landmarks_and_handedness_are_displayed() -> None:
    """Detected landmarks are drawn and their handedness is shown."""
    frame = FakeFrame()
    cv2, _capture = fake_cv2(opened=True, read_result=(True, frame), key=ord("Q"))
    classification = SimpleNamespace(label="Right", score=0.987)
    results = SimpleNamespace(
        multi_hand_landmarks=["landmarks"],
        multi_handedness=[SimpleNamespace(classification=[classification])],
    )
    media_pipe, _hands, drawing = fake_mediapipe(results)
    times = iter((5.0, 5.2))

    status = run_webcam(
        1,
        cv2_module=cv2,
        mediapipe_module=media_pipe,
        clock=lambda: next(times),
    )

    assert status == 0
    cv2.flip.assert_called_once_with(frame, 1)
    drawing.draw_landmarks.assert_called_once_with(
        frame,
        "landmarks",
        "connections",
    )
    displayed_text = [call.args[1] for call in cv2.putText.call_args_list]
    assert "HAND: RIGHT (0.99)" in displayed_text


def test_raw_camera_mode_does_not_mirror_frame() -> None:
    """The opt-out mode preserves raw camera coordinates when requested."""
    frame = FakeFrame()
    cv2, _capture = fake_cv2(opened=True, read_result=(True, frame), key=ord("q"))
    results = SimpleNamespace(multi_hand_landmarks=None, multi_handedness=None)
    media_pipe, _hands, _drawing = fake_mediapipe(results)
    times = iter((1.0, 1.1))

    status = run_webcam(
        0,
        cv2_module=cv2,
        mediapipe_module=media_pipe,
        clock=lambda: next(times),
        mirror=False,
    )

    assert status == 0
    cv2.flip.assert_not_called()


def test_handedness_falls_back_when_classification_is_missing() -> None:
    """Landmarks without classification still receive a readable status."""
    results = SimpleNamespace(multi_handedness=None)

    assert _handedness_text(results) == "HAND DETECTED"


def test_unmirrored_handedness_is_corrected() -> None:
    """Raw camera coordinates require swapping MediaPipe's selfie labels."""
    classification = SimpleNamespace(label="Left", score=0.75)
    results = SimpleNamespace(
        multi_handedness=[SimpleNamespace(classification=[classification])]
    )

    assert _handedness_text(results, mirrored=False) == "HAND: RIGHT (0.75)"


def test_keyboard_interrupt_exits_without_traceback_and_cleans_up() -> None:
    """Ctrl+C is a supported fallback when the video window is not focused."""
    frame = FakeFrame()
    cv2, capture = fake_cv2(opened=True, read_result=(True, frame))
    results = SimpleNamespace(multi_hand_landmarks=None, multi_handedness=None)
    media_pipe, hands, _drawing = fake_mediapipe(results)
    hands.process.side_effect = KeyboardInterrupt

    status = run_webcam(0, cv2_module=cv2, mediapipe_module=media_pipe)

    assert status == 130
    hands.close.assert_called_once_with()
    capture.release.assert_called_once_with()
    cv2.destroyAllWindows.assert_called_once_with()
