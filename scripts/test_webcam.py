"""Validate local OpenCV webcam capture and MediaPipe Hands landmarks.

This script is intentionally limited to input-pipeline validation. It does not
save frames or landmarks, load an ASL model, or perform recognition.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable, Sequence
from typing import Any

WINDOW_TITLE = "Phase 3 - Webcam + MediaPipe Hands (Q to quit)"


def non_negative_camera_index(value: str) -> int:
    """Parse a camera index and reject values OpenCV cannot use."""
    try:
        index = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("camera index must be an integer") from exc
    if index < 0:
        raise argparse.ArgumentTypeError("camera index must be zero or greater")
    return index


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(
        description="Validate webcam capture and draw MediaPipe hand landmarks."
    )
    parser.add_argument(
        "--camera",
        type=non_negative_camera_index,
        default=0,
        help="OpenCV camera index (default: 0)",
    )
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="Show the raw camera view instead of the default selfie mirror view.",
    )
    return parser


def _handedness_text(results: Any, *, mirrored: bool = True) -> str:
    """Return a readable handedness label from MediaPipe results."""
    handedness = getattr(results, "multi_handedness", None) or []
    if not handedness:
        return "HAND DETECTED"

    classification = handedness[0].classification[0]
    label = classification.label
    if not mirrored:
        label = {"Left": "Right", "Right": "Left"}.get(label, label)
    return f"HAND: {label.upper()} ({classification.score:.2f})"


def run_webcam(
    camera_index: int,
    *,
    cv2_module: Any | None = None,
    mediapipe_module: Any | None = None,
    clock: Callable[[], float] = time.perf_counter,
    mirror: bool = True,
) -> int:
    """Run the webcam validation loop and return a process-style status code.

    Status 0 means the user exited normally, 2 means the camera could not be
    opened, and 3 means capture opened but did not provide a frame.
    """
    if camera_index < 0:
        raise ValueError("camera_index must be zero or greater")

    if cv2_module is None:
        import cv2 as cv2_module
    if mediapipe_module is None:
        import mediapipe as mediapipe_module

    capture = cv2_module.VideoCapture(camera_index)
    hands = None

    try:
        if not capture.isOpened():
            print(
                f"ERROR: Could not open camera index {camera_index}. "
                "Check the index and macOS camera permission for VS Code.",
                file=sys.stderr,
            )
            return 2

        hands = mediapipe_module.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        drawing = mediapipe_module.solutions.drawing_utils
        hand_connections = mediapipe_module.solutions.hands.HAND_CONNECTIONS

        previous_time = clock()
        print(f"Camera {camera_index} opened.")
        print(
            "The video window may open behind VS Code. Click the video window, "
            "then press Q to exit. Ctrl+C in the terminal also exits safely."
        )

        try:
            while True:
                frame_ok, frame = capture.read()
                if not frame_ok or frame is None:
                    print(
                        "ERROR: Camera opened but no frame was received.",
                        file=sys.stderr,
                    )
                    return 3

                # MediaPipe handedness assumes a mirrored/selfie image. Mirror
                # before both inference and display so Left/Right matches the
                # user's actual hand rather than the raw camera coordinates.
                if mirror:
                    frame = cv2_module.flip(frame, 1)

                rgb_frame = cv2_module.cvtColor(frame, cv2_module.COLOR_BGR2RGB)
                rgb_frame.flags.writeable = False
                results = hands.process(rgb_frame)

                landmarks = getattr(results, "multi_hand_landmarks", None) or []
                if landmarks:
                    status = _handedness_text(results, mirrored=mirror)
                    status_color = (0, 200, 0)
                    for hand_landmarks in landmarks:
                        drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            hand_connections,
                        )
                else:
                    status = "NO HAND DETECTED"
                    status_color = (0, 165, 255)

                current_time = clock()
                elapsed = current_time - previous_time
                fps = 1.0 / elapsed if elapsed > 0 else 0.0
                previous_time = current_time

                cv2_module.putText(
                    frame,
                    status,
                    (20, 35),
                    cv2_module.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    status_color,
                    2,
                    cv2_module.LINE_AA,
                )
                cv2_module.putText(
                    frame,
                    f"FPS: {fps:.1f}",
                    (20, 70),
                    cv2_module.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                    cv2_module.LINE_AA,
                )
                cv2_module.putText(
                    frame,
                    "Q: quit",
                    (20, 105),
                    cv2_module.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2_module.LINE_AA,
                )

                cv2_module.imshow(WINDOW_TITLE, frame)
                key = cv2_module.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    print("Webcam validation closed normally.")
                    return 0
        except KeyboardInterrupt:
            print("Webcam validation interrupted; resources were released safely.")
            return 130
    finally:
        if hands is not None:
            hands.close()
        capture.release()
        cv2_module.destroyAllWindows()


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and run the Phase 3 webcam validation."""
    args = build_parser().parse_args(argv)
    return run_webcam(args.camera, mirror=not args.no_mirror)


if __name__ == "__main__":
    raise SystemExit(main())
