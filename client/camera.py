from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np


class Camera:
    """
    Webcam + MediaPipe hand tracking.

    Produces two feature formats:

    Alphabet:
        63 values
        21 landmarks × (x, y, z)

    Phrase:
        126 values
        Left hand  = 63 values
        Right hand = 63 values
    """

    def __init__(self, index: int = 0) -> None:
        self.index = index
        self.capture = None

        self.mp_hands = mp.solutions.hands

        # Phrase recognition requires up to TWO hands.
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.drawer = mp.solutions.drawing_utils
        self.connections = self.mp_hands.HAND_CONNECTIONS

    def start(self) -> None:
        self.capture = cv2.VideoCapture(self.index)

        if not self.capture.isOpened():
            raise RuntimeError(
                f"Unable to open webcam index {self.index}"
            )

    @staticmethod
    def _hand_to_array(hand_landmarks) -> np.ndarray:
        """
        Convert one MediaPipe hand into 63 values.

        21 landmarks × x,y,z = 63.
        """
        values = []

        for landmark in hand_landmarks.landmark:
            values.extend(
                [
                    landmark.x,
                    landmark.y,
                    landmark.z,
                ]
            )

        return np.asarray(values, dtype=np.float32)

    @staticmethod
    def _normalize_alphabet(hand_landmarks) -> list[float]:
        """
        Preserve the original alphabet feature format.

        The alphabet model uses wrist-relative,
        scale-normalized XYZ landmarks.
        """
        pts = np.array(
            [
                [p.x, p.y, p.z]
                for p in hand_landmarks.landmark
            ],
            dtype=np.float32,
        )

        # Wrist-relative coordinates
        pts -= pts[0]

        # Normalize scale using XY distance
        scale = np.max(
            np.linalg.norm(
                pts[:, :2],
                axis=1,
            )
        )

        if scale > 1e-6:
            pts /= scale

        return (
            pts.reshape(-1)
            .astype(float)
            .tolist()
        )

    def _extract_phrase_features(self, result) -> list[float] | None:
        """
        Build the exact phrase feature representation
        used by collect_phrase_sequences.py:

            [left hand 63][right hand 63]

        Missing hands are represented by zeros.
        """

        if not result.multi_hand_landmarks:
            return None

        left_hand = np.zeros(
            63,
            dtype=np.float32,
        )

        right_hand = np.zeros(
            63,
            dtype=np.float32,
        )

        if (
            result.multi_hand_landmarks
            and result.multi_handedness
        ):
            for landmarks, handedness in zip(
                result.multi_hand_landmarks,
                result.multi_handedness,
            ):
                label = (
                    handedness
                    .classification[0]
                    .label
                    .lower()
                )

                vector = self._hand_to_array(
                    landmarks
                )

                if label == "left":
                    left_hand = vector

                elif label == "right":
                    right_hand = vector

        features = np.concatenate(
            [
                left_hand,
                right_hand,
            ]
        )

        return (
            features
            .astype(float)
            .tolist()
        )

    def read(self):
        """
        Returns:

            success,
            frame,
            alphabet_features,
            phrase_features

        alphabet_features:
            63 normalized values or None

        phrase_features:
            126 values or None
        """

        if self.capture is None:
            return False, None, None, None

        ok, frame = self.capture.read()

        if not ok:
            return False, None, None, None

        # Same orientation used during phrase collection
        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        result = self.hands.process(rgb)

        alphabet_features = None
        phrase_features = None

        if result.multi_hand_landmarks:

            # Draw every detected hand
            for hand in result.multi_hand_landmarks:
                self.drawer.draw_landmarks(
                    frame,
                    hand,
                    self.connections,
                )

            # -----------------------------------------
            # Alphabet features
            # -----------------------------------------
            #
            # Preserve the existing alphabet behavior:
            # use the first detected hand.
            #
            alphabet_features = (
                self._normalize_alphabet(
                    result.multi_hand_landmarks[0]
                )
            )

            # -----------------------------------------
            # Phrase features
            # -----------------------------------------

            phrase_features = (
                self._extract_phrase_features(
                    result
                )
            )

        return (
            True,
            frame,
            alphabet_features,
            phrase_features,
        )

    def stop(self) -> None:

        if self.capture is not None:
            self.capture.release()
            self.capture = None

        self.hands.close()