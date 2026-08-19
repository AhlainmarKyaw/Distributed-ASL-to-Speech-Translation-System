"""Temporal buffer for dynamic phrase recognition."""

from __future__ import annotations

from collections import deque

import numpy as np


SEQUENCE_LENGTH = 30
FEATURE_COUNT = 126


class PhraseBuffer:
    """
    Collects live phrase frames until one complete
    30-frame sequence is available.

    Each frame must contain exactly 126 values:

        63 values for the left hand
        63 values for the right hand
    """

    def __init__(
        self,
        sequence_length: int = SEQUENCE_LENGTH,
        feature_count: int = FEATURE_COUNT,
    ) -> None:

        self.sequence_length = int(sequence_length)
        self.feature_count = int(feature_count)

        self.frames = deque(
            maxlen=self.sequence_length
        )

    @property
    def size(self) -> int:
        return len(self.frames)

    @property
    def progress(self) -> float:
        """
        Returns collection progress from 0.0 to 1.0.
        """

        if self.sequence_length <= 0:
            return 0.0

        return min(
            1.0,
            len(self.frames)
            / self.sequence_length,
        )

    @property
    def ready(self) -> bool:
        """
        True when exactly 30 frames are available.
        """

        return (
            len(self.frames)
            == self.sequence_length
        )

    def clear(self) -> None:
        """
        Remove all collected phrase frames.
        """

        self.frames.clear()

    def add(
        self,
        frame_features,
    ) -> bool:
        """
        Add one 126-feature phrase frame.

        Returns:
            True  -> buffer now contains 30 frames
            False -> still collecting
        """

        if frame_features is None:
            return False

        frame = np.asarray(
            frame_features,
            dtype=np.float32,
        )

        expected_shape = (
            self.feature_count,
        )

        if frame.shape != expected_shape:
            raise ValueError(
                "Phrase frame must have shape "
                f"{expected_shape}, "
                f"received {frame.shape}."
            )

        if not np.isfinite(frame).all():
            raise ValueError(
                "Phrase frame contains "
                "non-finite values."
            )

        self.frames.append(
            frame.copy()
        )

        return self.ready

    def get_sequence(
        self,
    ) -> list[list[float]] | None:
        """
        Return the current 30 × 126 sequence.

        Returns None until the buffer is complete.
        """

        if not self.ready:
            return None

        sequence = np.stack(
            list(self.frames),
            axis=0,
        )

        expected_shape = (
            self.sequence_length,
            self.feature_count,
        )

        if sequence.shape != expected_shape:
            raise RuntimeError(
                "Phrase buffer produced "
                "an invalid sequence shape. "
                f"Expected {expected_shape}, "
                f"received {sequence.shape}."
            )

        return (
            sequence
            .astype(float)
            .tolist()
        )

    def pop_sequence(
        self,
    ) -> list[list[float]] | None:
        """
        Return one complete phrase sequence
        and immediately clear the buffer.

        This is useful after submitting the
        sequence to the phrase worker.
        """

        sequence = self.get_sequence()

        if sequence is None:
            return None

        self.clear()

        return sequence