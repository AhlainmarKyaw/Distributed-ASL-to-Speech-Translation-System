"""Strict, shared input validation for API and worker boundaries."""

from __future__ import annotations

import math
from collections.abc import Sequence


ALPHABET_FEATURE_COUNT = 63
PHRASE_FRAME_COUNT = 30
PHRASE_FEATURE_COUNT = 63
MAX_PHRASE_TEXT_LENGTH = 1000


def _finite_number(value, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must contain only numeric values")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must contain only finite values")
    return number


def validate_alphabet_features(features) -> list[float]:
    if isinstance(features, (str, bytes)) or not isinstance(features, Sequence):
        raise ValueError("features must be a list of 63 numeric values")
    if len(features) != ALPHABET_FEATURE_COUNT:
        raise ValueError("features must contain exactly 63 values")
    return [_finite_number(value, "features") for value in features]


def validate_phrase_sequence(sequence) -> list[list[float]]:
    if isinstance(sequence, (str, bytes)) or not isinstance(sequence, Sequence):
        raise ValueError("sequence must be a list of 30 frames")
    if len(sequence) != PHRASE_FRAME_COUNT:
        raise ValueError("sequence must contain exactly 30 frames")
    validated = []
    for frame in sequence:
        if isinstance(frame, (str, bytes)) or not isinstance(frame, Sequence):
            raise ValueError("each phrase frame must be a list of 63 numeric values")
        if len(frame) != PHRASE_FEATURE_COUNT:
            raise ValueError("each phrase frame must contain exactly 63 values")
        validated.append([_finite_number(value, "sequence") for value in frame])
    return validated


def validate_phrase_text(text) -> str:
    if not isinstance(text, str):
        raise ValueError("text must be a string")
    value = text.strip()
    if not value:
        raise ValueError("text is required")
    if len(value) > MAX_PHRASE_TEXT_LENGTH:
        raise ValueError("text must be 1000 characters or fewer")
    return value
