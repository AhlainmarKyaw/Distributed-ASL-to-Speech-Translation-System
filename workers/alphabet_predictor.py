"""TensorFlow alphabet prediction behind an injectable inference boundary."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Callable

import numpy as np


BASE = Path(__file__).resolve().parents[1]
MODEL_PATH = Path(os.getenv("ALPHABET_MODEL", BASE / "models" / "alphabet_landmarks.keras"))
LABEL_PATH = BASE / "models" / "alphabet_labels.json"
FEATURE_DIMENSION = 63
LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _load_keras_model(path: Path):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    from tensorflow import keras

    return keras.models.load_model(path, compile=False)


class AlphabetPredictor:
    def __init__(
        self,
        confidence_threshold: float = 0.70,
        *,
        model=None,
        model_path: str | Path = MODEL_PATH,
        label_path: str | Path = LABEL_PATH,
        model_loader: Callable[[Path], object] = _load_keras_model,
    ) -> None:
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        self.threshold = confidence_threshold
        self.model_path = Path(model_path).resolve()
        self.labels = list(LABELS)
        label_file = Path(label_path)
        if label_file.exists():
            loaded_labels = json.loads(label_file.read_text(encoding="utf-8"))
            if not isinstance(loaded_labels, list) or not loaded_labels:
                raise ValueError("Alphabet labels must be a non-empty JSON list")
            self.labels = [str(label) for label in loaded_labels]
        self.model = model
        if self.model is None and self.model_path.exists():
            self.model = model_loader(self.model_path)

    @property
    def ready(self) -> bool:
        return self.model is not None

    def predict(self, features) -> dict:
        if self.model is None:
            return {
                "status": "model_missing",
                "letter": None,
                "confidence": None,
                "inference_ms": None,
                "message": "Train the alphabet model first: python training/train_alphabet.py",
            }
        try:
            values = np.asarray(features, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError("Expected 63 finite MediaPipe landmark features") from exc
        if values.shape != (FEATURE_DIMENSION,) or not np.isfinite(values).all():
            raise ValueError("Expected 63 finite MediaPipe landmark features")
        started = time.perf_counter()
        probabilities = np.asarray(self.model.predict(values[None, :], verbose=0))[0]
        inference_ms = (time.perf_counter() - started) * 1000
        if probabilities.ndim != 1 or probabilities.size != len(self.labels):
            raise ValueError("Alphabet model output does not match configured labels")
        index = int(np.argmax(probabilities))
        confidence = float(probabilities[index])
        predicted = confidence >= self.threshold
        return {
            "status": "predicted" if predicted else "uncertain",
            "letter": self.labels[index] if predicted else None,
            "confidence": confidence,
            "inference_ms": inference_ms,
        }
