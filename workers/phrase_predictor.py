from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np


# =========================================================
# PATHS
# =========================================================

BASE = Path(__file__).resolve().parents[1]

MODEL_PATH = Path(
    os.getenv(
        "PHRASE_MODEL",
        str(
            BASE
            / "models"
            / "phrase_sequence.keras"
        ),
    )
)

LABEL_PATH = (
    BASE
    / "models"
    / "phrase_labels.json"
)

NORMALIZATION_PATH = (
    BASE
    / "models"
    / "phrase_normalization.npz"
)

CONFIG_PATH = (
    BASE
    / "models"
    / "phrase_model_config.json"
)


# =========================================================
# DEFAULT MODEL CONFIGURATION
# =========================================================

DEFAULT_SEQUENCE_LENGTH = 30

# 21 landmarks × 3 coordinates × 2 hands
DEFAULT_FEATURE_COUNT = 126


# =========================================================
# PHRASE PREDICTOR
# =========================================================

class PhrasePredictor:

    def __init__(
        self,
        threshold: float = 0.72,
    ) -> None:

        self.threshold = float(
            threshold
        )

        self.model = None

        self.labels: list[str] = []

        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

        self.sequence_length = (
            DEFAULT_SEQUENCE_LENGTH
        )

        self.feature_count = (
            DEFAULT_FEATURE_COUNT
        )

        # -------------------------------------------------
        # Load configuration first
        # -------------------------------------------------

        self._load_config()

        # -------------------------------------------------
        # Load labels
        # -------------------------------------------------

        self._load_labels()

        # -------------------------------------------------
        # Load normalization
        # -------------------------------------------------

        self._load_normalization()

        # -------------------------------------------------
        # Load trained model
        # -------------------------------------------------

        self._load_model()

        # -------------------------------------------------
        # Validate everything
        # -------------------------------------------------

        self._validate_loaded_files()


    # =====================================================
    # LOAD CONFIG
    # =====================================================

    def _load_config(
        self,
    ) -> None:

        if not CONFIG_PATH.exists():
            return

        try:

            config = json.loads(
                CONFIG_PATH.read_text(
                    encoding="utf-8"
                )
            )

            self.sequence_length = int(
                config.get(
                    "sequence_length",
                    DEFAULT_SEQUENCE_LENGTH,
                )
            )

            self.feature_count = int(
                config.get(
                    "feature_count",
                    DEFAULT_FEATURE_COUNT,
                )
            )

        except Exception as error:

            print(
                "Warning: could not load "
                "phrase model configuration:"
            )

            print(
                error
            )


    # =====================================================
    # LOAD LABELS
    # =====================================================

    def _load_labels(
        self,
    ) -> None:

        if not LABEL_PATH.exists():
            return

        try:

            labels = json.loads(
                LABEL_PATH.read_text(
                    encoding="utf-8"
                )
            )

            if isinstance(
                labels,
                list,
            ):

                self.labels = [
                    str(label)
                    for label in labels
                ]

        except Exception as error:

            print(
                "Warning: could not load "
                "phrase labels:"
            )

            print(
                error
            )


    # =====================================================
    # LOAD NORMALIZATION
    # =====================================================

    def _load_normalization(
        self,
    ) -> None:

        if not NORMALIZATION_PATH.exists():
            return

        try:

            normalization = np.load(
                NORMALIZATION_PATH
            )

            mean = np.asarray(
                normalization["mean"],
                dtype=np.float32,
            )

            std = np.asarray(
                normalization["std"],
                dtype=np.float32,
            )

            # -------------------------------------------------
            # IMPORTANT FIX
            #
            # Training saved mean/std using keepdims=True.
            # Their shape may therefore be:
            #
            #     (1, 1, 126)
            #
            # Live prediction needs:
            #
            #     (126,)
            #
            # Otherwise NumPy broadcasting produces
            # (1, 30, 126) before batching and the final
            # model input becomes (1, 1, 30, 126).
            # -------------------------------------------------

            mean = np.squeeze(
                mean
            )

            std = np.squeeze(
                std
            )

            expected_shape = (
                self.feature_count,
            )

            if mean.shape != expected_shape:

                raise ValueError(
                    "Invalid normalization mean "
                    f"shape {mean.shape}. "
                    f"Expected {expected_shape}."
                )

            if std.shape != expected_shape:

                raise ValueError(
                    "Invalid normalization std "
                    f"shape {std.shape}. "
                    f"Expected {expected_shape}."
                )

            # Avoid division by zero
            std = np.where(
                std < 1e-6,
                1.0,
                std,
            ).astype(
                np.float32
            )

            self.mean = mean.astype(
                np.float32
            )

            self.std = std

        except Exception as error:

            print(
                "Warning: could not load "
                "phrase normalization:"
            )

            print(
                error
            )

            self.mean = None
            self.std = None


    # =====================================================
    # LOAD MODEL
    # =====================================================

    def _load_model(
        self,
    ) -> None:

        if not MODEL_PATH.exists():
            return

        # Phrase worker uses CPU by default.
        os.environ.setdefault(
            "CUDA_VISIBLE_DEVICES",
            "-1",
        )

        try:

            from tensorflow import keras

            self.model = (
                keras.models.load_model(
                    MODEL_PATH,
                    compile=False,
                )
            )

        except Exception as error:

            print(
                "Warning: could not load "
                "phrase model:"
            )

            print(
                error
            )

            self.model = None


    # =====================================================
    # VALIDATE MODEL
    # =====================================================

    def _validate_loaded_files(
        self,
    ) -> None:

        if self.model is None:
            return

        expected_input_shape = (
            None,
            self.sequence_length,
            self.feature_count,
        )

        actual_input_shape = tuple(
            self.model.input_shape
        )

        if (
            actual_input_shape
            != expected_input_shape
        ):

            raise RuntimeError(
                "Phrase model input shape mismatch. "
                f"Expected {expected_input_shape}, "
                f"got {actual_input_shape}."
            )

        if self.labels:

            expected_classes = len(
                self.labels
            )

            model_classes = int(
                self.model.output_shape[-1]
            )

            if (
                model_classes
                != expected_classes
            ):

                raise RuntimeError(
                    "Phrase model output count "
                    "does not match labels. "
                    f"Model={model_classes}, "
                    f"labels={expected_classes}."
                )


    # =====================================================
    # READY
    # =====================================================

    @property
    def ready(
        self,
    ) -> bool:

        return (
            self.model is not None
            and len(self.labels) > 0
            and self.mean is not None
            and self.std is not None
        )


    # =====================================================
    # EXPECTED SHAPE
    # =====================================================

    @property
    def expected_shape(
        self,
    ) -> tuple[int, int]:

        return (
            self.sequence_length,
            self.feature_count,
        )


    # =====================================================
    # NORMALIZE
    # =====================================================

    def _normalize(
        self,
        sequence: np.ndarray,
    ) -> np.ndarray:

        if (
            self.mean is None
            or self.std is None
        ):

            raise RuntimeError(
                "Phrase normalization is unavailable."
            )

        normalized = (
            sequence
            - self.mean
        ) / self.std

        normalized = np.asarray(
            normalized,
            dtype=np.float32,
        )

        # Safety check
        if (
            normalized.shape
            != self.expected_shape
        ):

            raise RuntimeError(
                "Normalized phrase sequence "
                "has incorrect shape. "
                f"Expected {self.expected_shape}, "
                f"got {normalized.shape}."
            )

        return normalized


    # =====================================================
    # PREDICT
    # =====================================================

    def predict(
        self,
        sequence,
    ):

        # -------------------------------------------------
        # Check model availability
        # -------------------------------------------------

        if self.model is None:

            return {
                "status": "model_missing",
                "phrase": None,
                "confidence": None,
                "message": (
                    "Phrase model is unavailable. "
                    "Run: "
                    "python training/train_phrases.py"
                ),
            }

        if not self.labels:

            return {
                "status": "model_missing",
                "phrase": None,
                "confidence": None,
                "message": (
                    "Phrase labels are unavailable."
                ),
            }

        if (
            self.mean is None
            or self.std is None
        ):

            return {
                "status": "model_missing",
                "phrase": None,
                "confidence": None,
                "message": (
                    "Phrase normalization file "
                    "is unavailable."
                ),
            }

        # -------------------------------------------------
        # Convert incoming sequence
        # -------------------------------------------------

        x = np.asarray(
            sequence,
            dtype=np.float32,
        )

        if (
            x.shape
            != self.expected_shape
        ):

            raise ValueError(
                "Expected phrase sequence shape "
                f"{self.expected_shape}, "
                f"received {x.shape}."
            )

        # -------------------------------------------------
        # Apply SAME normalization used for training
        # -------------------------------------------------

        x = self._normalize(
            x
        )

        # -------------------------------------------------
        # Add exactly ONE batch dimension
        #
        # Before:
        #     (30, 126)
        #
        # After:
        #     (1, 30, 126)
        # -------------------------------------------------

        model_input = np.expand_dims(
            x,
            axis=0,
        )

        expected_model_input = (
            1,
            self.sequence_length,
            self.feature_count,
        )

        if (
            model_input.shape
            != expected_model_input
        ):

            raise RuntimeError(
                "Phrase model input preparation "
                "failed. "
                f"Expected {expected_model_input}, "
                f"got {model_input.shape}."
            )

        # -------------------------------------------------
        # Inference
        # -------------------------------------------------

        started = time.perf_counter()

        probabilities = (
            self.model.predict(
                model_input,
                verbose=0,
            )
        )

        inference_ms = (
            time.perf_counter()
            - started
        ) * 1000.0

        probabilities = np.asarray(
            probabilities,
            dtype=np.float32,
        )

        # Model output should be:
        # (1, 12)
        if probabilities.ndim != 2:

            raise RuntimeError(
                "Unexpected phrase model "
                "output shape: "
                f"{probabilities.shape}"
            )

        probabilities = probabilities[
            0
        ]

        # -------------------------------------------------
        # Best prediction
        # -------------------------------------------------

        index = int(
            np.argmax(
                probabilities
            )
        )

        confidence = float(
            probabilities[
                index
            ]
        )

        predicted_label = (
            self.labels[
                index
            ]
        )

        # -------------------------------------------------
        # Threshold filtering
        # -------------------------------------------------

        if (
            confidence
            >= self.threshold
        ):

            phrase = (
                predicted_label
            )

            status = "predicted"

        else:

            phrase = None

            status = "uncertain"

        # -------------------------------------------------
        # Top-three classes
        # -------------------------------------------------

        top_indices = np.argsort(
            probabilities
        )[-3:][::-1]

        top_predictions = []

        for top_index in top_indices:

            top_index = int(
                top_index
            )

            top_predictions.append(
                {
                    "phrase":
                        self.labels[
                            top_index
                        ],

                    "confidence":
                        float(
                            probabilities[
                                top_index
                            ]
                        ),
                }
            )

        # -------------------------------------------------
        # Return distributed worker-friendly result
        # -------------------------------------------------

        return {
            "status": status,

            "phrase": phrase,

            # Always expose what the model thought was best,
            # even when it falls below the threshold.
            "predicted_label":
                predicted_label,

            "confidence":
                confidence,

            "inference_ms":
                inference_ms,

            "threshold":
                self.threshold,

            "top_predictions":
                top_predictions,
        }