from pathlib import Path
import json

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras


# =========================================================
# PATHS
# =========================================================

DATASET_ROOT = Path("datasets/phrases")

MODEL_PATH = Path("models/phrase_sequence.keras")
LABEL_PATH = Path("models/phrase_labels.json")
NORMALIZATION_PATH = Path("models/phrase_normalization.npz")
CONFIG_PATH = Path("models/phrase_model_config.json")


# =========================================================
# CONFIGURATION
# =========================================================

SEQUENCE_LENGTH = 30
FEATURE_COUNT = 126

TEST_SIZE = 0.20
RANDOM_STATE = 42


# =========================================================
# BASIC CHECKS
# =========================================================

print()
print("=" * 70)
print("BALANCED PHRASE MODEL TEST")
print("=" * 70)


required_files = [
    MODEL_PATH,
    LABEL_PATH,
    NORMALIZATION_PATH,
]


for file_path in required_files:

    if not file_path.exists():

        raise SystemExit(
            f"Missing required file: {file_path}"
        )


if not DATASET_ROOT.exists():

    raise SystemExit(
        f"Missing phrase dataset: {DATASET_ROOT}"
    )


# =========================================================
# LOAD SAVED LABELS
# =========================================================

with open(
    LABEL_PATH,
    "r",
    encoding="utf-8",
) as file:

    saved_labels = json.load(file)


print()
print("Classes:")

for index, label in enumerate(
    saved_labels
):

    print(
        f"{index:2} -> {label}"
    )


# =========================================================
# LOAD MODEL
# =========================================================

print()
print("Loading model...")

model = keras.models.load_model(
    MODEL_PATH
)

print(
    "Model loaded successfully."
)

print(
    "Input shape :",
    model.input_shape,
)

print(
    "Output shape:",
    model.output_shape,
)


# =========================================================
# VERIFY MODEL SHAPE
# =========================================================

expected_shape = (
    None,
    SEQUENCE_LENGTH,
    FEATURE_COUNT,
)


if model.input_shape != expected_shape:

    raise SystemExit(
        f"Unexpected model input shape.\n"
        f"Expected: {expected_shape}\n"
        f"Found   : {model.input_shape}"
    )


if model.output_shape[-1] != len(
    saved_labels
):

    raise SystemExit(
        "Model output size does not match labels."
    )


# =========================================================
# LOAD NORMALIZATION
# =========================================================

print()
print(
    "Loading normalization values..."
)


normalization = np.load(
    NORMALIZATION_PATH
)


mean = normalization["mean"]

std = normalization["std"]


print(
    "Normalization mean shape:",
    mean.shape,
)

print(
    "Normalization std shape :",
    std.shape,
)


# =========================================================
# LOAD CONFIG
# =========================================================

if CONFIG_PATH.exists():

    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as file:

        config = json.load(file)

    print()
    print("Saved configuration:")

    print(
        "Dataset:",
        config.get("dataset"),
    )

    print(
        "Total samples:",
        config.get("total_samples"),
    )

    print(
        "Saved test accuracy:",
        (
            f"{config.get('test_accuracy', 0) * 100:.2f}%"
        ),
    )


# =========================================================
# LOAD DATASET
# =========================================================

print()
print("=" * 70)
print("LOADING DATASET")
print("=" * 70)


X = []
y = []


for class_name in saved_labels:

    folder = (
        DATASET_ROOT
        / class_name
    )

    if not folder.exists():

        raise SystemExit(
            f"Missing class folder: {folder}"
        )

    files = sorted(
        folder.glob("*.npy")
    )

    valid = 0
    invalid = 0

    for file_path in files:

        try:

            sequence = np.load(
                file_path
            )

        except Exception as error:

            print(
                f"Cannot load "
                f"{file_path.name}: "
                f"{error}"
            )

            invalid += 1
            continue

        if sequence.shape != (
            SEQUENCE_LENGTH,
            FEATURE_COUNT,
        ):

            print(
                f"Wrong shape skipped: "
                f"{file_path.name} "
                f"{sequence.shape}"
            )

            invalid += 1
            continue

        X.append(
            sequence.astype(
                np.float32
            )
        )

        y.append(
            class_name
        )

        valid += 1

    print(
        f"{class_name:12} "
        f"valid={valid:3} "
        f"invalid={invalid:3}"
    )


X = np.asarray(
    X,
    dtype=np.float32,
)


# =========================================================
# ENCODE LABELS
# =========================================================

encoder = LabelEncoder()

y_encoded = encoder.fit_transform(
    y
)


if encoder.classes_.tolist() != saved_labels:

    raise SystemExit(
        "Dataset labels do not match "
        "the saved phrase labels."
    )


print()
print(
    "Dataset shape:",
    X.shape,
)

print(
    "Total samples:",
    len(X),
)


# =========================================================
# RECREATE SAME TEST SPLIT
# =========================================================

_, X_test, _, y_test = (
    train_test_split(
        X,
        y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )
)


print(
    "Held-out test samples:",
    len(X_test),
)


# =========================================================
# APPLY SAME NORMALIZATION AS TRAINING
# =========================================================

X_test_normalized = (
    X_test - mean
) / std


# =========================================================
# EVALUATE
# =========================================================

print()
print("=" * 70)
print("MODEL EVALUATION")
print("=" * 70)


test_loss, test_accuracy = (
    model.evaluate(
        X_test_normalized,
        y_test,
        verbose=0,
    )
)


print(
    f"Test loss     : "
    f"{test_loss:.4f}"
)

print(
    f"Test accuracy : "
    f"{test_accuracy * 100:.2f}%"
)


# =========================================================
# PREDICTIONS
# =========================================================

probabilities = model.predict(
    X_test_normalized,
    verbose=0,
)


predicted_indices = np.argmax(
    probabilities,
    axis=1,
)


print()
print("=" * 70)
print("INDIVIDUAL PREDICTIONS")
print("=" * 70)


correct = 0


for index in range(
    len(X_test)
):

    actual_index = int(
        y_test[index]
    )

    predicted_index = int(
        predicted_indices[index]
    )

    actual_label = (
        saved_labels[
            actual_index
        ]
    )

    predicted_label = (
        saved_labels[
            predicted_index
        ]
    )

    confidence = float(
        np.max(
            probabilities[index]
        )
    )

    if (
        actual_index
        == predicted_index
    ):

        status = "CORRECT"
        correct += 1

    else:

        status = "WRONG"

    print(
        f"{status:7} | "
        f"Actual: {actual_label:10} | "
        f"Predicted: {predicted_label:10} | "
        f"Confidence: "
        f"{confidence * 100:6.2f}%"
    )


# =========================================================
# FINAL RESULT
# =========================================================

accuracy = (
    correct
    / len(X_test)
) * 100


print()
print("=" * 70)
print("FINAL RESULT")
print("=" * 70)

print(
    f"Correct predictions : "
    f"{correct}/{len(X_test)}"
)

print(
    f"Accuracy            : "
    f"{accuracy:.2f}%"
)

print()
print(
    "Balanced phrase model test complete."
)

print()