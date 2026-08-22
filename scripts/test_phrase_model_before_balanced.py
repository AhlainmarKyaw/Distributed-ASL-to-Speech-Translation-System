from pathlib import Path
import json

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras


# =========================================================
# CONFIGURATION
# =========================================================

DATASET_ROOT = Path("datasets/phrases")

MODEL_PATH = Path("models/phrase_sequence.keras")
LABEL_PATH = Path("models/phrase_labels.json")
CONFIG_PATH = Path("models/phrase_model_config.json")

SEQUENCE_LENGTH = 30
FEATURE_COUNT = 126
MIN_SAMPLES_PER_CLASS = 4


# =========================================================
# CHECK REQUIRED FILES
# =========================================================

print()
print("=" * 65)
print("DISTRIBUTED ASL - PHRASE MODEL TEST")
print("=" * 65)


if not MODEL_PATH.exists():
    raise SystemExit(
        f"ERROR: Model not found: {MODEL_PATH}"
    )

if not LABEL_PATH.exists():
    raise SystemExit(
        f"ERROR: Labels not found: {LABEL_PATH}"
    )

if not DATASET_ROOT.exists():
    raise SystemExit(
        f"ERROR: Dataset not found: {DATASET_ROOT}"
    )


# =========================================================
# LOAD LABELS
# =========================================================

with open(
    LABEL_PATH,
    "r",
    encoding="utf-8",
) as file:
    saved_labels = json.load(file)


print()
print("Phrase classes:")

for index, label in enumerate(saved_labels):
    print(f"  {index}: {label}")


# =========================================================
# LOAD MODEL
# =========================================================

print()
print("Loading phrase model...")

model = keras.models.load_model(
    MODEL_PATH
)

print("Phrase model loaded successfully.")

print()
print("Model input shape :", model.input_shape)
print("Model output shape:", model.output_shape)


# =========================================================
# VERIFY MODEL SHAPE
# =========================================================

expected_input_shape = (
    None,
    SEQUENCE_LENGTH,
    FEATURE_COUNT,
)


if model.input_shape != expected_input_shape:

    raise SystemExit(
        "\nERROR: Unexpected phrase model input shape.\n"
        f"Expected: {expected_input_shape}\n"
        f"Received: {model.input_shape}"
    )


if model.output_shape[-1] != len(saved_labels):

    raise SystemExit(
        "\nERROR: Model output count does not match "
        "phrase_labels.json."
    )


# =========================================================
# READ CONFIG
# =========================================================

if CONFIG_PATH.exists():

    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        config = json.load(file)

    print()
    print("Model configuration:")
    print(
        "  Sequence length:",
        config.get("sequence_length"),
    )
    print(
        "  Feature count  :",
        config.get("feature_count"),
    )
    print(
        "  Number classes :",
        config.get("number_of_classes"),
    )


# =========================================================
# LOAD VALID DATASET
# =========================================================

print()
print("=" * 65)
print("LOADING TEST DATA")
print("=" * 65)


X = []
y = []


class_folders = sorted(
    [
        folder
        for folder in DATASET_ROOT.iterdir()
        if folder.is_dir()
    ]
)


for folder in class_folders:

    files = sorted(
        folder.glob("*.npy")
    )

    # Must match training behavior
    if len(files) < MIN_SAMPLES_PER_CLASS:

        print(
            f"{folder.name:15} "
            f"SKIPPED - {len(files)} sample(s)"
        )

        continue

    loaded = 0

    for file_path in files:

        try:
            sequence = np.load(
                file_path
            )

        except Exception as error:

            print(
                f"Could not load {file_path.name}: "
                f"{error}"
            )

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

            continue

        X.append(
            sequence.astype(
                np.float32
            )
        )

        y.append(
            folder.name.replace(
                "_",
                " ",
            )
        )

        loaded += 1

    print(
        f"{folder.name:15} "
        f"{loaded} loaded"
    )


if not X:

    raise SystemExit(
        "ERROR: No valid phrase sequences found."
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


dataset_labels = encoder.classes_.tolist()


print()
print("Dataset shape :", X.shape)
print("Dataset labels:", dataset_labels)


# =========================================================
# VERIFY LABEL ORDER
# =========================================================

if dataset_labels != saved_labels:

    print()
    print("ERROR: Model labels and dataset labels differ.")

    print(
        "Model labels  :",
        saved_labels,
    )

    print(
        "Dataset labels:",
        dataset_labels,
    )

    raise SystemExit(
        "Cannot safely evaluate the model."
    )


# =========================================================
# RECREATE SAME TEST SPLIT
# =========================================================

_, X_test, _, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.25,
    random_state=42,
    stratify=y_encoded,
)


print()
print(
    "Held-out test samples:",
    len(X_test),
)


# =========================================================
# EVALUATE MODEL
# =========================================================

print()
print("=" * 65)
print("MODEL EVALUATION")
print("=" * 65)


test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0,
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
# PREDICT TEST SAMPLES
# =========================================================

predictions = model.predict(
    X_test,
    verbose=0,
)


predicted_indices = np.argmax(
    predictions,
    axis=1,
)


print()
print("=" * 65)
print("INDIVIDUAL TEST PREDICTIONS")
print("=" * 65)


correct = 0


for index in range(len(X_test)):

    actual_index = int(
        y_test[index]
    )

    predicted_index = int(
        predicted_indices[index]
    )

    actual_label = saved_labels[
        actual_index
    ]

    predicted_label = saved_labels[
        predicted_index
    ]

    confidence = float(
        np.max(
            predictions[index]
        )
    )

    if actual_index == predicted_index:

        status = "CORRECT"
        correct += 1

    else:

        status = "WRONG"


    print(
        f"{status:7} | "
        f"Actual: {actual_label:10} | "
        f"Predicted: {predicted_label:10} | "
        f"Confidence: {confidence * 100:6.2f}%"
    )


# =========================================================
# FINAL RESULT
# =========================================================

calculated_accuracy = (
    correct / len(X_test)
) * 100


print()
print("=" * 65)
print("FINAL RESULT")
print("=" * 65)

print(
    f"Correct predictions : "
    f"{correct}/{len(X_test)}"
)

print(
    f"Accuracy            : "
    f"{calculated_accuracy:.2f}%"
)

print()
print("Phrase model test completed.")
print()