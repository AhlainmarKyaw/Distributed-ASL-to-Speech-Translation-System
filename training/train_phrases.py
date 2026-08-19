from pathlib import Path
import json

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras


# =========================================================
# PATHS
# =========================================================

DATASET_ROOT = Path("datasets/phrases")
MODEL_DIR = Path("models")

MODEL_PATH = MODEL_DIR / "phrase_sequence.keras"
LABEL_PATH = MODEL_DIR / "phrase_labels.json"
CONFIG_PATH = MODEL_DIR / "phrase_model_config.json"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# CONFIGURATION
# =========================================================

SEQUENCE_LENGTH = 30

# 21 landmarks × 3 coordinates × 2 hands
FEATURE_COUNT = 126

EXPECTED_CLASSES = [
    "GOOD",
    "GOODBYE",
    "HELLO",
    "HELP",
    "HOW",
    "LOVE",
    "MORNING",
    "NO",
    "PLEASE",
    "SORRY",
    "YES",
    "YOU",
]

RANDOM_STATE = 42

TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20

EPOCHS = 100
BATCH_SIZE = 16


# =========================================================
# LOAD DATASET
# =========================================================

print()
print("=" * 70)
print("BALANCED WLASL + WEBCAM PHRASE TRAINING")
print("=" * 70)


if not DATASET_ROOT.exists():
    raise SystemExit(
        f"Dataset folder not found: {DATASET_ROOT}"
    )


X = []
y = []

class_counts = {}


for class_name in EXPECTED_CLASSES:

    class_folder = DATASET_ROOT / class_name

    if not class_folder.exists():

        raise SystemExit(
            f"Missing class folder: {class_folder}"
        )

    files = sorted(
        class_folder.glob("*.npy")
    )

    valid_count = 0
    invalid_count = 0

    for file_path in files:

        try:
            sequence = np.load(
                file_path
            )

        except Exception as error:

            print(
                f"Could not load {file_path}: {error}"
            )

            invalid_count += 1
            continue

        if sequence.shape != (
            SEQUENCE_LENGTH,
            FEATURE_COUNT,
        ):

            print(
                f"Skipped invalid shape: "
                f"{file_path.name} "
                f"{sequence.shape}"
            )

            invalid_count += 1
            continue

        X.append(
            sequence.astype(
                np.float32
            )
        )

        y.append(
            class_name
        )

        valid_count += 1

    class_counts[class_name] = valid_count

    print(
        f"{class_name:12} "
        f"valid={valid_count:3} "
        f"invalid={invalid_count:3}"
    )


# =========================================================
# VALIDATE BALANCE
# =========================================================

print()
print("=" * 70)
print("CLASS DISTRIBUTION")
print("=" * 70)


for class_name in EXPECTED_CLASSES:

    print(
        f"{class_name:12}: "
        f"{class_counts[class_name]}"
    )


if any(
    class_counts[class_name] == 0
    for class_name in EXPECTED_CLASSES
):
    raise SystemExit(
        "At least one class has no valid samples."
    )


X = np.asarray(
    X,
    dtype=np.float32,
)


encoder = LabelEncoder()

y_encoded = encoder.fit_transform(
    y
)


if encoder.classes_.tolist() != EXPECTED_CLASSES:

    raise SystemExit(
        "\nLabel mismatch.\n"
        f"Expected: {EXPECTED_CLASSES}\n"
        f"Found: {encoder.classes_.tolist()}"
    )


print()
print(
    "Dataset shape:",
    X.shape,
)

print(
    "Number of classes:",
    len(
        encoder.classes_
    ),
)

print(
    "Total sequences:",
    len(X),
)


# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

X_train_full, X_test, y_train_full, y_test = (
    train_test_split(
        X,
        y_encoded,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_encoded,
    )
)


# =========================================================
# TRAIN / VALIDATION SPLIT
# =========================================================

X_train, X_val, y_train, y_val = (
    train_test_split(
        X_train_full,
        y_train_full,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train_full,
    )
)


print()
print("=" * 70)
print("DATA SPLIT")
print("=" * 70)

print(
    "Training samples  :",
    len(X_train),
)

print(
    "Validation samples:",
    len(X_val),
)

print(
    "Testing samples   :",
    len(X_test),
)


# =========================================================
# NORMALIZATION
# =========================================================

# Landmark coordinates are already normalized by MediaPipe,
# but standardizing across the training set can help the GRU.

mean = np.mean(
    X_train,
    axis=(0, 1),
    keepdims=True,
)

std = np.std(
    X_train,
    axis=(0, 1),
    keepdims=True,
)

std = np.where(
    std < 1e-6,
    1.0,
    std,
)


X_train_norm = (
    X_train - mean
) / std

X_val_norm = (
    X_val - mean
) / std

X_test_norm = (
    X_test - mean
) / std


# =========================================================
# MODEL
# =========================================================

model = keras.Sequential(
    [
        keras.layers.Input(
            shape=(
                SEQUENCE_LENGTH,
                FEATURE_COUNT,
            ),
            name="phrase_sequence_input",
        ),

        keras.layers.GRU(
            128,
            return_sequences=True,
            name="gru_1",
        ),

        keras.layers.Dropout(
            0.30,
            name="dropout_1",
        ),

        keras.layers.GRU(
            64,
            name="gru_2",
        ),

        keras.layers.Dropout(
            0.25,
            name="dropout_2",
        ),

        keras.layers.Dense(
            64,
            activation="relu",
            name="dense_1",
        ),

        keras.layers.Dropout(
            0.20,
            name="dropout_3",
        ),

        keras.layers.Dense(
            32,
            activation="relu",
            name="dense_2",
        ),

        keras.layers.Dense(
            len(EXPECTED_CLASSES),
            activation="softmax",
            name="phrase_output",
        ),
    ]
)


model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="sparse_categorical_crossentropy",
    metrics=[
        "accuracy"
    ],
)


print()
print("=" * 70)
print("MODEL")
print("=" * 70)

model.summary()


# =========================================================
# CALLBACKS
# =========================================================

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=15,
        restore_best_weights=True,
        verbose=1,
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=6,
        min_lr=1e-5,
        verbose=1,
    ),

    keras.callbacks.ModelCheckpoint(
        filepath=str(
            MODEL_DIR /
            "phrase_sequence_best.keras"
        ),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),
]


# =========================================================
# TRAIN
# =========================================================

print()
print("=" * 70)
print("TRAINING")
print("=" * 70)


history = model.fit(
    X_train_norm,
    y_train,
    validation_data=(
        X_val_norm,
        y_val,
    ),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    shuffle=True,
    verbose=2,
)


# =========================================================
# TEST
# =========================================================

test_loss, test_accuracy = model.evaluate(
    X_test_norm,
    y_test,
    verbose=0,
)


print()
print("=" * 70)
print("TEST RESULT")
print("=" * 70)

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
    X_test_norm,
    verbose=0,
)

predicted_indices = np.argmax(
    probabilities,
    axis=1,
)


print()
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        predicted_indices,
        target_names=
        encoder.classes_,
        digits=4,
        zero_division=0,
    )
)


print()
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print(
    confusion_matrix(
        y_test,
        predicted_indices,
    )
)


# =========================================================
# SAMPLE PREDICTIONS
# =========================================================

print()
print("=" * 70)
print("SAMPLE PREDICTIONS")
print("=" * 70)


for index in range(
    min(
        20,
        len(X_test_norm),
    )
):

    actual_index = int(
        y_test[index]
    )

    predicted_index = int(
        predicted_indices[index]
    )

    actual_label = (
        encoder.classes_[
            actual_index
        ]
    )

    predicted_label = (
        encoder.classes_[
            predicted_index
        ]
    )

    confidence = float(
        np.max(
            probabilities[index]
        )
    )

    status = (
        "CORRECT"
        if actual_index
        == predicted_index
        else "WRONG"
    )

    print(
        f"{status:7} | "
        f"Actual: {actual_label:10} | "
        f"Predicted: {predicted_label:10} | "
        f"Confidence: "
        f"{confidence * 100:6.2f}%"
    )


# =========================================================
# SAVE FINAL MODEL
# =========================================================

model.save(
    MODEL_PATH
)


LABEL_PATH.write_text(
    json.dumps(
        encoder.classes_.tolist(),
        indent=2,
    ),
    encoding="utf-8",
)


# =========================================================
# SAVE NORMALIZATION VALUES
# =========================================================

NORMALIZATION_PATH = (
    MODEL_DIR /
    "phrase_normalization.npz"
)


np.savez(
    NORMALIZATION_PATH,
    mean=mean.astype(
        np.float32
    ),
    std=std.astype(
        np.float32
    ),
)


# =========================================================
# SAVE CONFIGURATION
# =========================================================

CONFIG = {
    "sequence_length":
        SEQUENCE_LENGTH,

    "feature_count":
        FEATURE_COUNT,

    "classes":
        encoder.classes_.tolist(),

    "number_of_classes":
        len(
            encoder.classes_
        ),

    "total_samples":
        len(X),

    "training_samples":
        len(X_train),

    "validation_samples":
        len(X_val),

    "testing_samples":
        len(X_test),

    "test_accuracy":
        float(
            test_accuracy
        ),

    "normalization_file":
        str(
            NORMALIZATION_PATH
        ),

    "dataset":
        "WLASL + webcam",

    "model_type":
        "GRU",
}


CONFIG_PATH.write_text(
    json.dumps(
        CONFIG,
        indent=2,
    ),
    encoding="utf-8",
)


# =========================================================
# FINISHED
# =========================================================

print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)

print()
print(
    "Final model:"
)

print(
    MODEL_PATH
)

print()
print(
    "Best validation model:"
)

print(
    MODEL_DIR /
    "phrase_sequence_best.keras"
)

print()
print(
    "Labels:"
)

print(
    LABEL_PATH
)

print()
print(
    "Normalization:"
)

print(
    NORMALIZATION_PATH
)

print()
print(
    "Configuration:"
)

print(
    CONFIG_PATH
)

print()
print(
    f"FINAL TEST ACCURACY: "
    f"{test_accuracy * 100:.2f}%"
)

print()