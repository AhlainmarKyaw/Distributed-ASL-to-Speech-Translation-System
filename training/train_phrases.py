from pathlib import Path
import json

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow import keras


# =========================================================
# PATHS
# =========================================================

ROOT = Path("datasets/phrases")
MODEL_DIR = Path("models")

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# =========================================================
# PHRASE MODEL CONFIGURATION
# =========================================================

SEQUENCE_LENGTH = 30

# Two hands:
# 21 landmarks × 3 coordinates × 2 hands
FEATURE_COUNT = 126

# Classes with fewer than this number of samples
# will not be used for training.
MIN_SAMPLES_PER_CLASS = 4


# =========================================================
# LOAD DATASET
# =========================================================

X = []
y = []

print()
print("=" * 60)
print("LOADING WLASL PHRASE DATASET")
print("=" * 60)


if not ROOT.exists():
    raise SystemExit(
        f"Dataset directory not found: {ROOT}"
    )


class_folders = sorted(
    [
        folder
        for folder in ROOT.iterdir()
        if folder.is_dir()
    ]
)


if not class_folders:
    raise SystemExit(
        "No phrase class folders were found."
    )


for class_folder in class_folders:

    files = sorted(
        class_folder.glob("*.npy")
    )

    sample_count = len(files)

    # -----------------------------------------------------
    # Skip classes with too few samples
    # -----------------------------------------------------

    if sample_count < MIN_SAMPLES_PER_CLASS:

        print(
            f"{class_folder.name:15} "
            f"SKIPPED - only {sample_count} samples"
        )

        continue

    loaded = 0
    skipped = 0

    for file_path in files:

        try:
            sequence = np.load(
                file_path
            )

        except Exception as error:

            print(
                f"Could not load "
                f"{file_path.name}: {error}"
            )

            skipped += 1
            continue

        # -------------------------------------------------
        # Expected WLASL phrase sequence:
        #
        # 30 frames × 126 features
        # -------------------------------------------------

        if sequence.shape != (
            SEQUENCE_LENGTH,
            FEATURE_COUNT,
        ):

            skipped += 1

            print(
                f"Invalid shape skipped: "
                f"{file_path.name} "
                f"{sequence.shape}"
            )

            continue

        X.append(
            sequence.astype(
                np.float32
            )
        )

        label = class_folder.name.replace(
            "_",
            " ",
        )

        y.append(label)

        loaded += 1

    print(
        f"{class_folder.name:15} "
        f"loaded={loaded} "
        f"skipped={skipped}"
    )


# =========================================================
# VALIDATE DATA
# =========================================================

if len(X) == 0:
    raise SystemExit(
        "No valid phrase sequences were found."
    )


unique_labels = sorted(
    set(y)
)


if len(unique_labels) < 2:
    raise SystemExit(
        "At least two phrase classes are required."
    )


X = np.asarray(
    X,
    dtype=np.float32,
)


encoder = LabelEncoder()

y_encoded = encoder.fit_transform(
    y
)


print()
print("=" * 60)
print("DATASET SUMMARY")
print("=" * 60)

print(
    "Dataset shape:",
    X.shape,
)

print(
    "Classes:",
    encoder.classes_.tolist(),
)

print(
    "Number of classes:",
    len(
        encoder.classes_
    ),
)

print(
    "Total usable samples:",
    len(X),
)


# =========================================================
# SHOW CLASS COUNTS
# =========================================================

print()
print("Class distribution:")
print("-" * 60)


for class_name in encoder.classes_:

    count = y.count(
        class_name
    )

    print(
        f"{class_name:15} "
        f"{count} samples"
    )


# =========================================================
# TRAIN / TEST SPLIT
# =========================================================

# A larger test proportion is used because this
# WLASL subset is currently quite small.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.25,
    random_state=42,
    stratify=y_encoded,
)


print()
print("=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

print(
    "Training samples:",
    len(X_train),
)

print(
    "Testing samples:",
    len(X_test),
)


# =========================================================
# BUILD GRU MODEL
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
            0.25,
            name="dropout_1",
        ),

        keras.layers.GRU(
            64,
            name="gru_2",
        ),

        keras.layers.Dropout(
            0.20,
            name="dropout_2",
        ),

        keras.layers.Dense(
            64,
            activation="relu",
            name="dense_features",
        ),

        keras.layers.Dropout(
            0.15,
            name="dropout_3",
        ),

        keras.layers.Dense(
            len(
                encoder.classes_
            ),
            activation="softmax",
            name="phrase_output",
        ),
    ]
)


# =========================================================
# COMPILE
# =========================================================

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
print("=" * 60)
print("PHRASE MODEL")
print("=" * 60)

model.summary()


# =========================================================
# CALLBACKS
# =========================================================

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1,
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        min_lr=0.00001,
        verbose=1,
    ),
]


# =========================================================
# TRAIN
# =========================================================

print()
print("=" * 60)
print("TRAINING PHRASE MODEL")
print("=" * 60)


history = model.fit(
    X_train,
    y_train,

    validation_split=0.20,

    epochs=70,

    batch_size=8,

    callbacks=callbacks,

    shuffle=True,

    verbose=2,
)


# =========================================================
# EVALUATE
# =========================================================

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0,
)


print()
print("=" * 60)
print("PHRASE MODEL TEST RESULT")
print("=" * 60)

print(
    f"Test loss     : "
    f"{test_loss:.4f}"
)

print(
    f"Test accuracy : "
    f"{test_accuracy * 100:.2f}%"
)


# =========================================================
# INDIVIDUAL TEST PREDICTIONS
# =========================================================

print()
print("=" * 60)
print("SAMPLE TEST PREDICTIONS")
print("=" * 60)


predictions = model.predict(
    X_test,
    verbose=0,
)


predicted_indices = np.argmax(
    predictions,
    axis=1,
)


for index in range(
    min(
        len(X_test),
        10,
    )
):

    actual_label = encoder.inverse_transform(
        [
            y_test[index]
        ]
    )[0]

    predicted_label = encoder.inverse_transform(
        [
            predicted_indices[index]
        ]
    )[0]

    confidence = float(
        np.max(
            predictions[index]
        )
    )

    print(
        f"Actual: {actual_label:12} "
        f"Predicted: {predicted_label:12} "
        f"Confidence: {confidence:.2%}"
    )


# =========================================================
# SAVE MODEL
# =========================================================

MODEL_PATH = (
    MODEL_DIR
    / "phrase_sequence.keras"
)

LABEL_PATH = (
    MODEL_DIR
    / "phrase_labels.json"
)


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
# SAVE MODEL CONFIG
# =========================================================

CONFIG_PATH = (
    MODEL_DIR
    / "phrase_model_config.json"
)


config = {
    "sequence_length": SEQUENCE_LENGTH,
    "feature_count": FEATURE_COUNT,
    "number_of_classes": int(
        len(
            encoder.classes_
        )
    ),
    "classes": encoder.classes_.tolist(),
    "minimum_samples_per_class": MIN_SAMPLES_PER_CLASS,
}


CONFIG_PATH.write_text(
    json.dumps(
        config,
        indent=2,
    ),
    encoding="utf-8",
)


# =========================================================
# FINISHED
# =========================================================

print()
print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(
    "Phrase model saved:"
)

print(
    MODEL_PATH
)

print()

print(
    "Phrase labels saved:"
)

print(
    LABEL_PATH
)

print()

print(
    "Phrase config saved:"
)

print(
    CONFIG_PATH
)

print()