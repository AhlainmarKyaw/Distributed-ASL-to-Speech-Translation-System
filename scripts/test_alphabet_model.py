from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_DATASET = PROJECT_ROOT / "datasets" / "alphabet_landmarks.csv"
DEFAULT_MODEL = PROJECT_ROOT / "models" / "alphabet_landmarks.keras"
DEFAULT_LABELS = PROJECT_ROOT / "models" / "alphabet_labels.json"

FEATURE_DIMENSION = 63
DEFAULT_THRESHOLD = 0.70
FALLBACK_LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def load_labels(path: Path) -> list[str]:
    if path.is_file():
        labels = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(labels, list) or not labels:
            raise ValueError(f"Invalid labels file: {path}")
        return [str(label).strip().upper() for label in labels]
    return FALLBACK_LABELS.copy()


def _looks_like_label_series(series: pd.Series) -> bool:
    values = series.dropna().astype(str).str.strip().str.upper()
    if values.empty:
        return False

    unique = set(values.unique().tolist())
    return (
        len(unique) >= 2
        and len(unique) <= 26
        and all(len(value) == 1 and "A" <= value <= "Z" for value in unique)
    )


def find_label_column(frame: pd.DataFrame) -> str:
    preferred = ("label", "letter", "class", "target", "y")

    lower_to_original = {
        str(column).strip().lower(): column
        for column in frame.columns
    }

    for name in preferred:
        if name in lower_to_original:
            return str(lower_to_original[name])

    for column in frame.columns:
        if _looks_like_label_series(frame[column]):
            return str(column)

    raise ValueError(
        "Could not find the label column. Expected a column such as "
        "'label', 'letter', 'class', or a column containing A-Z values."
    )


def load_dataset(path: Path) -> tuple[np.ndarray, np.ndarray, str, list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")

    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Dataset is empty: {path}")

    label_column = find_label_column(frame)
    y = (
        frame[label_column]
        .astype(str)
        .str.strip()
        .str.upper()
        .to_numpy()
    )

    candidate = frame.drop(columns=[label_column])

    numeric_columns: list[str] = []
    ignored_columns: list[str] = []

    for column in candidate.columns:
        converted = pd.to_numeric(candidate[column], errors="coerce")

        if converted.notna().all():
            candidate[column] = converted
            numeric_columns.append(str(column))
        else:
            ignored_columns.append(str(column))

    if len(numeric_columns) != FEATURE_DIMENSION:
        raise ValueError(
            "Expected exactly 63 numeric landmark feature columns after "
            f"removing label/metadata, but found {len(numeric_columns)}.\n"
            f"Label column: {label_column}\n"
            f"Numeric columns found: {numeric_columns[:10]}"
            f"{' ...' if len(numeric_columns) > 10 else ''}\n"
            f"Ignored non-numeric columns: {ignored_columns}"
        )

    x = candidate[numeric_columns].to_numpy(dtype=np.float32)

    if x.shape[1] != FEATURE_DIMENSION:
        raise ValueError(f"Expected feature shape (*, 63), got {x.shape}")

    if not np.isfinite(x).all():
        raise ValueError("Dataset contains NaN or infinite landmark values")

    return x, y, label_column, ignored_columns


def make_confusion_matrix(
    y_true_index: np.ndarray,
    y_pred_index: np.ndarray,
    class_count: int,
) -> np.ndarray:
    matrix = np.zeros((class_count, class_count), dtype=np.int64)

    for true_index, pred_index in zip(y_true_index, y_pred_index):
        matrix[int(true_index), int(pred_index)] += 1

    return matrix


def save_confusion_csv(
    path: Path,
    matrix: np.ndarray,
    labels: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame(
        matrix,
        index=[f"actual_{label}" for label in labels],
        columns=[f"pred_{label}" for label in labels],
    )
    frame.to_csv(path)


def evaluate(
    dataset_path: Path,
    model_path: Path,
    labels_path: Path,
    threshold: float,
    confusion_output: Path | None,
) -> int:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("--threshold must be between 0 and 1")

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            "Train it first with: python training/train_alphabet.py"
        )

    labels = load_labels(labels_path)
    label_to_index = {
        label: index
        for index, label in enumerate(labels)
    }

    x, y_text, label_column, ignored_columns = load_dataset(dataset_path)

    unknown = sorted(set(y_text) - set(labels))
    if unknown:
        raise ValueError(
            "Dataset contains labels not present in alphabet_labels.json: "
            + ", ".join(unknown)
        )

    y_true = np.asarray(
        [label_to_index[label] for label in y_text],
        dtype=np.int64,
    )

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    from tensorflow import keras

    print("=" * 72)
    print("CURRENT PROJECT ALPHABET MODEL EVALUATION")
    print("=" * 72)
    print(f"Model:       {model_path}")
    print(f"Dataset:     {dataset_path}")
    print(f"Labels:      {labels_path}")
    print(f"Samples:     {len(x)}")
    print(f"Input shape: {x.shape}")
    print(f"Label col:   {label_column}")
    print(f"Threshold:   {threshold:.2f}")

    if ignored_columns:
        print(f"Metadata ignored: {', '.join(ignored_columns)}")

    print("\nLoading model...")
    model = keras.models.load_model(model_path, compile=False)

    print(f"Model input shape:  {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")

    expected_input_last = int(model.input_shape[-1])
    if expected_input_last != FEATURE_DIMENSION:
        raise ValueError(
            f"Current evaluator expects a 63-feature alphabet model, "
            f"but model input is {model.input_shape}"
        )

    probabilities = np.asarray(
        model.predict(x, verbose=0),
        dtype=np.float32,
    )

    if probabilities.ndim != 2:
        raise ValueError(
            f"Expected model output shape (samples, classes), got {probabilities.shape}"
        )

    if probabilities.shape[1] != len(labels):
        raise ValueError(
            "Model output class count does not match labels file: "
            f"{probabilities.shape[1]} outputs vs {len(labels)} labels"
        )

    y_pred = np.argmax(probabilities, axis=1)
    confidence = probabilities[np.arange(len(probabilities)), y_pred]

    raw_correct = y_pred == y_true
    raw_accuracy = float(np.mean(raw_correct))

    accepted = confidence >= threshold
    accepted_count = int(np.sum(accepted))
    rejected_count = len(x) - accepted_count
    coverage = accepted_count / len(x)

    if accepted_count:
        accepted_accuracy = float(np.mean(raw_correct[accepted]))
        accepted_mean_confidence = float(np.mean(confidence[accepted]))
    else:
        accepted_accuracy = float("nan")
        accepted_mean_confidence = float("nan")

    thresholded_correct_count = int(np.sum(raw_correct & accepted))
    thresholded_overall_accuracy = thresholded_correct_count / len(x)

    print("\n" + "-" * 72)
    print("OVERALL RESULTS")
    print("-" * 72)
    print(f"Raw top-1 correct:             {int(np.sum(raw_correct))}/{len(x)}")
    print(f"Raw top-1 accuracy:            {raw_accuracy * 100:.2f}%")
    print(f"Mean prediction confidence:    {float(np.mean(confidence)) * 100:.2f}%")
    print(f"Accepted at threshold:         {accepted_count}/{len(x)}")
    print(f"Rejected/uncertain:            {rejected_count}/{len(x)}")
    print(f"Coverage:                      {coverage * 100:.2f}%")

    if accepted_count:
        print(f"Accuracy among accepted:       {accepted_accuracy * 100:.2f}%")
        print(f"Mean confidence (accepted):    {accepted_mean_confidence * 100:.2f}%")
    else:
        print("Accuracy among accepted:       N/A")
        print("Mean confidence (accepted):    N/A")

    print(
        f"Thresholded overall success:   "
        f"{thresholded_overall_accuracy * 100:.2f}%"
    )

    print("\n" + "-" * 72)
    print("PER-CLASS RESULTS")
    print("-" * 72)
    print(
        f"{'Class':<7}"
        f"{'Samples':>9}"
        f"{'Correct':>10}"
        f"{'Raw Acc':>11}"
        f"{'Accepted':>11}"
        f"{'Accepted Acc':>15}"
    )

    for index, label in enumerate(labels):
        class_mask = y_true == index
        class_total = int(np.sum(class_mask))

        if class_total == 0:
            print(
                f"{label:<7}{0:>9}{0:>10}"
                f"{'N/A':>11}{0:>11}{'N/A':>15}"
            )
            continue

        class_correct = int(np.sum(raw_correct & class_mask))
        class_raw_accuracy = class_correct / class_total

        class_accepted_mask = class_mask & accepted
        class_accepted = int(np.sum(class_accepted_mask))

        if class_accepted:
            class_accepted_accuracy = float(
                np.mean(raw_correct[class_accepted_mask])
            )
            accepted_text = f"{class_accepted_accuracy * 100:.2f}%"
        else:
            accepted_text = "N/A"

        print(
            f"{label:<7}"
            f"{class_total:>9}"
            f"{class_correct:>10}"
            f"{class_raw_accuracy * 100:>10.2f}%"
            f"{class_accepted:>11}"
            f"{accepted_text:>15}"
        )

    matrix = make_confusion_matrix(
        y_true,
        y_pred,
        len(labels),
    )

    if confusion_output is not None:
        save_confusion_csv(
            confusion_output,
            matrix,
            labels,
        )
        print(f"\nConfusion matrix saved to: {confusion_output}")

    print("\n" + "=" * 72)
    print("INTERPRETATION")
    print("=" * 72)

    if dataset_path.resolve() == DEFAULT_DATASET.resolve():
        print(
            "WARNING: This is the project's alphabet_landmarks.csv. "
            "If this same CSV was used to train the saved model, the result is "
            "dataset-fit accuracy, NOT independent test accuracy."
        )
        print(
            "For your report/viva, create or keep a separate unseen labelled "
            "landmark CSV and run this same script with --dataset <file>."
        )
    else:
        print(
            "You supplied a different dataset file. If none of its samples were "
            "used during training/model selection, this can be reported as "
            "independent test accuracy."
        )

    print(
        "Also test live webcam performance separately because lighting, signer, "
        "distance, orientation, and MediaPipe preprocessing can change accuracy."
    )

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="labelled CSV containing 63 alphabet landmark features",
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="current Keras alphabet model",
    )

    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_LABELS,
        help="alphabet labels JSON file",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="confidence threshold used by the current AlphabetPredictor",
    )

    parser.add_argument(
        "--save-confusion",
        type=Path,
        default=PROJECT_ROOT / "reports" / "alphabet_confusion.csv",
        help="path for the raw top-1 confusion matrix CSV",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        return evaluate(
            dataset_path=args.dataset.resolve(),
            model_path=args.model.resolve(),
            labels_path=args.labels.resolve(),
            threshold=args.threshold,
            confusion_output=args.save_confusion.resolve()
            if args.save_confusion
            else None,
        )
    except Exception as exc:
        print(f"\nERROR: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
