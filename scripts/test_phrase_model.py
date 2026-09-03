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

DEFAULT_DATASET = PROJECT_ROOT / "datasets" / "phrases"
DEFAULT_MODEL_BEST = PROJECT_ROOT / "models" / "phrase_sequence_best.keras"
DEFAULT_MODEL_FALLBACK = PROJECT_ROOT / "models" / "phrase_sequence.keras"
DEFAULT_LABELS = PROJECT_ROOT / "models" / "phrase_labels.json"
DEFAULT_NORMALIZATION = PROJECT_ROOT / "models" / "phrase_normalization.npz"
DEFAULT_CONFIG = PROJECT_ROOT / "models" / "phrase_model_config.json"

DEFAULT_SEQUENCE_LENGTH = 30
DEFAULT_FEATURE_COUNT = 126
DEFAULT_THRESHOLD = 0.70

FALLBACK_LABELS = [
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


def load_json(path: Path) -> dict | list:
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_labels(path: Path) -> list[str]:
    if path.is_file():
        data = load_json(path)

        if isinstance(data, list):
            labels = data
        elif isinstance(data, dict):
            # Be tolerant of common label-file layouts.
            for key in ("labels", "classes", "phrase_labels"):
                if isinstance(data.get(key), list):
                    labels = data[key]
                    break
            else:
                raise ValueError(
                    f"Could not find labels list inside {path}. "
                    "Expected a JSON list or a dict containing labels/classes."
                )
        else:
            raise ValueError(f"Invalid labels JSON structure: {path}")

        labels = [str(label).strip().upper() for label in labels]

        if not labels:
            raise ValueError(f"No labels found in: {path}")

        return labels

    return FALLBACK_LABELS.copy()


def load_config(path: Path) -> dict:
    if not path.is_file():
        return {}

    data = load_json(path)

    if not isinstance(data, dict):
        raise ValueError(f"Phrase config must be a JSON object: {path}")

    return data


def config_number(
    config: dict,
    keys: tuple[str, ...],
    default: float | int,
):
    for key in keys:
        if key in config:
            return config[key]
    return default


def resolve_model_path(requested: Path | None) -> Path:
    if requested is not None:
        return requested

    if DEFAULT_MODEL_BEST.is_file():
        return DEFAULT_MODEL_BEST

    return DEFAULT_MODEL_FALLBACK


def load_normalization(
    path: Path,
    sequence_length: int,
    feature_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(
            f"Phrase normalization file not found: {path}"
        )

    data = np.load(path)

    available = set(data.files)

    mean_key = None
    std_key = None

    for key in ("mean", "x_mean", "feature_mean"):
        if key in available:
            mean_key = key
            break

    for key in ("std", "x_std", "feature_std"):
        if key in available:
            std_key = key
            break

    if mean_key is None or std_key is None:
        raise ValueError(
            f"Could not find mean/std arrays in {path}. "
            f"Available arrays: {sorted(available)}"
        )

    mean = np.asarray(data[mean_key], dtype=np.float32)
    std = np.asarray(data[std_key], dtype=np.float32)

    # Runtime normalization may save shapes such as:
    #   (126,)
    #   (1, 126)
    #   (30, 126)
    #   (1, 1, 126)
    #   (1, 30, 126)
    #
    # Evaluation data has shape (N, sequence_length, feature_count), so
    # normalization only needs to broadcast across that 3-D batch layout.
    target_shape = (1, sequence_length, feature_count)

    try:
        np.broadcast_to(mean, target_shape)
        np.broadcast_to(std, target_shape)
    except ValueError as exc:
        raise ValueError(
            "Normalization arrays cannot broadcast to phrase batch shape "
            f"{target_shape}. mean={mean.shape}, std={std.shape}"
        ) from exc

    # Avoid divide-by-zero.
    std = np.where(np.abs(std) < 1e-8, 1.0, std).astype(np.float32)

    return mean, std


def load_dataset(
    root: Path,
    labels: list[str],
    sequence_length: int,
    feature_count: int,
) -> tuple[np.ndarray, np.ndarray, list[Path], dict[str, int], list[str]]:
    if not root.is_dir():
        raise FileNotFoundError(f"Phrase dataset folder not found: {root}")

    sequences: list[np.ndarray] = []
    y_true: list[int] = []
    paths: list[Path] = []
    counts: dict[str, int] = {}
    skipped: list[str] = []

    for class_index, label in enumerate(labels):
        class_dir = root / label

        if not class_dir.is_dir():
            counts[label] = 0
            skipped.append(f"Missing class folder: {class_dir}")
            continue

        files = sorted(class_dir.glob("*.npy"))
        counts[label] = 0

        for file_path in files:
            try:
                sequence = np.asarray(
                    np.load(file_path),
                    dtype=np.float32,
                )
            except Exception as exc:
                skipped.append(
                    f"{file_path}: could not load ({type(exc).__name__}: {exc})"
                )
                continue

            expected_shape = (sequence_length, feature_count)

            if sequence.shape != expected_shape:
                skipped.append(
                    f"{file_path}: shape {sequence.shape}, expected {expected_shape}"
                )
                continue

            if not np.isfinite(sequence).all():
                skipped.append(
                    f"{file_path}: contains NaN or infinite values"
                )
                continue

            sequences.append(sequence)
            y_true.append(class_index)
            paths.append(file_path)
            counts[label] += 1

    if not sequences:
        raise ValueError(
            f"No valid phrase .npy sequences found under: {root}"
        )

    x = np.stack(sequences).astype(np.float32)
    y = np.asarray(y_true, dtype=np.int64)

    return x, y, paths, counts, skipped


def make_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_count: int,
) -> np.ndarray:
    matrix = np.zeros(
        (class_count, class_count),
        dtype=np.int64,
    )

    for actual, predicted in zip(y_true, y_pred):
        matrix[int(actual), int(predicted)] += 1

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


def top_confusions(
    matrix: np.ndarray,
    labels: list[str],
    limit: int = 10,
) -> list[tuple[str, str, int]]:
    items: list[tuple[str, str, int]] = []

    for actual_index, actual_label in enumerate(labels):
        for pred_index, pred_label in enumerate(labels):
            if actual_index == pred_index:
                continue

            count = int(matrix[actual_index, pred_index])

            if count > 0:
                items.append(
                    (actual_label, pred_label, count)
                )

    items.sort(
        key=lambda item: item[2],
        reverse=True,
    )

    return items[:limit]


def evaluate(
    dataset_path: Path,
    model_path: Path,
    labels_path: Path,
    normalization_path: Path,
    config_path: Path,
    threshold_override: float | None,
    confusion_output: Path | None,
) -> int:
    labels = load_labels(labels_path)
    config = load_config(config_path)

    sequence_length = int(
        config_number(
            config,
            ("sequence_length", "sequence_len", "frames"),
            DEFAULT_SEQUENCE_LENGTH,
        )
    )

    feature_count = int(
        config_number(
            config,
            ("feature_count", "feature_dimension", "features"),
            DEFAULT_FEATURE_COUNT,
        )
    )

    configured_threshold = float(
        config_number(
            config,
            (
                "threshold",
                "confidence_threshold",
                "prediction_threshold",
            ),
            DEFAULT_THRESHOLD,
        )
    )

    threshold = (
        float(threshold_override)
        if threshold_override is not None
        else configured_threshold
    )

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            f"Phrase confidence threshold must be between 0 and 1, got {threshold}"
        )

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Phrase model not found: {model_path}\n"
            "Train it with: python training/train_phrases.py"
        )

    mean, std = load_normalization(
        normalization_path,
        sequence_length,
        feature_count,
    )

    x_raw, y_true, sample_paths, counts, skipped = load_dataset(
        dataset_path,
        labels,
        sequence_length,
        feature_count,
    )

    x = (
        (x_raw - mean) / std
    ).astype(np.float32)

    if x.shape[1:] != (sequence_length, feature_count):
        raise ValueError(
            f"Normalized dataset has unexpected shape: {x.shape}"
        )

    if not np.isfinite(x).all():
        raise ValueError(
            "Normalized phrase data contains NaN or infinite values"
        )

    os.environ.setdefault(
        "CUDA_VISIBLE_DEVICES",
        "-1",
    )
    os.environ.setdefault(
        "TF_CPP_MIN_LOG_LEVEL",
        "2",
    )

    from tensorflow import keras

    print("=" * 78)
    print("CURRENT PROJECT DYNAMIC PHRASE MODEL EVALUATION")
    print("=" * 78)
    print(f"Model:          {model_path}")
    print(f"Dataset:        {dataset_path}")
    print(f"Labels:         {labels_path}")
    print(f"Normalization:  {normalization_path}")
    print(f"Config:         {config_path}")
    print(f"Classes:        {len(labels)}")
    print(f"Samples:        {len(x)}")
    print(f"Input shape:    {x.shape}")
    print(f"Sequence:       {sequence_length} frames")
    print(f"Features/frame: {feature_count}")
    print(f"Threshold:      {threshold:.2f}")

    print("\nClass sample counts:")
    for label in labels:
        print(f"  {label:<12} {counts.get(label, 0)}")

    if skipped:
        print(f"\nSkipped invalid/missing entries: {len(skipped)}")
        for item in skipped[:10]:
            print(f"  - {item}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")

    print("\nLoading model...")
    model = keras.models.load_model(
        model_path,
        compile=False,
    )

    print(f"Model input shape:  {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")

    expected_model_input = (
        sequence_length,
        feature_count,
    )

    actual_input = tuple(
        int(value)
        for value in model.input_shape[-2:]
    )

    if actual_input != expected_model_input:
        raise ValueError(
            "Model input does not match phrase configuration: "
            f"model expects {model.input_shape}, "
            f"evaluator expects (None, {sequence_length}, {feature_count})"
        )

    probabilities = np.asarray(
        model.predict(
            x,
            verbose=0,
        ),
        dtype=np.float32,
    )

    if probabilities.ndim != 2:
        raise ValueError(
            f"Expected model output (samples, classes), got {probabilities.shape}"
        )

    if probabilities.shape[1] != len(labels):
        raise ValueError(
            "Phrase model class count does not match phrase labels: "
            f"{probabilities.shape[1]} outputs vs {len(labels)} labels"
        )

    y_pred = np.argmax(
        probabilities,
        axis=1,
    )

    confidence = probabilities[
        np.arange(len(probabilities)),
        y_pred,
    ]

    raw_correct = y_pred == y_true
    raw_correct_count = int(
        np.sum(raw_correct)
    )
    raw_accuracy = float(
        np.mean(raw_correct)
    )

    accepted = confidence >= threshold
    accepted_count = int(
        np.sum(accepted)
    )
    rejected_count = (
        len(x) - accepted_count
    )
    coverage = (
        accepted_count / len(x)
    )

    if accepted_count:
        accepted_accuracy = float(
            np.mean(
                raw_correct[accepted]
            )
        )
        accepted_mean_confidence = float(
            np.mean(
                confidence[accepted]
            )
        )
    else:
        accepted_accuracy = float("nan")
        accepted_mean_confidence = float("nan")

    thresholded_correct_count = int(
        np.sum(
            raw_correct & accepted
        )
    )

    thresholded_overall_success = (
        thresholded_correct_count
        / len(x)
    )

    print("\n" + "-" * 78)
    print("OVERALL RESULTS")
    print("-" * 78)
    print(
        f"Raw top-1 correct:             "
        f"{raw_correct_count}/{len(x)}"
    )
    print(
        f"Raw top-1 accuracy:            "
        f"{raw_accuracy * 100:.2f}%"
    )
    print(
        f"Mean prediction confidence:    "
        f"{float(np.mean(confidence)) * 100:.2f}%"
    )
    print(
        f"Accepted at threshold:         "
        f"{accepted_count}/{len(x)}"
    )
    print(
        f"Rejected/uncertain:            "
        f"{rejected_count}/{len(x)}"
    )
    print(
        f"Coverage:                      "
        f"{coverage * 100:.2f}%"
    )

    if accepted_count:
        print(
            f"Accuracy among accepted:       "
            f"{accepted_accuracy * 100:.2f}%"
        )
        print(
            f"Mean confidence (accepted):    "
            f"{accepted_mean_confidence * 100:.2f}%"
        )
    else:
        print(
            "Accuracy among accepted:       N/A"
        )
        print(
            "Mean confidence (accepted):    N/A"
        )

    print(
        f"Thresholded overall success:   "
        f"{thresholded_overall_success * 100:.2f}%"
    )

    print("\n" + "-" * 78)
    print("PER-CLASS RESULTS")
    print("-" * 78)
    print(
        f"{'Phrase':<12}"
        f"{'Samples':>9}"
        f"{'Correct':>10}"
        f"{'Raw Acc':>11}"
        f"{'Accepted':>11}"
        f"{'Accepted Acc':>15}"
    )

    for class_index, label in enumerate(labels):
        class_mask = y_true == class_index
        total = int(
            np.sum(class_mask)
        )

        if total == 0:
            print(
                f"{label:<12}"
                f"{0:>9}"
                f"{0:>10}"
                f"{'N/A':>11}"
                f"{0:>11}"
                f"{'N/A':>15}"
            )
            continue

        correct = int(
            np.sum(
                raw_correct & class_mask
            )
        )

        raw_class_accuracy = (
            correct / total
        )

        accepted_mask = (
            class_mask & accepted
        )

        accepted_for_class = int(
            np.sum(accepted_mask)
        )

        if accepted_for_class:
            accepted_class_accuracy = float(
                np.mean(
                    raw_correct[
                        accepted_mask
                    ]
                )
            )

            accepted_text = (
                f"{accepted_class_accuracy * 100:.2f}%"
            )
        else:
            accepted_text = "N/A"

        print(
            f"{label:<12}"
            f"{total:>9}"
            f"{correct:>10}"
            f"{raw_class_accuracy * 100:>10.2f}%"
            f"{accepted_for_class:>11}"
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

        print(
            f"\nConfusion matrix saved to: "
            f"{confusion_output}"
        )

    confusions = top_confusions(
        matrix,
        labels,
        limit=10,
    )

    print("\n" + "-" * 78)
    print("MOST COMMON WRONG PREDICTIONS")
    print("-" * 78)

    if not confusions:
        print("No top-1 confusions in this dataset.")
    else:
        for actual, predicted, count in confusions:
            print(
                f"{actual:<12} -> "
                f"{predicted:<12} : {count}"
            )

    # Also provide a few lowest-confidence samples for debugging.
    order = np.argsort(confidence)

    print("\n" + "-" * 78)
    print("LOWEST-CONFIDENCE SAMPLES")
    print("-" * 78)

    for index in order[: min(10, len(order))]:
        actual = labels[int(y_true[index])]
        predicted = labels[int(y_pred[index])]
        print(
            f"{sample_paths[index].name:<24} "
            f"actual={actual:<12} "
            f"pred={predicted:<12} "
            f"conf={confidence[index] * 100:6.2f}%"
        )

    print("\n" + "=" * 78)
    print("INTERPRETATION")
    print("=" * 78)

    if dataset_path.resolve() == DEFAULT_DATASET.resolve():
        print(
            "WARNING: You evaluated datasets/phrases. If these same .npy "
            "sequences were used during training or model selection, this is "
            "dataset-fit accuracy, not independent test accuracy."
        )
        print(
            "For a defensible report result, test on a separate unseen folder "
            "with the same structure: <test_root>/<PHRASE>/*.npy."
        )
    else:
        print(
            "You supplied a different phrase dataset folder. If none of these "
            "sequences were used during training/model selection, this can be "
            "reported as independent phrase test accuracy."
        )

    print(
        "Also measure live-webcam phrase accuracy separately because hand "
        "ordering, missing-hand handling, 30-frame sampling, signer, camera "
        "distance, and lighting can change real-world performance."
    )

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=(
            "root folder containing "
            "<PHRASE>/*.npy sequence files"
        ),
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help=(
            "Keras phrase model. By default uses "
            "phrase_sequence_best.keras if present."
        ),
    )

    parser.add_argument(
        "--labels",
        type=Path,
        default=DEFAULT_LABELS,
        help="phrase labels JSON file",
    )

    parser.add_argument(
        "--normalization",
        type=Path,
        default=DEFAULT_NORMALIZATION,
        help="saved phrase mean/std NPZ file",
    )

    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="phrase model configuration JSON file",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "override confidence threshold. "
            "Otherwise uses phrase_model_config.json."
        ),
    )

    parser.add_argument(
        "--save-confusion",
        type=Path,
        default=(
            PROJECT_ROOT
            / "reports"
            / "phrase_confusion.csv"
        ),
        help="CSV path for confusion matrix",
    )

    return parser


def main(
    argv: list[str] | None = None,
) -> int:
    args = build_parser().parse_args(
        argv
    )

    model_path = resolve_model_path(
        args.model.resolve()
        if args.model is not None
        else None
    )

    try:
        return evaluate(
            dataset_path=args.dataset.resolve(),
            model_path=model_path.resolve(),
            labels_path=args.labels.resolve(),
            normalization_path=args.normalization.resolve(),
            config_path=args.config.resolve(),
            threshold_override=args.threshold,
            confusion_output=(
                args.save_confusion.resolve()
                if args.save_confusion
                else None
            ),
        )
    except Exception as exc:
        print(
            f"\nERROR: "
            f"{type(exc).__name__}: {exc}"
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
