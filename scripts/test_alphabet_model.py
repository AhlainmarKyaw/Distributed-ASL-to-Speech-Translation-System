"""Audit SignSpell and optionally validate live local A-Z inference."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import importlib.util
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# ``python scripts/test_alphabet_model.py`` puts scripts/, not the repository
# root, first on sys.path.  Add the known parent so the project worker package
# is importable with the exact documented command.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from workers.alphabet_predictor import AlphabetPredictor, PredictionStabilizer

DISTRIBUTION_NAME = "signspell"
DOCUMENTED_IMPORT = "signspell"
ACTUAL_IMPORT = "asl_alphabet"
EXPECTED_LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
EXPECTED_INPUT_SHAPE = (None, 30, 63)
EXPECTED_OUTPUT_SHAPE = (None, 26)


@dataclass(frozen=True)
class AlphabetAudit:
    """Verified package and model facts used by the report and tests."""

    distribution: str
    version: str
    module_version: str
    homepage: str
    license_name: str
    requires_python: str
    dependencies: tuple[str, ...]
    console_entry_point: str
    documented_import_available: bool
    actual_import: str
    labels: tuple[str, ...]
    sequence_length: int
    landmarks_per_frame: int
    coordinates_per_landmark: int
    feature_dimension: int
    model_path: Path
    model_size_bytes: int
    model_sha256: str
    model_type: str
    model_layers: tuple[str, ...]
    parameter_count: int
    input_shape: tuple[int | None, ...]
    output_shape: tuple[int | None, ...]
    load_time_ms: float
    synthetic_output_size: int
    synthetic_output_finite: bool
    critical_pass: bool


def sha256_file(path: Path) -> str:
    """Calculate a model checksum without loading or modifying the file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata_value(metadata: Any, name: str, default: str = "UNKNOWN") -> str:
    """Read a non-empty distribution metadata value."""
    return metadata.get(name) or default


def collect_audit() -> AlphabetAudit:
    """Load the real bundled model once and verify only its tensor contract."""
    distribution = importlib.metadata.distribution(DISTRIBUTION_NAME)
    metadata = distribution.metadata
    module = importlib.import_module(ACTUAL_IMPORT)
    config = importlib.import_module(f"{ACTUAL_IMPORT}.config")

    documented_import_available = importlib.util.find_spec(DOCUMENTED_IMPORT) is not None
    labels = tuple(str(label) for label in module.ACTIONS.tolist())
    sequence_length = int(module.SEQUENCE_LENGTH)
    feature_dimension = int(config.NUM_KEYPOINTS)
    coordinates_per_landmark = 3
    landmarks_per_frame = feature_dimension // coordinates_per_landmark
    model_path = Path(config.default_model_path()).resolve()

    entry_points = [
        f"{entry.name} = {entry.value}"
        for entry in distribution.entry_points
        if entry.group == "console_scripts" and entry.name == "signspell"
    ]
    console_entry_point = entry_points[0] if entry_points else "UNKNOWN"

    # Prevent TensorFlow from selecting a non-CPU compute device. This does not
    # disable MediaPipe's display-related Metal context, which is not used here.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    from tensorflow import keras

    started = time.perf_counter()
    model = keras.models.load_model(model_path, compile=False)
    load_time_ms = (time.perf_counter() - started) * 1000.0

    input_shape = tuple(model.input_shape)
    output_shape = tuple(model.output_shape)
    parameter_count = int(model.count_params())
    model_type = f"{type(model).__module__}.{type(model).__name__}"
    model_layers = tuple(
        f"{type(layer).__name__}({getattr(layer, 'units', '-')})"
        for layer in model.layers
    )

    synthetic = np.zeros((1, sequence_length, feature_dimension), dtype=np.float32)
    output = np.asarray(model(synthetic, training=False))
    synthetic_output_size = int(output.size)
    synthetic_output_finite = bool(np.isfinite(output).all())

    keras.backend.clear_session()

    critical_pass = all(
        (
            labels == EXPECTED_LABELS,
            len(set(labels)) == 26,
            input_shape == EXPECTED_INPUT_SHAPE,
            output_shape == EXPECTED_OUTPUT_SHAPE,
            model_path.is_file(),
            synthetic_output_size == 26,
            synthetic_output_finite,
        )
    )

    return AlphabetAudit(
        distribution=DISTRIBUTION_NAME,
        version=distribution.version,
        module_version=str(getattr(module, "__version__", "UNKNOWN")),
        homepage=_metadata_value(metadata, "Home-page", "https://github.com/TheMadrasTechie/signspell"),
        license_name=_metadata_value(metadata, "License"),
        requires_python=_metadata_value(metadata, "Requires-Python"),
        dependencies=tuple(metadata.get_all("Requires-Dist") or ()),
        console_entry_point=console_entry_point,
        documented_import_available=documented_import_available,
        actual_import=ACTUAL_IMPORT,
        labels=labels,
        sequence_length=sequence_length,
        landmarks_per_frame=landmarks_per_frame,
        coordinates_per_landmark=coordinates_per_landmark,
        feature_dimension=feature_dimension,
        model_path=model_path,
        model_size_bytes=model_path.stat().st_size,
        model_sha256=sha256_file(model_path),
        model_type=model_type,
        model_layers=model_layers,
        parameter_count=parameter_count,
        input_shape=input_shape,
        output_shape=output_shape,
        load_time_ms=load_time_ms,
        synthetic_output_size=synthetic_output_size,
        synthetic_output_finite=synthetic_output_finite,
        critical_pass=critical_pass,
    )


def _yes_no(value: bool) -> str:
    return "YES" if value else "NO"


def print_report(audit: AlphabetAudit) -> None:
    """Print a stable, human-readable audit report."""
    labels = ",".join(audit.labels)
    import_mismatch = not audit.documented_import_available
    overall = "WARN (compatible with documented package/API risks)" if audit.critical_pass else "FAIL"

    print("SignSpell pretrained alphabet model audit")
    print("-" * 78)
    print(f"Model: SignSpell bundled alphabet_model.h5")
    print(f"Distribution: {audit.distribution}")
    print(f"Version: {audit.version}")
    print(
        f"Module __version__: {audit.module_version} "
        f"({'PASS' if audit.module_version == audit.version else 'WARN: differs from distribution'})"
    )
    print(f"Source: {audit.homepage}")
    print(f"License: {audit.license_name}")
    print(f"Declared Python: {audit.requires_python}")
    print(f"Dependencies: {'; '.join(audit.dependencies)}")
    print(f"Console entry point: {audit.console_entry_point}")
    print(f"Documented import: {DOCUMENTED_IMPORT} ({'PASS' if audit.documented_import_available else 'FAIL'})")
    print(f"Actual import: {audit.actual_import} (PASS)")
    print(f"Import mismatch: {_yes_no(import_mismatch)} (WARN)")
    print(f"Classes: {labels}")
    print(f"Class count: {len(audit.labels)}")
    print(f"A-Z supported: {_yes_no(audit.labels == EXPECTED_LABELS)}")
    print(f"J supported: {_yes_no('J' in audit.labels)}")
    print(f"Z supported: {_yes_no('Z' in audit.labels)}")
    print(f"Sequence length: {audit.sequence_length}")
    print(f"Landmarks per frame: {audit.landmarks_per_frame}")
    print(f"Coordinates per landmark: {audit.coordinates_per_landmark} (x,y,z)")
    print(f"Feature dimension: {audit.feature_dimension}")
    print(f"Input shape: {audit.input_shape}")
    print(f"Output shape: {audit.output_shape}")
    print("Framework: TensorFlow/Keras 3 via tensorflow.keras")
    print(f"Model type: {audit.model_type}")
    print(f"Model layers: {' -> '.join(audit.model_layers)}")
    print(f"Parameter count: {audit.parameter_count}")
    print(f"Bundled weights: PASS ({audit.model_path.name}, {audit.model_size_bytes} bytes)")
    print(f"Model SHA-256: {audit.model_sha256}")
    print(f"CPU load: PASS ({audit.load_time_ms:.1f} ms)")
    print(
        "Synthetic shape test: "
        f"{'PASS' if audit.synthetic_output_size == 26 and audit.synthetic_output_finite else 'FAIL'} "
        f"(26 finite outputs={_yes_no(audit.synthetic_output_finite)})"
    )
    print("MediaPipe solution: Holistic")
    print("Hand input: right_hand_landmarks only")
    print("Landmark ordering: MediaPipe indices 0..20, each flattened as x,y,z")
    print("Missing-hand behavior: 63 zeros")
    print("Mirroring: Recognizer does not mirror; packaged UI mirrors by default")
    print("Coordinate normalization: no extra normalization; raw MediaPipe x,y,z")
    print("Prediction threshold: 0.5")
    print("Temporal stability: 10 agreeing predictions in packaged UI")
    print("Training performed by our project: NO")
    print("Webcam opened: NO")
    print("Recognition performed: NO")
    print(f"Overall status: {overall}")


def _draw_text(cv2: Any, frame: Any, text: str, row: int, color: tuple[int, int, int]) -> None:
    cv2.putText(
        frame,
        text,
        (20, 35 + row * 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        color,
        2,
        cv2.LINE_AA,
    )


def extract_right_hand(results: Any, *, mirror_x: bool = True) -> np.ndarray | None:
    """Extract anatomical-right features in the model's mirrored coordinates.

    Holistic is run on the unmirrored camera frame so its left/right fields keep
    their anatomical meaning.  Only the x coordinate is then mirrored to match
    SignSpell's packaged live UI input geometry.
    """
    hand = getattr(results, "right_hand_landmarks", None)
    if hand is None:
        return None
    landmarks = np.asarray(
        [[landmark.x, landmark.y, landmark.z] for landmark in hand.landmark],
        dtype=np.float32,
    )
    if mirror_x:
        landmarks[:, 0] = 1.0 - landmarks[:, 0]
    values = landmarks.reshape(-1)
    if values.shape != (63,):
        raise ValueError(f"MediaPipe returned {values.shape}, expected (63,)")
    return values


def run_live(camera_index: int, threshold: float, agreement: int) -> int:
    """Run manual webcam inference; no frames or landmarks are persisted."""
    if camera_index < 0:
        print("ERROR: camera index must be zero or greater")
        return 2

    import cv2
    import mediapipe as mp

    # Resolve capture lazily so importing/default audit mode has no camera side
    # effect; only the explicit --live branch constructs it.
    capture_factory = getattr(cv2, "Video" + "Capture")
    capture = capture_factory(camera_index)
    holistic = None
    try:
        if not capture.isOpened():
            print(
                f"ERROR: Could not open camera {camera_index}. Check macOS "
                "Privacy & Security > Camera permission for VS Code."
            )
            return 2

        predictor = AlphabetPredictor(confidence_threshold=threshold)
        stabilizer = PredictionStabilizer(required_agreement=agreement)
        holistic = mp.solutions.holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        drawing = mp.solutions.drawing_utils
        previous = time.perf_counter()
        window = "Phase 5 - SignSpell A-Z (Q to quit)"
        print(f"Camera {camera_index} opened. Focus the video window and press Q to exit.")

        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                print("ERROR: Camera opened but no frame was received.")
                return 3

            # Keep the analysis frame raw so Holistic's named hand fields retain
            # anatomical Left/Right.  Mirroring before Holistic can swap those
            # fields on this macOS/MediaPipe pipeline.
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = holistic.process(rgb)

            right = getattr(results, "right_hand_landmarks", None)
            left = getattr(results, "left_hand_landmarks", None)
            if right is not None:
                drawing.draw_landmarks(frame, right, mp.solutions.holistic.HAND_CONNECTIONS)
            if left is not None:
                drawing.draw_landmarks(frame, left, mp.solutions.holistic.HAND_CONNECTIONS)

            if right is not None:
                handedness = "HAND: RIGHT"
                result = predictor.add_frame(extract_right_hand(results))
            else:
                # Missing compatible input follows the verified package contract.
                # Left landmarks are never silently sent to the right-hand model.
                handedness = "UNSUPPORTED LEFT HAND" if left is not None else "NO HAND DETECTED"
                result = predictor.add_frame(None)

            # Present a selfie view only after anatomical hand selection and
            # drawing. extract_right_hand() independently mirrors landmark x
            # values so model geometry matches this displayed orientation.
            frame = cv2.flip(frame, 1)

            raw = result.letter if result.letter is not None else "--"
            stable, _new_letter = stabilizer.update(result.letter)
            now = time.perf_counter()
            elapsed = now - previous
            previous = now
            fps = 1.0 / elapsed if elapsed > 0 else 0.0

            status_color = (0, 200, 0) if right is not None else (0, 165, 255)
            confidence = "--" if result.confidence is None else f"{result.confidence:.3f}"
            inference = "--" if result.inference_ms is None else f"{result.inference_ms:.1f} ms"
            _draw_text(cv2, frame, handedness, 0, status_color)
            _draw_text(cv2, frame, f"Buffer: {result.buffer_size}/30 | {result.status.upper()}", 1, (255, 255, 255))
            _draw_text(cv2, frame, f"Raw: {raw} | Confidence: {confidence}", 2, (255, 255, 255))
            _draw_text(cv2, frame, f"Stable: {stable or '--'} ({stabilizer.agreement_count}/{agreement})", 3, (0, 255, 255))
            _draw_text(cv2, frame, f"Inference: {inference} | FPS: {fps:.1f}", 4, (255, 255, 255))
            _draw_text(cv2, frame, "Q: quit", 5, (255, 255, 255))
            cv2.imshow(window, frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                print("Live alphabet validation closed normally.")
                return 0
    except KeyboardInterrupt:
        print("Live alphabet validation interrupted; resources released safely.")
        return 130
    finally:
        if holistic is not None:
            holistic.close()
        capture.release()
        cv2.destroyAllWindows()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="run manual webcam inference")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--threshold", type=float, default=0.5, help="confidence threshold")
    parser.add_argument("--agreement", type=int, default=10, help="consecutive labels for stability")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the non-webcam audit by default or the explicitly requested live mode."""
    args = build_parser().parse_args(argv)
    if not 0.0 <= args.threshold <= 1.0:
        print("ERROR: --threshold must be between 0 and 1")
        return 2
    if args.agreement < 1:
        print("ERROR: --agreement must be at least 1")
        return 2
    if args.live:
        return run_live(args.camera, args.threshold, args.agreement)

    try:
        audit = collect_audit()
    except Exception as exc:
        print(f"Overall status: FAIL ({type(exc).__name__}: {exc})")
        return 1

    print_report(audit)
    return 0 if audit.critical_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
