"""Audit Phase 2 runtime imports without running recognition or opening a webcam."""

from __future__ import annotations

import importlib
import importlib.metadata
import argparse
import platform
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Dependency:
    """A distribution name and its corresponding Python import name."""

    distribution: str
    module: str


DEPENDENCIES = (
    Dependency("numpy", "numpy"),
    Dependency("opencv-python", "cv2"),
    Dependency("mediapipe", "mediapipe"),
    Dependency("fastapi", "fastapi"),
    Dependency("uvicorn", "uvicorn"),
    Dependency("redis", "redis"),
    Dependency("celery", "celery"),
    Dependency("pyttsx3", "pyttsx3"),
    Dependency("psutil", "psutil"),
    Dependency("torch", "torch"),
    Dependency("tensorflow", "tensorflow"),
    Dependency("huggingface-hub", "huggingface_hub"),
)


def version_of(distribution: str) -> str:
    """Return installed distribution version or a clear missing marker."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def check_dependency(dependency: Dependency) -> bool:
    """Import one dependency and print a stable PASS/FAIL line."""
    version = version_of(dependency.distribution)
    try:
        importlib.import_module(dependency.module)
    except Exception as exc:  # Native wheels can raise errors other than ImportError.
        print(
            f"FAIL  {dependency.distribution:<20} {version:<14} "
            f"{type(exc).__name__}: {exc}"
        )
        return False

    print(f"PASS  {dependency.distribution:<20} {version}")
    return True


def check_python() -> bool:
    """Require the audited Python 3.11 runtime exactly."""
    passed = sys.version_info[:2] == (3, 11)
    status = "PASS" if passed else "FAIL"
    print(f"{status}  Python               {platform.python_version()} (required: 3.11.x)")
    return passed


def check_signspell() -> bool:
    """Check SignSpell packaging and bundled model without loading the model."""
    version = version_of("signspell")
    if version == "not installed":
        print("FAIL  signspell            not installed")
        return False

    documented_import_ok = True
    try:
        importlib.import_module("signspell")
    except Exception:
        documented_import_ok = False

    try:
        package = importlib.import_module("asl_alphabet")
        model_path = Path(package.__file__).parent / "models" / "alphabet_model.h5"
        model_ok = model_path.is_file() and model_path.stat().st_size > 0
    except Exception as exc:
        print(f"FAIL  signspell            {version} actual import failed: {exc}")
        return False

    if documented_import_ok and model_ok:
        print(f"PASS  signspell            {version} documented import and model found")
        return True

    if not documented_import_ok and model_ok:
        print(
            f"WARN  signspell            {version} model and public API are available as "
            "'asl_alphabet'; documented 'import signspell' is missing"
        )
        return True

    print(f"FAIL  signspell            {version} bundled alphabet_model.h5 is missing")
    return False


def check_phrase_model() -> None:
    """Confirm whether the selected phrase checkpoint and label maps exist locally."""
    root = Path(__file__).resolve().parents[1] / "pretrained" / "phrases"
    checkpoint = root / "best_bigru_attention_model.pt"
    label_to_id = root / "label_to_id.json"
    id_to_label = root / "id_to_label.json"
    passed = all(path.is_file() for path in (checkpoint, label_to_id, id_to_label))
    status = "PASS" if passed else "PENDING"
    print(
        f"{status}  phrase model         checkpoint + both label maps "
        f"{'found' if passed else 'not installed/verified'}"
    )


def check_mediapipe_wheel_tag() -> bool:
    """Expose the known 0.10.18 internal wheel-tag mismatch on Apple Silicon."""
    distribution = importlib.metadata.distribution("mediapipe")
    wheel_files = [
        item for item in (distribution.files or ()) if str(item).endswith("dist-info/WHEEL")
    ]
    if not wheel_files:
        print("FAIL  mediapipe wheel tag  WHEEL metadata not found")
        return False

    wheel_path = Path(distribution.locate_file(wheel_files[0]))
    tags = [
        line.removeprefix("Tag: ").strip()
        for line in wheel_path.read_text(encoding="utf-8").splitlines()
        if line.startswith("Tag: ")
    ]
    if platform.machine() == "arm64" and tags and all("x86_64" in tag for tag in tags):
        print(
            "WARN  mediapipe wheel tag  internal tag is x86_64 although the installed "
            "universal2 binaries import and construct successfully"
        )
        return True

    print(f"PASS  mediapipe wheel tag  {', '.join(tags) if tags else 'no tag declared'}")
    return True


def check_runtime_constructors() -> bool:
    """Construct CPU-capable graphs and model without webcam access or prediction."""
    try:
        import mediapipe as mp

        hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1)
        hands.close()
        holistic = mp.solutions.holistic.Holistic(static_image_mode=True)
        holistic.close()

        alphabet = importlib.import_module("asl_alphabet")
        recognizer = alphabet.Recognizer()
        input_shape = recognizer._model.input_shape
        output_shape = recognizer._model.output_shape
        recognizer.close()
    except Exception as exc:
        print(f"FAIL  constructors          {type(exc).__name__}: {exc}")
        return False

    shapes_ok = input_shape == (None, 30, 63) and output_shape == (None, 26)
    status = "PASS" if shapes_ok else "FAIL"
    print(
        f"{status}  constructors          Hands + Holistic + SignSpell; "
        f"input={input_shape}, output={output_shape}, webcam=NO, prediction=NO"
    )
    return shapes_ok


def main() -> int:
    """Run all non-recognition compatibility checks and return a shell status."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-constructors",
        action="store_true",
        help="Construct MediaPipe graphs and the SignSpell model without a webcam.",
    )
    args = parser.parse_args()

    print("Distributed ASL Phase 2 environment audit")
    print(f"Platform: {platform.platform()}")
    print(f"Machine:  {platform.machine()}")
    print("-" * 72)

    results = [check_python()]
    results.extend(check_dependency(item) for item in DEPENDENCIES)
    results.append(check_mediapipe_wheel_tag())
    results.append(check_signspell())
    if args.runtime_constructors:
        results.append(check_runtime_constructors())
    check_phrase_model()

    print("-" * 72)
    passed = sum(results)
    print(f"SUMMARY: {passed}/{len(results)} required environment checks passed")
    print("PHRASE MODEL: PENDING until the dedicated label-map audit")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
