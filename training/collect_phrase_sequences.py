from pathlib import Path
import time

import cv2
import mediapipe as mp
import numpy as np


# =========================================================
# CONFIGURATION
# =========================================================

SEQUENCE_LENGTH = 30

# 21 landmarks × 3 coordinates × 2 hands
FEATURE_COUNT = 126

DATASET_ROOT = Path("datasets/phrases")

CAMERA_INDEX = 0

COUNTDOWN_SECONDS = 3

FRAME_DELAY_MS = 30


# =========================================================
# MEDIAPIPE SETUP
# =========================================================

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def empty_hand():
    return np.zeros(
        63,
        dtype=np.float32,
    )


def hand_to_array(hand_landmarks):
    values = []

    for landmark in hand_landmarks.landmark:
        values.extend(
            [
                landmark.x,
                landmark.y,
                landmark.z,
            ]
        )

    return np.asarray(
        values,
        dtype=np.float32,
    )


def extract_two_hand_features(results):
    left_hand = empty_hand()
    right_hand = empty_hand()

    if (
        results.multi_hand_landmarks
        and results.multi_handedness
    ):

        for landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness,
        ):

            handedness_label = (
                handedness.classification[0]
                .label
                .lower()
            )

            vector = hand_to_array(
                landmarks
            )

            if handedness_label == "left":
                left_hand = vector

            elif handedness_label == "right":
                right_hand = vector

    features = np.concatenate(
        [
            left_hand,
            right_hand,
        ]
    )

    return features


def draw_detected_hands(
    frame,
    results,
):
    if not results.multi_hand_landmarks:
        return frame

    for hand_landmarks in results.multi_hand_landmarks:

        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style(),
        )

    return frame


def get_next_sample_number(
    output_folder,
):
    existing_files = list(
        output_folder.glob("*.npy")
    )

    numbers = []

    for file_path in existing_files:

        try:
            numbers.append(
                int(
                    file_path.stem
                )
            )

        except ValueError:
            continue

    if not numbers:
        return 0

    return max(numbers) + 1


def countdown(
    cap,
    hands,
    label,
    sample_number,
):
    start_time = time.time()

    while True:

        success, frame = cap.read()

        if not success:
            return False

        frame = cv2.flip(
            frame,
            1,
        )

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        results = hands.process(
            rgb
        )

        draw_detected_hands(
            frame,
            results,
        )

        elapsed = (
            time.time()
            - start_time
        )

        remaining = (
            COUNTDOWN_SECONDS
            - int(elapsed)
        )

        cv2.putText(
            frame,
            f"Phrase: {label}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Sample: {sample_number}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Starting in {remaining}",
            (20, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 255),
            3,
        )

        cv2.putText(
            frame,
            "Press Q to quit",
            (20, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.imshow(
            "ASL Phrase Dataset Collector",
            frame,
        )

        key = cv2.waitKey(
            1
        ) & 0xFF

        if key == ord("q"):
            return False

        if elapsed >= COUNTDOWN_SECONDS:
            return True


def record_sequence(
    cap,
    hands,
    label,
    sample_number,
):
    sequence = []

    while len(sequence) < SEQUENCE_LENGTH:

        success, frame = cap.read()

        if not success:
            return None

        frame = cv2.flip(
            frame,
            1,
        )

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        results = hands.process(
            rgb
        )

        features = extract_two_hand_features(
            results
        )

        sequence.append(
            features
        )

        draw_detected_hands(
            frame,
            results,
        )

        cv2.putText(
            frame,
            f"Phrase: {label}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Sample: {sample_number}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            f"Recording: {len(sequence)}/{SEQUENCE_LENGTH}",
            (20, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            "Perform the full sign movement",
            (20, 170),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        cv2.imshow(
            "ASL Phrase Dataset Collector",
            frame,
        )

        key = cv2.waitKey(
            FRAME_DELAY_MS
        ) & 0xFF

        if key == ord("q"):
            return None

    sequence = np.asarray(
        sequence,
        dtype=np.float32,
    )

    if sequence.shape != (
        SEQUENCE_LENGTH,
        FEATURE_COUNT,
    ):
        print(
            "Invalid sequence shape:",
            sequence.shape,
        )

        return None

    return sequence


# =========================================================
# MAIN DATA COLLECTION
# =========================================================

def collect_phrase(
    label,
    samples,
):
    label = (
        label.strip()
        .upper()
        .replace(" ", "_")
    )

    output_folder = (
        DATASET_ROOT
        / label
    )

    output_folder.mkdir(
        parents=True,
        exist_ok=True,
    )

    start_number = get_next_sample_number(
        output_folder
    )

    print()
    print("=" * 60)
    print("ASL PHRASE DATA COLLECTION")
    print("=" * 60)

    print(
        "Label:",
        label,
    )

    print(
        "Samples to collect:",
        samples,
    )

    print(
        "Existing samples:",
        start_number,
    )

    print(
        "Expected shape:",
        (
            SEQUENCE_LENGTH,
            FEATURE_COUNT,
        ),
    )

    print()
    print(
        "Press Q in the camera window to stop."
    )

    cap = cv2.VideoCapture(
        CAMERA_INDEX
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Could not open webcam."
        )

    saved = 0

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:

        while saved < samples:

            sample_number = (
                start_number
                + saved
            )

            ready = countdown(
                cap,
                hands,
                label,
                sample_number,
            )

            if not ready:
                break

            sequence = record_sequence(
                cap,
                hands,
                label,
                sample_number,
            )

            if sequence is None:
                break

            output_path = (
                output_folder
                / f"{sample_number:04d}.npy"
            )

            np.save(
                output_path,
                sequence,
            )

            saved += 1

            print(
                f"Saved {saved}/{samples}: "
                f"{output_path}"
            )

    cap.release()

    cv2.destroyAllWindows()

    print()
    print("=" * 60)

    print(
        f"Finished. Saved {saved} "
        f"new {label} sequences."
    )

    print(
        "Output folder:",
        output_folder,
    )

    print("=" * 60)


# =========================================================
# COMMAND LINE
# =========================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Collect two-hand ASL phrase sequences "
            "for the Distributed ASL-to-Speech system."
        )
    )

    parser.add_argument(
        "--label",
        required=True,
        help=(
            "Phrase/sign label, for example HELLO"
        ),
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=30,
        help=(
            "Number of new sequences to record"
        ),
    )

    args = parser.parse_args()

    if args.samples <= 0:

        raise SystemExit(
            "--samples must be greater than 0."
        )

    collect_phrase(
        label=args.label,
        samples=args.samples,
    )