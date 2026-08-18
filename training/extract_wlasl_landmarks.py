from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


INPUT_ROOT = Path("datasets/wlasl_clips")
OUTPUT_ROOT = Path("datasets/phrases")

SEQUENCE_LENGTH = 30

mp_hands = mp.solutions.hands


def empty_hand():
    return np.zeros(63, dtype=np.float32)


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

    return np.asarray(values, dtype=np.float32)


def extract_frame_features(results):

    left = empty_hand()
    right = empty_hand()

    if (
        results.multi_hand_landmarks
        and results.multi_handedness
    ):

        for landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness,
        ):

            label = (
                handedness.classification[0]
                .label
                .lower()
            )

            vector = hand_to_array(landmarks)

            if label == "left":
                left = vector

            elif label == "right":
                right = vector

    return np.concatenate([left, right])


def resize_sequence(sequence):

    sequence = np.asarray(
        sequence,
        dtype=np.float32,
    )

    if len(sequence) == 0:
        return None

    indexes = np.linspace(
        0,
        len(sequence) - 1,
        SEQUENCE_LENGTH,
    )

    indexes = np.round(
        indexes
    ).astype(int)

    return sequence[indexes]


def process_video(video_path, hands):

    cap = cv2.VideoCapture(
        str(video_path)
    )

    sequence = []

    while True:

        success, frame = cap.read()

        if not success:
            break

        rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        results = hands.process(rgb)

        features = extract_frame_features(
            results
        )

        sequence.append(features)

    cap.release()

    return resize_sequence(sequence)


def main():

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.4,
        min_tracking_confidence=0.4,
    ) as hands:

        for label_folder in INPUT_ROOT.iterdir():

            if not label_folder.is_dir():
                continue

            output_folder = (
                OUTPUT_ROOT /
                label_folder.name
            )

            output_folder.mkdir(
                parents=True,
                exist_ok=True,
            )

            video_files = list(
                label_folder.glob("*.mp4")
            )

            saved = 0

            for video_path in video_files:

                sequence = process_video(
                    video_path,
                    hands,
                )

                if sequence is None:
                    continue

                output_path = (
                    output_folder /
                    f"{video_path.stem}.npy"
                )

                np.save(
                    output_path,
                    sequence,
                )

                saved += 1

            print(
                f"{label_folder.name}: "
                f"{saved} sequences"
            )


if __name__ == "__main__":
    main()