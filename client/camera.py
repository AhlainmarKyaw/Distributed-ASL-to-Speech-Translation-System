from __future__ import annotations
import cv2
import mediapipe as mp
import numpy as np

class Camera:
    def __init__(self, index: int = 0) -> None:
        self.index = index
        self.capture = None
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.55,
        )
        self.drawer = mp.solutions.drawing_utils
        self.connections = mp.solutions.hands.HAND_CONNECTIONS

    def start(self) -> None:
        self.capture = cv2.VideoCapture(self.index)
        if not self.capture.isOpened():
            raise RuntimeError(f'Unable to open webcam index {self.index}')

    def read(self):
        if self.capture is None:
            return False, None, None
        ok, frame = self.capture.read()
        if not ok:
            return False, None, None
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)
        features = None
        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            self.drawer.draw_landmarks(frame, hand, self.connections)
            # wrist-relative, scale-normalized xyz features (21 * 3 = 63)
            pts = np.array([[p.x, p.y, p.z] for p in hand.landmark], dtype=np.float32)
            pts -= pts[0]
            scale = np.max(np.linalg.norm(pts[:, :2], axis=1))
            if scale > 1e-6:
                pts /= scale
            features = pts.reshape(-1).astype(float).tolist()
        return True, frame, features

    def stop(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.hands.close()
